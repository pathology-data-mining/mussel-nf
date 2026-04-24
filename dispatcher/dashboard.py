#!/usr/bin/env python3
"""Mussel-nf Dispatcher Monitoring Dashboard.

Reads configuration from the dispatcher YAML file (same file used to run the
dispatcher) and exposes a browser-based dashboard via stdlib http.server
(no external dependencies required).

Usage:
    python dispatcher/dashboard.py dispatcher/tcga_dispatcher.yaml
    python dispatcher/dashboard.py dispatcher/tcga_dispatcher.yaml --port 8080
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Import Config / WatcherConfig from the sibling mussel-dispatcher.py
# ---------------------------------------------------------------------------
_DISPATCHER_PY = Path(__file__).parent / "mussel-dispatcher.py"
_spec = importlib.util.spec_from_file_location("mussel_dispatcher", _DISPATCHER_PY)
_disp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_disp)
Config = _disp.Config
WatcherConfig = _disp.WatcherConfig

# ---------------------------------------------------------------------------
# S3 stats cache
# ---------------------------------------------------------------------------
_s3_cache: dict = {}          # model → {shards, objects, ts}
_S3_CACHE_TTL = 300           # seconds — ECS listing is slow, cache for 5 min


def _s3_stats(watcher: WatcherConfig, wds_prefix: str) -> dict[str, dict]:
    """Return {model: {shards, objects}} from ECS, with 60-second cache."""
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError

    results: dict[str, dict] = {}
    now = time.time()

    client_kwargs: dict = {}
    endpoint = watcher.s3_endpoint or os.environ.get("ECS_ENDPOINT_URL")
    ak = watcher.s3_access_key or os.environ.get("ECS_ACCESS_KEY")
    sk = watcher.s3_secret_key or os.environ.get("ECS_SECRET_KEY")
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    if ak:
        client_kwargs["aws_access_key_id"] = ak
    if sk:
        client_kwargs["aws_secret_access_key"] = sk
    # Short timeouts so a slow ECS response doesn't block the dashboard
    client_kwargs["config"] = BotoConfig(
        connect_timeout=5, read_timeout=10, retries={"max_attempts": 1}
    )

    for model, dest in watcher.wds_destinations.items():
        cached = _s3_cache.get(model)
        if cached and (now - cached["ts"]) < _S3_CACHE_TTL:
            results[model] = {k: v for k, v in cached.items() if k != "ts"}
            continue

        # dest is like s3://bucket/prefix
        if dest.startswith("s3://"):
            rest = dest[5:]
            bucket, _, prefix = rest.partition("/")
        else:
            results[model] = {"shards": 0, "objects": 0, "error": "not an s3 path"}
            continue

        try:
            s3 = boto3.client("s3", **client_kwargs)
            paginator = s3.get_paginator("list_objects_v2")
            shards = objects = 0
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith(".tar"):
                        shards += 1
                    objects += 1
            entry = {"shards": shards, "objects": objects, "ts": now}
            _s3_cache[model] = entry
            results[model] = {"shards": shards, "objects": objects}
        except (BotoCoreError, ClientError, Exception) as exc:
            results[model] = {"shards": 0, "objects": 0, "error": str(exc)[:120]}

    return results


# ---------------------------------------------------------------------------
# HTTP request handler (stdlib — no FastAPI/uvicorn dependency)
# ---------------------------------------------------------------------------

import csv
import json
import re as _re
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _build_handler(cfg: Config):
    """Return a BaseHTTPRequestHandler subclass closed over cfg."""

    db_path = os.path.join(cfg.state_dir, "dispatcher.db")
    wds_manifest = os.path.join(cfg.outdir, "wds_manifest.csv")
    nf_log_path = os.path.join(cfg.repo_dir, ".nextflow.log")

    tcga_watcher = None
    for w in cfg.watchers:
        if w.wds_destinations:
            tcga_watcher = w
            break

    # Pre-warm S3 cache in background
    if tcga_watcher and tcga_watcher.wds_destinations:
        threading.Thread(
            target=_s3_stats, args=(tcga_watcher, "wds"), daemon=True, name="s3-prewarm"
        ).start()

    def _db():
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _api_status():
        with _db() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM slides GROUP BY status"
            ).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        total = sum(counts.values())
        succeeded = counts.get("SUCCEEDED", 0)
        pct = round(succeeded / total * 100, 1) if total else 0
        with _db() as conn:
            n_running = conn.execute(
                "SELECT COUNT(*) FROM batches WHERE status='RUNNING'"
            ).fetchone()[0]
            n_blacklisted = conn.execute(
                "SELECT COUNT(*) FROM slides WHERE fail_count >= 100"
            ).fetchone()[0]
        return {
            "counts": counts,
            "total": total,
            "pct_done": pct,
            "running_batches": n_running,
            "blacklisted": n_blacklisted,
        }

    def _api_batches():
        with _db() as conn:
            rows = conn.execute("""
                SELECT batch_id, status, slide_count, dispatched_at,
                       completed_at, nextflow_exit, log_path
                FROM batches
                ORDER BY dispatched_at DESC
                LIMIT 30
            """).fetchall()
        result = []
        for r in rows:
            start = r["dispatched_at"] or ""
            end = r["completed_at"] or ""
            duration = None
            if start and end:
                try:
                    def _parse(s):
                        try:
                            return datetime.fromisoformat(s)
                        except Exception:
                            return None
                    t0, t1 = _parse(start), _parse(end)
                    if t0 and t1:
                        duration = int((t1 - t0).total_seconds())
                except Exception:
                    pass
            result.append({
                "batch_id": r["batch_id"],
                "status": r["status"],
                "slide_count": r["slide_count"],
                "dispatched_at": start,
                "completed_at": end,
                "duration_s": duration,
                "nextflow_exit": r["nextflow_exit"],
                "has_log": bool(r["log_path"] and os.path.exists(r["log_path"])),
            })
        return result

    def _api_logs(batch_id: str):
        with _db() as conn:
            row = conn.execute(
                "SELECT log_path FROM batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
        if not row or not row["log_path"]:
            return None, "no log path"
        log_path = row["log_path"]
        if not os.path.exists(log_path):
            return None, "log file not found"
        try:
            ansi = _re.compile(r'\x1b\[[0-9;]*[mABCDEFGHJKSTfnsuhl]|\r')
            with open(log_path, "r", errors="replace") as fh:
                content = fh.read()
            content = ansi.sub('', content)
            last = content.rfind("executor >")
            block = content[last:].strip() if last >= 0 else content.strip()
            lines = block.splitlines()
        except Exception as exc:
            return None, str(exc)
        nf_lines: list = []
        if os.path.exists(nf_log_path):
            try:
                with open(nf_log_path, "r", errors="replace") as fh:
                    nf_lines = [ln.rstrip() for ln in fh.readlines()[-40:]]
            except Exception:
                pass
        return {"lines": lines, "nf_log": nf_lines}, None

    def _api_wds():
        # WDS manifest counts
        wds_counts: dict = {}
        if os.path.exists(wds_manifest):
            try:
                with open(wds_manifest, newline="") as f:
                    for row in csv.DictReader(f):
                        model = row.get("model", "unknown")
                        wds_counts[model] = wds_counts.get(model, 0) + 1
            except Exception as exc:
                return {"models": {}, "total": 0, "error": str(exc)}
        # S3 shard stats (from cache; refresh happens in background)
        s3_stats: dict = {}
        if tcga_watcher and tcga_watcher.wds_destinations:
            s3_stats = _s3_stats(tcga_watcher, "wds")

        models: dict = {}
        all_keys = set(wds_counts) | set(s3_stats)
        for m in sorted(all_keys):
            models[m] = {
                "slides": wds_counts.get(m, 0),
                "shards": s3_stats.get(m, {}).get("shards", 0),
                "error": s3_stats.get(m, {}).get("error"),
            }
        return {"models": models, "total": sum(wds_counts.values())}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # suppress default access log spam
            pass

        def _send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str):
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path in ("/", "/index.html"):
                    self._send_html(_HTML)
                elif path == "/api/status":
                    self._send_json(_api_status())
                elif path == "/api/batches":
                    self._send_json(_api_batches())
                elif path.startswith("/api/logs/"):
                    batch_id = path[len("/api/logs/"):]
                    data, err = _api_logs(batch_id)
                    if data is None:
                        self._send_json({"lines": [], "error": err}, status=404)
                    else:
                        self._send_json(data)
                elif path == "/api/wds":
                    self._send_json(_api_wds())
                else:
                    self.send_response(404)
                    self.end_headers()
            except Exception as exc:
                try:
                    self._send_json({"error": str(exc)}, status=500)
                except Exception:
                    pass

    return Handler


# ---------------------------------------------------------------------------
# Embedded HTML + JS dashboard
# ---------------------------------------------------------------------------
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mussel Dispatcher Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js" async
  onload="onChartJsReady()"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  header { background: #1e293b; border-bottom: 1px solid #334155; padding: 12px 24px;
           display: flex; justify-content: space-between; align-items: center; }
  header h1 { font-size: 1.2rem; font-weight: 600; color: #f8fafc; }
  #last-updated { font-size: 0.75rem; color: #94a3b8; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
  .full { grid-column: 1 / -1; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 0.85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;
             letter-spacing: .05em; margin-bottom: 12px; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
  .stat { background: #0f172a; border-radius: 6px; padding: 12px 16px; }
  .stat .val { font-size: 1.8rem; font-weight: 700; }
  .stat .lbl { font-size: 0.7rem; color: #94a3b8; margin-top: 2px; text-transform: uppercase; }
  .chart-wrap { display: flex; justify-content: center; align-items: center; max-height: 220px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  th { text-align: left; color: #64748b; font-weight: 600; padding: 6px 8px;
       border-bottom: 1px solid #334155; text-transform: uppercase; font-size: 0.7rem; }
  td { padding: 6px 8px; border-bottom: 1px solid #1e293b; vertical-align: middle; }
  tr:hover td { background: #0f172a; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.7rem; font-weight: 600; }
  .badge-running   { background:#0c4a6e; color:#38bdf8; }
  .badge-succeeded { background:#14532d; color:#4ade80; }
  .badge-failed    { background:#450a0a; color:#f87171; }
  .badge-pending   { background:#312e81; color:#a5b4fc; }
  .log-block { background: #0f172a; border-radius: 6px; padding: 10px 12px;
               font-family: monospace; font-size: 0.72rem; line-height: 1.5;
               max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-all;
               color: #94a3b8; margin-bottom: 8px; }
  .log-header { font-size: 0.75rem; color: #e2e8f0; margin-bottom: 4px; font-weight: 600; }
  .progress-bar-wrap { background: #334155; border-radius: 999px; height: 8px; margin-top: 4px; }
  .progress-bar { background: #22c55e; border-radius: 999px; height: 8px; transition: width .5s; }
  .wds-row { display: flex; justify-content: space-between; padding: 4px 0;
             border-bottom: 1px solid #334155; font-size: 0.8rem; }
  .wds-row:last-child { border-bottom: none; }
  .no-data { color: #475569; font-size: 0.8rem; }
  details summary { cursor: pointer; padding: 4px 0; color: #94a3b8; font-size: 0.8rem; }
  details summary:hover { color: #e2e8f0; }
  details[open] summary { color: #38bdf8; }
  .btn-refresh { background: #334155; border: none; color: #94a3b8; padding: 4px 12px;
                border-radius: 4px; cursor: pointer; font-size: 0.75rem; }
  .btn-refresh:hover { background: #475569; color: #e2e8f0; }
  /* Log modal */
  #log-modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,.7);
               z-index:1000; align-items:center; justify-content:center; }
  #log-modal.open { display:flex; }
  #log-modal-box { background:#1e293b; border:1px solid #334155; border-radius:10px;
                   width:90vw; max-width:1100px; max-height:85vh; display:flex; flex-direction:column; }
  #log-modal-header { display:flex; justify-content:space-between; align-items:center;
                      padding:12px 16px; border-bottom:1px solid #334155; }
  #log-modal-title { font-size:0.85rem; font-weight:600; color:#f8fafc; font-family:monospace; }
  #log-modal-close { background:none; border:none; color:#94a3b8; font-size:1.4rem;
                     cursor:pointer; line-height:1; padding:0 4px; }
  #log-modal-close:hover { color:#f8fafc; }
  #log-modal-body { flex:1; overflow-y:auto; padding:12px 16px;
                    font-family:monospace; font-size:0.72rem; line-height:1.5;
                    white-space:pre-wrap; word-break:break-all; color:#94a3b8;
                    background:#0f172a; border-radius:0 0 10px 10px; }
</style>
</head>
<body>
<div id="log-modal">
  <div id="log-modal-box">
    <div id="log-modal-header">
      <span id="log-modal-title"></span>
      <button id="log-modal-close" onclick="closeLogModal()">✕</button>
    </div>
    <pre id="log-modal-body">Loading…</pre>
  </div>
</div>
<header>
  <h1>🦪 Mussel Dispatcher Dashboard</h1>
  <div style="display:flex;gap:12px;align-items:center">
    <span id="last-updated">Loading…</span>
    <button class="btn-refresh" onclick="refresh()">↻ Refresh</button>
  </div>
</header>

<div class="grid">
  <!-- Summary -->
  <div class="card full">
    <h2>Overview</h2>
    <div class="summary-grid" id="summary-grid">
      <div class="stat"><div class="val" id="s-total">—</div><div class="lbl">Total Slides</div></div>
      <div class="stat"><div class="val" id="s-done" style="color:#4ade80">—</div><div class="lbl">Succeeded</div></div>
      <div class="stat"><div class="val" id="s-failed" style="color:#f87171">—</div><div class="lbl">Failed</div></div>
      <div class="stat"><div class="val" id="s-pending" style="color:#a5b4fc">—</div><div class="lbl">Pending</div></div>
      <div class="stat"><div class="val" id="s-dispatched" style="color:#38bdf8">—</div><div class="lbl">Dispatched</div></div>
      <div class="stat"><div class="val" id="s-pct">—</div><div class="lbl">% Done</div></div>
      <div class="stat"><div class="val" id="s-running">—</div><div class="lbl">Running Batches</div></div>
      <div class="stat"><div class="val" id="s-blacklisted" style="color:#fb923c">—</div><div class="lbl">Blacklisted</div></div>
    </div>
    <div class="progress-bar-wrap" style="margin-top:12px">
      <div class="progress-bar" id="progress-bar" style="width:0%"></div>
    </div>
  </div>

  <!-- Slide status chart -->
  <div class="card">
    <h2>Slide Status</h2>
    <div class="chart-wrap"><canvas id="statusChart"></canvas></div>
  </div>

  <!-- WDS + S3 -->
  <div class="card">
    <h2>WDS</h2>
    <div id="wds-content"><span class="no-data">Loading…</span></div>
  </div>

  <!-- Batch table -->
  <div class="card full">
    <h2>Recent Batches</h2>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Batch ID</th>
            <th>Status</th>
            <th>Slides</th>
            <th>Started</th>
            <th>Duration</th>
            <th>NF Exit</th>
            <th>Log</th>
          </tr>
        </thead>
        <tbody id="batch-tbody"><tr><td colspan="7" class="no-data">Loading…</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- Log tails -->
  <div class="card full" id="logs-card">
    <h2>Live Logs (Running Batches)</h2>
    <div id="logs-content"><span class="no-data">No running batches.</span></div>
  </div>

  <!-- .nextflow.log tail -->
  <div class="card full" id="nflog-card">
    <h2>.nextflow.log <span style="font-weight:400;color:#475569;font-size:0.75rem">(shared, last 40 lines)</span></h2>
    <pre id="nflog-content" style="background:#0f172a;border-radius:6px;padding:10px 12px;font-size:0.7rem;line-height:1.5;color:#64748b;max-height:220px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;margin:0">Loading…</pre>
  </div>
</div>

<script>
let statusChart = null;
let _chartJsReady = false;

function onChartJsReady() {
  _chartJsReady = true;
  // Draw chart with latest data if already loaded
  loadStatus();
}

function badge(status) {
  const cls = {RUNNING:'running',SUCCEEDED:'succeeded',FAILED:'failed',PENDING:'pending',DISPATCHED:'running'}[status] || 'pending';
  return `<span class="badge badge-${cls}">${status}</span>`;
}

function fmtDuration(s) {
  if (s === null || s === undefined) return '—';
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s/60) + 'm ' + (s%60) + 's';
  return Math.floor(s/3600) + 'h ' + Math.floor((s%3600)/60) + 'm';
}

function fmtTime(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString();
  } catch { return iso; }
}

async function apiFetch(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}

async function loadStatus() {
  try {
    const d = await apiFetch('/api/status');
    const c = d.counts || {};
    document.getElementById('s-total').textContent = d.total ?? '—';
    document.getElementById('s-done').textContent = c.SUCCEEDED ?? 0;
    document.getElementById('s-failed').textContent = c.FAILED ?? 0;
    document.getElementById('s-pending').textContent = c.PENDING ?? 0;
    document.getElementById('s-dispatched').textContent = c.DISPATCHED ?? 0;
    document.getElementById('s-pct').textContent = (d.pct_done ?? 0) + '%';
    document.getElementById('s-running').textContent = d.running_batches ?? 0;
    document.getElementById('s-blacklisted').textContent = d.blacklisted ?? 0;
    document.getElementById('progress-bar').style.width = (d.pct_done ?? 0) + '%';

    const labels = Object.keys(c);
    const values = Object.values(c);
    const colors = labels.map(l => ({
      SUCCEEDED:'#22c55e', FAILED:'#ef4444', PENDING:'#6366f1', DISPATCHED:'#0ea5e9'
    }[l] || '#64748b'));

    if (typeof Chart === 'undefined' || !_chartJsReady) return; // Chart.js not yet loaded
    if (!statusChart) {
      statusChart = new Chart(document.getElementById('statusChart'), {
        type: 'doughnut',
        data: { labels, datasets: [{ data: values, backgroundColor: colors, borderWidth: 0 }] },
        options: { plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
                   cutout: '60%' }
      });
    } else {
      statusChart.data.labels = labels;
      statusChart.data.datasets[0].data = values;
      statusChart.data.datasets[0].backgroundColor = colors;
      statusChart.update();
    }
  } catch(e) {
    document.getElementById('s-total').textContent = 'ERR';
    console.error('loadStatus failed:', e);
  }
}

async function loadBatches() {
  const tbody = document.getElementById('batch-tbody');
  const logsDiv = document.getElementById('logs-content');
  try {
    const batches = await apiFetch('/api/batches');
    if (!batches.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="no-data">No batches yet.</td></tr>';
      logsDiv.innerHTML = '<span class="no-data">No running batches.</span>';
      return;
    }
    tbody.innerHTML = batches.map(b => `
      <tr>
        <td style="font-family:monospace;font-size:0.7rem">${b.batch_id}</td>
        <td>${badge(b.status)}</td>
        <td>${b.slide_count ?? '—'}</td>
        <td>${fmtTime(b.dispatched_at)}</td>
        <td>${fmtDuration(b.duration_s)}</td>
        <td>${b.nextflow_exit !== null && b.nextflow_exit !== undefined ? b.nextflow_exit : '—'}</td>
        <td>${b.has_log ? `<button class="btn-refresh" onclick="showLog('${b.batch_id}')">View</button>` : '—'}</td>
      </tr>`).join('');

    const running = batches.filter(b => b.status === 'RUNNING');
    if (!running.length) {
      logsDiv.innerHTML = '<span class="no-data">No running batches.</span>';
      return;
    }

    // Add panels for new running batches; keep existing ones (update in-place)
    const existingIds = new Set([...logsDiv.querySelectorAll('[data-batch]')].map(el => el.dataset.batch));
    const runningIds = new Set(running.map(b => b.batch_id));

    // Remove panels for batches no longer running
    logsDiv.querySelectorAll('[data-batch]').forEach(el => {
      if (!runningIds.has(el.dataset.batch)) el.remove();
    });
    if (!logsDiv.querySelector('[data-batch]')) logsDiv.innerHTML = '';

    // Add new panels
    for (const b of running) {
      if (!existingIds.has(b.batch_id)) {
        const card = document.createElement('div');
        card.dataset.batch = b.batch_id;
        card.style.marginBottom = '12px';
        card.innerHTML = `
          <div class="log-header" style="display:flex;justify-content:space-between">
            <span style="font-family:monospace">${b.batch_id} <span style="color:#475569">(${b.slide_count} slides)</span></span>
            <button class="btn-refresh" onclick="showLog('${b.batch_id}')">Full log</button>
          </div>
          <pre class="log-block" id="logtext-${b.batch_id}" style="max-height:160px">Loading…</pre>`;
        logsDiv.appendChild(card);
      }
    }

    // Refresh log content for all running batches
    running.forEach(b => refreshInlineLog(b.batch_id));
  } catch(e) {
    tbody.innerHTML = '<tr><td colspan="7" class="no-data">Failed to load batches.</td></tr>';
    console.error('loadBatches failed:', e);
  }
}

async function refreshInlineLog(batch_id) {
  const el = document.getElementById('logtext-' + batch_id);
  if (!el) return;
  try {
    const d = await apiFetch('/api/logs/' + batch_id);
    const text = (d.lines || []).join('\\n') || d.error || '(empty)';
    el.textContent = text;
    el.scrollTop = el.scrollHeight;
    // Update shared .nextflow.log panel (last writer wins, that's fine)
    if (d.nf_log && d.nf_log.length) {
      const nfEl = document.getElementById('nflog-content');
      if (nfEl) {
        const atBottom = nfEl.scrollHeight - nfEl.scrollTop <= nfEl.clientHeight + 20;
        nfEl.textContent = d.nf_log.join('\\n');
        if (atBottom) nfEl.scrollTop = nfEl.scrollHeight;
      }
    }
  } catch { if (el) el.textContent = 'Failed to load log.'; }
}

let _modalBatchId = null;
let _modalInterval = null;

function openLogModal(batch_id) {
  _modalBatchId = batch_id;
  document.getElementById('log-modal-title').textContent = batch_id;
  document.getElementById('log-modal-body').textContent = 'Loading…';
  document.getElementById('log-modal').classList.add('open');
  refreshModalLog();
  clearInterval(_modalInterval);
  _modalInterval = setInterval(refreshModalLog, 5000);
}

function closeLogModal() {
  document.getElementById('log-modal').classList.remove('open');
  clearInterval(_modalInterval);
  _modalInterval = null;
  _modalBatchId = null;
}

async function refreshModalLog() {
  if (!_modalBatchId) return;
  const el = document.getElementById('log-modal-body');
  try {
    const d = await apiFetch('/api/logs/' + _modalBatchId);
    const text = (d.lines || []).join('\\n') + (d.nf_log && d.nf_log.length ? '\\n\\n--- .nextflow.log ---\\n' + d.nf_log.join('\\n') : '');
    const atBottom = el.scrollHeight - el.scrollTop <= el.clientHeight + 20;
    el.textContent = text || d.error || '(empty)';
    if (atBottom) el.scrollTop = el.scrollHeight;
  } catch { el.textContent = 'Failed to load log.'; }
}

// Close modal on backdrop click
document.getElementById('log-modal').addEventListener('click', function(e) {
  if (e.target === this) closeLogModal();
});

async function showLog(batch_id) { openLogModal(batch_id); }

async function loadWds() {
  const el = document.getElementById('wds-content');
  try {
    const [wds, s3] = await Promise.allSettled([apiFetch('/api/wds'), apiFetch('/api/s3')]);
    const slides = (wds.status === 'fulfilled' && wds.value.models) ? wds.value.models : {};
    const shards = (s3.status === 'fulfilled' && s3.value.models) ? s3.value.models : {};
    const models = new Set([...Object.keys(slides), ...Object.keys(shards)]);
    if (!models.size) { el.innerHTML = '<span class="no-data">No WDS data yet.</span>'; return; }
    const rows = [...models].map(m => {
      const n = slides[m] || 0;
      const sv = shards[m];
      let shardStr = sv
        ? (sv.error ? `<span style="color:#f87171">${sv.error}</span>`
                    : `<span style="color:#38bdf8">${sv.shards} shard${sv.shards === 1 ? '' : 's'}</span>`)
        : '<span style="color:#6b7280">–</span>';
      return `<div class="wds-row">
        <span>${m}</span>
        <span><span style="color:#4ade80;font-weight:600">${n} slides</span> &nbsp;·&nbsp; ${shardStr}</span>
      </div>`;
    });
    const total = Object.values(slides).reduce((a,b) => a+b, 0);
    if (total) rows.push(`<div class="wds-row" style="margin-top:4px;border-top:1px solid #334155"><span>Total slides</span><b>${total}</b></div>`);
    el.innerHTML = rows.join('');
  } catch(e) {
    el.innerHTML = '<span class="no-data">Failed to load WDS data.</span>';
    console.error('loadWds failed:', e);
  }
}

async function loadS3() { /* merged into loadWds */ }

async function refresh() {
  document.getElementById('last-updated').textContent = 'Refreshing…';
  await Promise.allSettled([loadStatus(), loadBatches(), loadWds()]);
  document.getElementById('last-updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

refresh();
setInterval(refresh, 10000);
setInterval(loadWds, 60000);  // S3 stats are expensive — refresh less often via loadWds
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to dispatcher YAML config file")
    parser.add_argument("--port", type=int, default=8050,
                        help="HTTP port to listen on (default: 8050)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    cfg = Config.load(args.config)

    # If ECS credentials aren't in env, try to read them from Nextflow secrets
    for env_key, nf_secret in [("ECS_ACCESS_KEY", "ECS_ACCESS_KEY"),
                                ("ECS_SECRET_KEY", "ECS_SECRET_KEY")]:
        if not os.environ.get(env_key):
            try:
                val = subprocess.check_output(
                    ["nextflow", "secrets", "get", nf_secret],
                    text=True, stderr=subprocess.DEVNULL, timeout=5,
                ).strip()
                if val:
                    os.environ[env_key] = val
            except Exception:
                pass

    handler_cls = _build_handler(cfg)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Dashboard: http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
