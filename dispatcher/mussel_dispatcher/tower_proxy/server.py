"""
Persistent Tower proxy for mussel-dispatcher.

NF processes connect here (-with-tower) instead of directly to the dashboard.
All requests are forwarded to the dashboard (upstream) when it is reachable,
and answered with an empty 200 OK when it is not — so a dashboard restart
never causes NF to abort a running batch.

Usage:
    python -m mussel_dispatcher.tower_proxy --upstream http://localhost:8050 --port 8049

The proxy port (e.g. 8049) should be set as tower_endpoint in the dispatcher yaml.
The dashboard port (e.g. 8050) remains the user-facing URL.
"""
from __future__ import annotations

import argparse
import http.client
import json
import logging
import socket
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

log = logging.getLogger("tower-proxy")


def _forward(upstream_netloc: str, method: str, path: str,
             headers: dict, body: bytes) -> tuple[int, bytes]:
    """Forward request to upstream; return (status, response_body).

    Returns (200, b'{}') if upstream is unreachable.
    """
    try:
        conn = http.client.HTTPConnection(upstream_netloc, timeout=5)
        fwd_headers = {k: v for k, v in headers.items()
                       if k.lower() not in ("host", "connection", "transfer-encoding")}
        conn.request(method, path, body=body or None, headers=fwd_headers)
        resp = conn.getresponse()
        status = resp.status
        data = resp.read()
        conn.close()
        return status, data
    except (ConnectionRefusedError, OSError, http.client.HTTPException, socket.timeout):
        return 200, b"{}"


def make_handler(upstream_url: str):
    parsed = urllib.parse.urlparse(upstream_url)
    upstream_netloc = parsed.netloc  # e.g. "localhost:8050"

    class ProxyHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # suppress default access log spam
            pass

        def _proxy(self, method: str):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            headers = dict(self.headers)
            status, data = _forward(upstream_netloc, method, self.path, headers, body)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._proxy("GET")

        def do_POST(self):
            self._proxy("POST")

        def do_PUT(self):
            self._proxy("PUT")

        def do_DELETE(self):
            self._proxy("DELETE")

    return ProxyHandler


def main():
    parser = argparse.ArgumentParser(description="Persistent Tower proxy for mussel-dispatcher")
    parser.add_argument("--upstream", default="http://localhost:8050",
                        help="Dashboard URL to forward Tower calls to (default: http://localhost:8050)")
    parser.add_argument("--port", type=int, default=8049,
                        help="Port to listen on (default: 8049)")
    parser.add_argument("--host", default="localhost",
                        help="Host to bind to (default: localhost)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    log.info("Tower proxy listening on %s:%d → forwarding to %s",
             args.host, args.port, args.upstream)

    handler = make_handler(args.upstream)
    server = HTTPServer((args.host, args.port), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
