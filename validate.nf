params.manifest_csv = null
params.results_dir = "results"
params.batch_size = 500
params.validation_result = "validation_results.txt"

process VALIDATE_H5 {
    label "parallelTask"

    input:
    path(files)

    output:
    stdout

    script:
    """
    #!/usr/bin/env python
    import h5py, sys
    from concurrent.futures import ThreadPoolExecutor

    errors = []

    def validate(file):
        try:
            with h5py.File(file, 'r') as f:
                if 'coords' not in f.keys():
                    errors.append(f"{file}: 'coords' key not found")
        except Exception as exc:
            errors.append(f"{file}: could not open — {exc}")

    with ThreadPoolExecutor(max_workers=${task.cpus}) as executor:
        list(executor.map(validate, "${files.join(',')}".split(",")))

    for e in errors:
        print(e)
    if errors:
        sys.exit(1)
    """
}

process VALIDATE_PT {
    label "parallelTask"

    input:
    path(files)

    output:
    stdout

    script:
    """
    #!/usr/bin/env python
    import sys, torch
    from concurrent.futures import ThreadPoolExecutor

    errors = []

    def validate(file):
        try:
            torch.load(file, weights_only=True)
        except Exception as exc:
            errors.append(f"{file}: could not load — {exc}")

    with ThreadPoolExecutor(max_workers=${task.cpus}) as executor:
        list(executor.map(validate, "${files.join(',')}".split(",")))

    for e in errors:
        print(e)
    if errors:
        sys.exit(1)
    """
}

// Validate WebDataset shards: each .tar must contain at least one .pt entry.
process VALIDATE_WDS_SHARDS {
    label "parallelTask"

    input:
    path(files)

    output:
    stdout

    script:
    """
    #!/usr/bin/env python
    import sys, tarfile
    from concurrent.futures import ThreadPoolExecutor

    def validate(tar_path):
        errs = []
        try:
            with tarfile.open(tar_path) as t:
                members = t.getmembers()
            if not members:
                errs.append(f"{tar_path}: empty shard (no members)")
            elif not any(m.name.endswith('.pt') for m in members):
                errs.append(f"{tar_path}: no .pt entries — members: {[m.name for m in members]}")
        except Exception as exc:
            errs.append(f"{tar_path}: could not open — {exc}")
        return errs

    tar_paths = "${files instanceof List ? files.join(',') : files}".split(",")
    with ThreadPoolExecutor(max_workers=${task.cpus}) as executor:
        results = list(executor.map(validate, tar_paths))

    errors = [e for errs in results for e in errs]
    for e in errors:
        print(e)
    if errors:
        sys.exit(1)
    """
}

workflow {
    // ── Scan dir for WDS shards (when outdir and wds.enabled are set) ─────────
    scan_dir = params.outdir ?: params.results_dir

    ch_out = Channel.empty()

    // ── Manifest-based pt / h5 validation ────────────────────────────────────
    if (params.manifest_csv) {
        manifest_ch = Channel
            .fromPath(params.manifest_csv)
            .splitCsv(header: ["slide_id", "reef_id", "key", "value"])
            .map { row ->
                file(params.results_dir).resolve(row.value)
            }
            .branch { v ->
                pt: v.extension == "pt"
                h5: v.extension == "h5"
            }

        pt_ch = VALIDATE_PT(manifest_ch.pt.buffer(size: params.batch_size, remainder: true))
        h5_ch = VALIDATE_H5(manifest_ch.h5.buffer(size: params.batch_size, remainder: true))
        ch_out = ch_out.mix(pt_ch, h5_ch)
    }

    // ── WDS shard validation ──────────────────────────────────────────────────
    if (params.wds?.enabled) {
        ch_shards = Channel
            .fromPath("${scan_dir}/wds/**/*.tar")
            .ifEmpty { error "No WDS shards found under ${scan_dir}/wds/ — was the pipeline run with wds.enabled=true?" }
            .buffer(size: params.batch_size, remainder: true)

        wds_ch = VALIDATE_WDS_SHARDS(ch_shards)
        ch_out = ch_out.mix(wds_ch)
    }

    ch_out.view().collectFile(name: params.validation_result, storeDir: scan_dir)
}
