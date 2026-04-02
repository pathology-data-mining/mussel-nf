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
    import h5py
    from concurrent.futures import ThreadPoolExecutor

    def validate(file):
        try:
            f = h5py.File(file, 'r')
        except:
            print(f"{file}: read failure")
        if 'coords' not in f.keys():
            print(f"{file}: coords not found")

    with ThreadPoolExecutor(max_workers=${task.cpus}) as executor:
        executor.map(validate, "${files.join(',')}".split(","))
        executor.shutdown(wait=True)

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
    from concurrent.futures import ThreadPoolExecutor
    import torch

    def validate(file):
        try:
            torch.load(file, weights_only=True)
        except:
            print(f"{file}: unable to load weights")

    with ThreadPoolExecutor(max_workers=${task.cpus}) as executor:
        for file in "${files.join(',')}".split(","):
            executor.submit(validate, file)
    executor.shutdown(wait=True)
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
    import tarfile
    from concurrent.futures import ThreadPoolExecutor

    def validate(tar_path):
        errors = []
        try:
            with tarfile.open(tar_path) as t:
                members = t.getmembers()
            if not members:
                errors.append(f"{tar_path}: empty shard (no members)")
            elif not any(m.name.endswith('.pt') for m in members):
                names = [m.name for m in members]
                errors.append(f"{tar_path}: no .pt entries found — members: {names}")
        except Exception as exc:
            errors.append(f"{tar_path}: could not open tar — {exc}")
        return errors

    tar_paths = "${files instanceof List ? files.join(',') : files}".split(",")
    with ThreadPoolExecutor(max_workers=${task.cpus}) as executor:
        results = list(executor.map(validate, tar_paths))

    for errs in results:
        for e in errs:
            print(e)
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
