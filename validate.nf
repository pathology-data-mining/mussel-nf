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

workflow {
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

    Channel.empty().mix(pt_ch, h5_ch).view().collectFile(name: params.validation_result, storeDir: params.outdir)
}
