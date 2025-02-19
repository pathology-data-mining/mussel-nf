params.results_dir = "results"

process VALIDATE_H5 {
    input:
    tuple val(slide_id), val(file_type), path(file)

    script:
    """
    #!/usr/bin/env python

    import h5py
    f = h5py.File("${file}", 'r')
    assert 'coords' in f.keys()
    """
}

process VALIDATE_PT {
    input:
    tuple val(slide_id), val(file_type), path(file)

    script:
    """
    #!/usr/bin/env python

    import torch
    torch.load("${file}", weights_only=True)
    """
}

workflow {
    manifest_ch = Channel
        .fromPath(params.manifest_csv)
        .splitCsv(header: ["slide_id", "reef_id", "key", "value"])
        .map { row ->
            tuple(row.slide_id, row.key, file(params.results_dir).resolve(row.value))
        }
        .branch { v ->
            pt: v[2].extension == "pt"
            h5: v[2].extension == "h5"
        }

    // manifest_ch.h5.view()
    VALIDATE_PT(manifest_ch.pt)
    VALIDATE_H5(manifest_ch.h5)
}
