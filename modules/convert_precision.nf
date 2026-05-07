process CONVERT_FEATURES_PRECISION {
    label "cpuTask"

    publishDir path: { "${params.outdir}/features/${model_type}_${target_precision}" }, mode: "${params.publish_mode}", pattern: "*.features.h5"

    input:
    tuple val(meta), val(model_type), path(src_h5, stageAs: 'src.features.h5'), val(target_precision)

    output:
    tuple val(meta), val("${model_type}_${target_precision}"), path("*.features.h5"), emit: h5

    script:
    slide_id = meta.slide_id
    """
    #!/usr/bin/env python3
    import h5py
    import numpy as np
    import ml_dtypes

    src_path = "src.features.h5"
    dst_path = "${slide_id}.features.h5"
    target = "${target_precision}"

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # Copy all datasets verbatim except features
        for key in src.keys():
            if key == "features":
                continue
            src.copy(key, dst)

        features = np.array(src["features"])

        # Handle source bfloat16 stored as |V2 opaque void
        if features.dtype.kind == "V" and features.dtype.itemsize == 2:
            features = features.view(ml_dtypes.bfloat16).astype(np.float32)

        if target == "float16":
            features = features.astype(np.float16)
            dst.create_dataset("features", data=features)
        elif target == "bfloat16":
            # h5py does not support bfloat16 natively; store raw bytes as |V2 opaque void.
            # The reader (mussel._numpy_to_torch) detects dtype.kind=="V", itemsize==2 and
            # reinterprets as bfloat16.
            features_bf16 = features.astype(ml_dtypes.bfloat16)
            dst.create_dataset("features", data=features_bf16.view(np.uint16), dtype=np.dtype("|V2"))
        else:
            dst.create_dataset("features", data=features)

        # Copy root-level attributes, overriding embedding_precision
        for attr_key, attr_val in src.attrs.items():
            dst.attrs[attr_key] = attr_val
        dst.attrs["embedding_precision"] = target
    """

    stub:
    slide_id = meta.slide_id
    """
    #!/usr/bin/env python3
    import h5py, numpy as np
    with h5py.File("${slide_id}.features.h5", "w") as f:
        f.create_dataset("features", data=np.zeros((1, 8), dtype="float32"))
        f.attrs["embedding_precision"] = "${target_precision}"
    """
}
