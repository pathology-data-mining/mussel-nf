/*
 * Download a single GDC (TCGA) slide using gdc-client.
 *
 * Inputs:
 *   meta – slide metadata map, must contain:
 *     file_id   – GDC file UUID
 *     file_name – original filename (e.g. TCGA-BR-A44T-01Z-00-DX1.<uuid>.svs)
 *
 * Outputs:
 *   tuple val(meta), path(slide) – the downloaded slide file
 *
 * The file is cached via storeDir at:
 *   params.download.local_dir/<file_id>/<file_name>
 *
 * storeDir means the task is skipped automatically on re-runs if the file
 * already exists at that path (equivalent to -resume for downloads).
 */
process DOWNLOAD_SLIDE {
    label "downloadTask"

    // No container — uses the gdc conda env (gdc_env.yaml) when -profile conda is active.
    // The binary path is set via params.download.gdc_client_bin (see tcga_params.yaml).
    container null

    // Limit concurrent downloads to avoid overwhelming the GDC API.
    // Override with params.download.max_concurrent if needed.
    maxForks params.download.max_concurrent as int

    // Cache downloads outside the work directory so they survive -resume and
    // are shared across pipeline runs.  The path mirrors the layout written by
    // gdc-client: <local_dir>/<file_id>/<file_name>.
    storeDir "${params.download.local_dir}/${meta.file_id}"

    errorStrategy { task.attempt <= 3 ? 'retry' : 'ignore' }
    maxRetries 3

    input:
    val meta

    output:
    tuple val(meta), path("${meta.file_name}")

    script:
    token_opt  = (params.download?.gdc_token_file)
        ? "-t ${params.download.gdc_token_file}"
        : ""
    n_conn    = params.download?.n_connections ?: 8
    jitter    = params.download?.jitter_seconds ?: 30
    """
    # Spread concurrent tasks to avoid hammering GDC's metadata API simultaneously.
    sleep \$((RANDOM % ${jitter}))

    ${params.download?.gdc_client_bin ?: "gdc-client"} download \\
        --no-related-files \\
        -n ${n_conn} \\
        -d . \\
        ${token_opt} \\
        ${meta.file_id}

    # gdc-client creates <file_id>/<file_name>; lift to CWD so Nextflow
    # can find the output and move it into storeDir.
    mv ${meta.file_id}/${meta.file_name} .
    """
}
