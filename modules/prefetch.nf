/*
 * Pre-fetch a slide from S3/ECS to a local storeDir cache before the GPU job.
 *
 * Why: When slide_path is an S3 URI, the naive approach lets Nextflow stage it
 * into the process work dir just before running the GPU task.  This means:
 *   - With local executor: download is sequential with tiling/featurization,
 *     GPU sits idle during a ~400 MB download.
 *   - With cluster executor (Condor/SLURM + GPFS): head node bottlenecks on
 *     downloading the entire batch before any GPU job starts; no parallelism
 *     between downloading and computing.
 *   - No caching: re-downloaded on every retry.
 *
 * This process fixes all three by downloading on a CPU node via downloadTask,
 * storing in a persistent storeDir cache, and outputting a local path so the
 * GPU process never touches S3.
 *
 * Inputs:
 *   meta       – slide metadata map (must contain slide_id; slide_path as string)
 *   slide_path – S3/ECS URI string (val, not path — avoids Nextflow auto-staging)
 *
 * Outputs:
 *   tuple val(meta), path(slide)
 *
 * The file is cached at:
 *   params.prefetch.local_dir/<slide_id>/<filename>
 *
 * storeDir means the task is skipped on re-runs if the file already exists
 * (equivalent to -resume for downloads).
 *
 * maxForks limits concurrent downloads to avoid saturating ECS bandwidth.
 */
process PREFETCH_SLIDE {
    label "downloadTask"

    // Expose ECS/S3 credentials so `aws s3 cp` in the script can authenticate.
    // These are the same secrets used by Nextflow's internal S3 client.
    secret 'ECS_ACCESS_KEY'
    secret 'ECS_SECRET_KEY'

    maxForks params.prefetch.max_concurrent as int

    storeDir "${params.prefetch.local_dir}/${meta.slide_id}"

    input:
    tuple val(meta), val(slide_path)

    output:
    tuple val(meta), path("${file(slide_path).name}")

    script:
    filename = file(slide_path).name
    endpoint_opt = params.prefetch.s3_endpoint
        ? "--endpoint-url ${params.prefetch.s3_endpoint}"
        : ""
    """
    export AWS_ACCESS_KEY_ID=\$ECS_ACCESS_KEY
    export AWS_SECRET_ACCESS_KEY=\$ECS_SECRET_KEY
    aws s3 cp ${endpoint_opt} '${slide_path}' '${filename}'
    """
}
