/**
 * Opt-in task profiling (params.profile_dir) for Mussel tasks, collected under
 * <profile_dir>/<task hash>/:
 *   - the task script's start/end times and nvidia-smi samples every 2 s;
 *   - Mussel's own stage timings (model load, each slide's tiles/features, time
 *     waiting on the DataLoader vs in the model, GPU memory peak), which Mussel
 *     writes when given timingsArg(); see mussel.utils.timings;
 *   - task.json (process, tag) and the task's .command.log so far.
 * Summarize with bin/summarize_task_profiles.py, joined with the Nextflow trace
 * (whose realtime minus the script's wall time is the container start-up).
 *
 * At the top of a process's script:  ${TaskProfile.script(params.profile_dir, task)}
 * and on the Mussel command line:    ${TaskProfile.timingsArg(params.profile_dir)}
 *
 * It runs inside the task script (and container), not as beforeScript/afterScript,
 * which would shadow a profile's process.beforeScript (see CONTRIBUTING.md). An EXIT
 * trap stops the GPU sampler and collects the files however the script ends,
 * including a failure or SIGTERM, without changing its exit status. Files are
 * written to $PWD, the scratch dir when process.scratch is set.
 *
 * The task's work dir (and so its hash) isn't known when the script is rendered,
 * since the hash covers the script; Nextflow runs .command.sh by its path in the
 * work dir, so the script finds it as $(dirname "$0"), in containers as well.
 */
class TaskProfile {
    static String script(Object profileDir, Object task) {
        if (!profileDir) {
            return ''
        }
        String root = new File(profileDir.toString()).getCanonicalPath()
        String meta = groovy.json.JsonOutput.toJson([process: task.process?.toString(), tag: task.tag?.toString() ?: ''])
        return [
            '# Task profile (params.profile_dir, lib/TaskProfile.groovy)',
            '_task_profile_wd=$(dirname "$0")',
            '# <profile_dir>/<2-char hash dir><rest of hash>, matching the trace\'s hash column',
            '_task_profile_out=' + shellQuote(root) + '/"$(basename "$(dirname "$_task_profile_wd")")$(basename "$_task_profile_wd")"',
            'mkdir -p "$_task_profile_out"',
            "cat > \"\$_task_profile_out/task.json\" <<'TASK_PROFILE_JSON'",
            meta,
            'TASK_PROFILE_JSON',
            'date +%s.%N > .task_profile_start',
            '_task_profile_gpu_pid=',
            'if command -v nvidia-smi >/dev/null 2>&1; then',
            '    nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} --query-gpu=timestamp,index,name,utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 > .task_profile_gpu.csv 2>/dev/null &',
            '    _task_profile_gpu_pid=$!',
            'fi',
            '_task_profile_stop() {',
            '    set +e',
            '    date +%s.%N > .task_profile_end',
            '    [ -n "$_task_profile_gpu_pid" ] && kill "$_task_profile_gpu_pid" 2>/dev/null',
            '    for f in .task_profile_start .task_profile_end .task_profile_gpu.csv .task_timings.jsonl "$_task_profile_wd/.command.log"; do',
            '        [ -e "$f" ] && cp "$f" "$_task_profile_out/"',
            '    done',
            '    true',
            '}',
            'trap _task_profile_stop EXIT',
        ].join('\n')
    }

    /**
     * Mussel option writing stage timings to .task_timings.jsonl in the
     * directory the script runs in, or '' when not profiling. "++" adds the
     * key, so a Mussel without timings_path accepts (and ignores) it.
     */
    static String timingsArg(Object profileDir) {
        return profileDir ? "'++timings_path=.task_timings.jsonl'" : ''
    }

    static String shellQuote(String s) {
        return "'" + s.replace("'", "'\\''") + "'"
    }
}
