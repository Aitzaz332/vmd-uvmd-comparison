#!/usr/bin/env bash
# Heartbeat logger: every 60 s append one block with the state of the jobs, GPUs, memory and the user's cgroup
# (memory.events counts out-of-memory kills). If a job dies without an error message, the last block before the
# gap shows what the machine looked like. Read-only, safe to run next to the training jobs.
#   tmux new -s watch -d 'bash ~/vmd-uvmd-comparison/replication/scripts/watch_jobs.sh'
#   tail -n 20 ~/job_heartbeat.log
OUT=${1:-$HOME/job_heartbeat.log}
CG=/sys/fs/cgroup/user.slice/user-$(id -u).slice
while true; do
  {
    echo "=== $(date '+%F %T')"
    echo "job processes (main.py / uvmd.py): $(pgrep -u "$(id -un)" -fc 'experiments/astgcn/main.py|uvmd.py')"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    free -m | awk 'NR==2{print "host mem used MB",$3,"available MB",$7}'
    [ -r "$CG/memory.events" ]  && echo "user cgroup events: $(tr '\n' ' ' < "$CG/memory.events")"
    [ -r "$CG/memory.current" ] && echo "user cgroup mem.current $(cat "$CG/memory.current") max $(cat "$CG/memory.max" 2>/dev/null)"
    uptime
  } >> "$OUT" 2>&1
  sleep 60
done
