#!/bin/bash
# usage: code/run.sh <tag> <python args...>  -- runs python -u from the repo root, log -> logs/<tag>.log, start/end events -> experiments/COMMANDS.jsonl
TAG=$1; shift
REPO=/home/alon/Code/pile_manipulation; cd $REPO
PY=/home/alon/anaconda3/envs/pme/bin/python; LOG=experiments/EXP-0076-expert-demos/logs/$TAG.log
RID="EXP-0076/$TAG-$RANDOM$RANDOM"; SHA=$(git rev-parse --short HEAD); DIRTY=$(git status --porcelain | grep -q . && echo dirty || echo clean)
CMD="PYTHONPATH=. OMP_NUM_THREADS=4 $PY -u $*"
$PY -c "import json,sys,time;print(json.dumps(dict(ts=time.strftime('%Y-%m-%dT%H:%M:%S'),run_id=sys.argv[1],event='start',cmd=sys.argv[2],note='EXP-0076 HEAD '+sys.argv[3]+' '+sys.argv[4])))" "$RID" "$CMD" "$SHA" "$DIRTY" >> experiments/COMMANDS.jsonl
T0=$(date +%s)
PYTHONPATH=. OMP_NUM_THREADS=4 $PY -u "$@" > $LOG 2>&1; RC=$?
$PY -c "import json,sys,time;print(json.dumps(dict(ts=time.strftime('%Y-%m-%dT%H:%M:%S'),run_id=sys.argv[1],event='end',status='ok' if sys.argv[2]=='0' else 'failed',returncode=int(sys.argv[2]),seconds=int(sys.argv[3]))))" "$RID" "$RC" "$(( $(date +%s) - T0 ))" >> experiments/COMMANDS.jsonl
exit $RC
