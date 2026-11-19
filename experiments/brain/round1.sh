set -e
PY=/tmp/nncoss-golden/env/bin/python
D=/private/tmp/claude-501/-Users-admin-Desktop-prj2026-nncoss/bd67780e-15b5-491e-8cfc-82e09129165c/scratchpad/brain
cd $D
while pgrep -f "agent.py --label base --conditions think-notool,think-tool" >/dev/null; do sleep 10; done
echo STAGE collect
$PY improve.py collect --model /Users/admin/Desktop/prj2026/nncoss/model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm --output $D/experience_r1.json --seed 1
echo STAGE train
$PY improve.py train --source /Users/admin/Desktop/prj2026/nncoss/model/Qwen3-0.6B --examples $D/experience_r1.json --output $D/models/school-r1
echo STAGE export
$PY improve.py export --source $D/models/school-r1 --output $D/models/school-r1.nncossllm
echo STAGE eval
$PY agent.py --model $D/models/school-r1.nncossllm --label r1
echo ROUND1_DONE
