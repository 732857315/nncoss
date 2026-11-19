set -e
PY=/tmp/nncoss-golden/env/bin/python
D=/private/tmp/claude-501/-Users-admin-Desktop-prj2026-nncoss/bd67780e-15b5-491e-8cfc-82e09129165c/scratchpad/brain
cd $D
until grep -q ROUND1_DONE round1.log; do sleep 20; done
echo STAGE collect
$PY improve.py collect --model $D/models/school-r1.nncossllm --output $D/experience_r2.json --seed 2 --plan r2
echo STAGE train
$PY improve.py train --source $D/models/school-r1 --examples $D/experience_r2.json --output $D/models/school-r2 --anchor /Users/admin/Desktop/prj2026/nncoss/model/Qwen3-0.6B --beta 0.2 --repeat-corrected 2
echo STAGE export
$PY improve.py export --source $D/models/school-r2 --output $D/models/school-r2.nncossllm
echo STAGE eval
$PY agent.py --model $D/models/school-r2.nncossllm --label r2
echo ROUND2_DONE
