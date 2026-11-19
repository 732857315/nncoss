set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
ROOT=../..
OUT=runs
mkdir -p $OUT/models
$PY improve.py collect --model $OUT/models/school-r1.nncossllm --output $OUT/experience_r2.json --seed 2 --plan r2
$PY improve.py train --source $OUT/models/school-r1 --examples $OUT/experience_r2.json --output $OUT/models/school-r2 --anchor $ROOT/model/Qwen3-0.6B --beta 0.2 --repeat-corrected 2
$PY improve.py export --source $OUT/models/school-r2 --output $OUT/models/school-r2.nncossllm
$PY agent.py --model $OUT/models/school-r2.nncossllm --label r2
