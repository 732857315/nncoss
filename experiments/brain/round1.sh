set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
ROOT=../..
OUT=runs
mkdir -p $OUT/models
$PY improve.py collect --model $ROOT/model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm --output $OUT/experience_r1.json --seed 1 --plan r1
$PY improve.py train --source $ROOT/model/Qwen3-0.6B --examples $OUT/experience_r1.json --output $OUT/models/school-r1 --max-run 0
$PY improve.py export --source $OUT/models/school-r1 --output $OUT/models/school-r1.nncossllm
$PY agent.py --model $OUT/models/school-r1.nncossllm --label r1
