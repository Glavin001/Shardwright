#!/usr/bin/env bash
# Golden oracle regression tests: for every benchmarks/golden/<asset>/ with a
# test.json, bake the asset, score it against the FROZEN oracle solutions
# (FEM bond fidelity, Rankine crack surfaces) and compare with the committed
# baseline (expected.json). Needs only numpy/scipy, no Kratos/gmsh: minutes
# instead of hours. UPDATE=1 rewrites the baselines; STRICT=1 also enforces
# the spec targets.
#   tools/ci/golden.sh [asset ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
FRACENV="${FRACENV:-/opt/fracenv}"
PY="${PREFRACTURE_PYTHON:-$FRACENV/bin/python}"
[ -x "$PY" ] || PY=python3
bin=./target/release/prefracture
work="${GOLDEN_WORK:-$(mktemp -d)}"
assets=("$@")
if [ ${#assets[@]} -eq 0 ]; then
  for t in benchmarks/golden/*/test.json; do assets+=("$(basename "$(dirname "$t")")"); done
fi
status=0
for a in "${assets[@]}"; do
  t="benchmarks/golden/$a/test.json"
  input=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['input'])" "$t")
  config=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['config'])" "$t")
  checks=$("$PY" -c "import json,sys;print(' '.join(json.load(open(sys.argv[1])).get('oracles',['bond','crack'])))" "$t")
  echo "== $a ($input, $config: $checks)"
  out="$work/$a"; cache="$work/$a/cache"; mkdir -p "$cache"
  $bin bake --input "$input" --config "$config" --out "$out" --allow-gate-failures | tail -1
  if [[ " $checks " == *" bond "* ]]; then
    $bin validate --input "$out/$a.asset.json" --oracle-cache "$cache" > "$out/validate.md"
    grep -q "oracle: golden" "$out/validate.md" || { echo "bond oracle not served from the golden set (solid changed?)"; status=1; }
  fi
  if [[ " $checks " == *" crack "* ]]; then
    "$PY" tools/harness/crack_oracle.py --asset "$out/$a.asset.json" --cache "$cache" > "$out/crack.md"
    grep -q "oracle: golden" "$out/crack.md" || { echo "crack oracle not served from the golden set (solid changed?)"; status=1; }
  fi
  flags=()
  [ "${UPDATE:-0}" = 1 ] && flags+=(--update)
  [ "${STRICT:-0}" = 1 ] && flags+=(--strict)
  "$PY" tools/harness/golden_check.py --asset "$a" --cache "$cache" --report "$out/$a.report.json" "${flags[@]}" || status=1
done
echo "work dir: $work"
exit $status
