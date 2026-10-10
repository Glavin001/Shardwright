#!/usr/bin/env bash
# Bake the benchmark suite with one config and summarize per asset: wall
# time, peak RSS, gate verdict (failing gates listed) and the end-to-end
# time budget.
#
#   tools/ci/suite.sh [config] [asset ...]
#
# Defaults: benchmarks/configs/bake.toml, every benchmarks/assets/*.glb.
# HELD_OUT=1 adds benchmarks/assets/held_out/*.glb. BUDGET_S (default 300)
# is the per-asset end-to-end budget; OUT (default out/suite) keeps the
# outputs and logs (KEEP_OUTPUTS=0 deletes each asset's payloads after its
# row is recorded, keeping reports and logs). Exits non-zero if any asset
# fails a gate or the budget.
set -euo pipefail
cd "$(dirname "$0")/../.."
config=${1:-benchmarks/configs/bake.toml}
shift || true
budget=${BUDGET_S:-300}
out=${OUT:-out/suite}
mkdir -p "$out"
if [ $# -gt 0 ]; then
    assets=("$@")
else
    assets=()
    for f in benchmarks/assets/*.glb; do assets+=("$(basename "$f" .glb)"); done
    if [ "${HELD_OUT:-0}" = 1 ]; then
        for f in benchmarks/assets/held_out/*.glb; do assets+=("held_out/$(basename "$f" .glb)"); done
    fi
fi
cargo build --release -q -p frac-cli
bin=target/release/prefracture
summary="$out/summary.md"
echo "| asset | time (s) | peak RSS (MB) | gates | budget |" > "$summary"
echo "|---|---|---|---|---|" >> "$summary"
status=0
for a in "${assets[@]}"; do
    name=$(basename "$a")
    log="$out/$name.log"
    t0=$(date +%s.%N)
    # peak RSS of the bake process, sampled from /proc (no /usr/bin/time needed)
    FRAC_LOG=1 "$bin" bake --input "benchmarks/assets/$a.glb" --config "$config" --out "$out/" > "$log" 2>&1 &
    pid=$!
    peak=0
    while kill -0 "$pid" 2> /dev/null; do
        r=$(awk '/VmHWM/ {print int($2 / 1024)}' "/proc/$pid/status" 2> /dev/null || echo 0)
        [ -n "$r" ] && [ "$r" -gt "$peak" ] && peak=$r
        sleep 0.5
    done
    rc=0
    wait "$pid" || rc=$?
    t=$(echo "$(date +%s.%N) - $t0" | bc)
    if [ "$rc" = 0 ]; then
        gates=PASS
    else
        fails=$(grep -o '| [a-z_]* | \*\*FAIL\*\*' "$out/$name.report.md" 2> /dev/null | awk '{print $2}' | paste -sd, - || true)
        gates="FAIL (${fails:-exit $rc})"
        status=1
    fi
    if (($(echo "$t > $budget" | bc))); then
        verdict="over ($budget s)"
        status=1
    else
        verdict=ok
    fi
    printf '| %s | %.1f | %s | %s | %s |\n' "$name" "$t" "$peak" "$gates" "$verdict" >> "$summary"
    tail -1 "$summary"
    if [ "${KEEP_OUTPUTS:-1}" = 0 ]; then
        rm -f "$out/$name.glb" "$out/$name.fracphys" "$out/$name.asset.json"
    fi
done
cat "$summary"
exit $status
