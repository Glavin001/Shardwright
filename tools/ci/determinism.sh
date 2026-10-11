#!/usr/bin/env bash
# Determinism check (spec §13.1): bake each asset twice in separate
# processes with different thread counts and require byte-identical glTF and
# physics payloads. Writes "<file> <sha256>" lines to $HASHES (default
# determinism.sha256) so CI can compare hashes across operating systems.
# usage: tools/ci/determinism.sh [config] [assets...]
set -euo pipefail
cd "$(dirname "$0")/../.."
cfg=${1:-benchmarks/configs/fast.toml}
shift || true
assets=("$@")
[ ${#assets[@]} -eq 0 ] && assets=(ceramic_bowl glass_annealed_pane held_out/drywall_door timber_beam)
bin=./target/release/prefracture
sha() { if command -v sha256sum > /dev/null; then sha256sum | cut -d' ' -f1; else shasum -a 256 | cut -d' ' -f1; fi; }
out=$(mktemp -d)
hashes=${HASHES:-determinism.sha256}
: > "$hashes"
status=0
for a in "${assets[@]}"; do
  b=$(basename "$a")
  RAYON_NUM_THREADS=1 $bin bake --input "benchmarks/assets/$a.glb" --config "$cfg" --out "$out/t1/" > /dev/null
  RAYON_NUM_THREADS=4 $bin bake --input "benchmarks/assets/$a.glb" --config "$cfg" --out "$out/t4/" > /dev/null
  for e in glb fracphys; do
    h1=$(sha < "$out/t1/$b.$e")
    h4=$(sha < "$out/t4/$b.$e")
    echo "$b.$e $h1" >> "$hashes"
    if [ "$h1" != "$h4" ]; then echo "NON-DETERMINISTIC: $b.$e ($h1 vs $h4)"; status=1; else echo "ok $b.$e $h1"; fi
  done
done
rm -rf "$out"
exit $status
