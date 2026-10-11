#!/usr/bin/env bash
# Rebuild the "Why Shardwright?" explainer from scratch: comparison bakes,
# measurements (out/why/data/evidence.json), renders, the video
# (docs/readme/why_shardwright.{mp4,gif}) and the README stills.
#
# The uniform baseline is the same pipeline with an isotropic Voronoi recipe
# on every part and geometry-only clustering instead of fracture modes.
# Needs ffmpeg and the oracle-harness Python (numpy, scipy).
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo build --release -q -p frac-cli
cargo build --release -q -p frac-pipeline --example bond_geometry
B=target/release/prefracture
PY=${PREFRACTURE_PYTHON:-${FRACENV:-/opt/fracenv}/bin/python}
mkdir -p out/why/meta out/why/shardwright out/why/uniform

# uniform-baseline metadata: every part gets the isotropic Voronoi recipe
python3 - <<'PY'
import json
for a in ["rc_column", "glass_annealed_pane", "glass_tempered_pane", "timber_beam", "brick_wall_window", "two_storey_building"]:
    m = json.load(open(f"benchmarks/assets/{a}.glb.meta.json"))
    for k, v in m["parts"].items():
        v["recipe"] = None
    m["parts"].setdefault("*", {})["recipe"] = "voronoi"
    json.dump(m, open(f"out/why/meta/{a}.meta.json", "w"), indent=1)
PY

for a in rc_column glass_annealed_pane glass_tempered_pane timber_beam brick_wall_window; do
    $B bake --input benchmarks/assets/$a.glb --config benchmarks/configs/bake.toml --out out/why/shardwright
    $B bake --input benchmarks/assets/$a.glb --meta out/why/meta/$a.meta.json \
        --config benchmarks/configs/ablation_fallback.toml --out out/why/uniform
done
$B bake --input benchmarks/assets/two_storey_building.glb --meta out/why/meta/two_storey_building.meta.json \
    --config benchmarks/configs/building_lite.toml --out out/why/uniform
[ -f out/support_clusters/two_storey_building.asset.json ] ||
    $B bake --input benchmarks/assets/two_storey_building.glb --config benchmarks/configs/support_clusters.toml --out out/support_clusters

# fracture-modes demo models (manifold3d, oracle env) and random Voronoi
# fractures of the same cells for comparison
"$PY" tools/explainer/models.py
mkdir -p out/why/models/random
for spec in "bracket 2 3 5" "bone 2 3 5" "creature 9 9 9" "chair 6 6 6"; do
    set -- $spec; m=$1; shift
    $B bake --input out/why/models/$m.obj --config out/why/models/$m.toml --out out/why/models/modes
    k=0
    for n in "$@"; do
        k=$((k + 1))
        "$PY" tools/explainer/random_voronoi.py out/why/models/modes/$m.asset.json out/why/models/random/${m}_$k.json $n $((k * 7))
    done
done

# inputs steer the result: the same pane with other materials or impact point,
# the same beam with its grain rotated
mkdir -p out/why/inputs
python3 - <<'PY'
import json
pane = lambda **kw: {"parts": {"pane": dict({"material": "glass_annealed", "impact_center": [0.45, 0.65, 0.003]}, **kw)}, "connections": [], "ground_height": None}
for name, meta in (("pane_tempered", pane(material="glass_tempered")), ("pane_concrete", pane(material="concrete_c30")),
                   ("pane_corner", pane(impact_center=[0.2, 0.25, 0.003]))):
    json.dump(meta, open(f"out/why/inputs/{name}.meta.json", "w"))
json.dump({"parts": {"beam": {"material": "pine", "grain": [0.0, 1.0, 0.0]}}, "connections": [], "ground_height": None},
          open("out/why/inputs/beam_grain_y.meta.json", "w"))
PY
for v in pane_tempered pane_concrete pane_corner; do
    cp benchmarks/assets/glass_annealed_pane.glb out/why/inputs/$v.glb
    $B bake --input out/why/inputs/$v.glb --meta out/why/inputs/$v.meta.json --config benchmarks/configs/bake.toml --out out/why/inputs
done
cp benchmarks/assets/timber_beam.glb out/why/inputs/beam_grain_y.glb
$B bake --input out/why/inputs/beam_grain_y.glb --meta out/why/inputs/beam_grain_y.meta.json --config benchmarks/configs/bake.toml --out out/why/inputs

# problem slide: a random Voronoi shatter of the whole building, ignoring parts
mkdir -p out/why/data
"$PY" tools/explainer/random_voronoi.py out/support_clusters/two_storey_building.asset.json out/why/data/building_random.json 120 3 --highlight-spanning

"$PY" tools/explainer/metrics.py
python3 tools/explainer/render.py
python3 tools/explainer/render_zoom.py
python3 tools/explainer/compose.py

ffmpeg -y -loglevel error -framerate 12 -i out/why/frames/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 20 \
    -preset slow -movflags +faststart docs/readme/why_shardwright.mp4
ffmpeg -y -loglevel error -framerate 12 -i out/why/frames/%05d.png -vf "fps=10,scale=960:-1:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=112:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
    docs/readme/why_shardwright.gif
ls -la docs/readme/why_shardwright.*
