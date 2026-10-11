#!/usr/bin/env bash
# Destruction demo: a wrecking ball through the two-storey building, simulated
# with Rapier on the baked asset's chunks, bonds and hulls (tools/sim), then
# rendered with `prefracture preview --sim` and captioned (compose.py).
#
# Needs: the baked building (benchmarks/configs/support_clusters.toml, see
# tools/explainer/make.sh), Python 3 with Pillow, and ffmpeg.
# Writes docs/readme/collapse_two_storey.{gif,mp4} and out/sim/ (frames).
set -euo pipefail
cd "$(dirname "$0")/../.."
ASSET=out/support_clusters/two_storey_building.asset.json
[ -f "$ASSET" ] || ./target/release/prefracture bake --input benchmarks/assets/two_storey_building.glb \
    --meta benchmarks/assets/two_storey_building.glb.meta.json --config benchmarks/configs/support_clusters.toml \
    --out out/support_clusters
cargo build --release -p frac-cli
cargo build --release --manifest-path tools/sim/Cargo.toml
mkdir -p out/sim
# defaults: steel ball r 0.4 m at 15 m/s at 1.4 m height along the front
# column row; 3.5 s; bonds design-checked for 2x self-weight (README.md)
tools/sim/target/release/frac-sim "$ASSET" out/sim/collapse.json "$@"
rm -rf out/sim/render && mkdir -p out/sim/render
./target/release/prefracture preview --input "$ASSET" --sim out/sim/collapse.json \
    --out out/sim/render/frame.png --width 960 --height 600
# frames are 60 per sim second; played at 30 fps = 2x slow motion
python3 tools/sim/compose.py out/sim/collapse.json out/sim/render out/sim/frames 2
ffmpeg -y -loglevel error -framerate 30 -i out/sim/frames/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 20 \
    -preset slow -movflags +faststart docs/readme/collapse_two_storey.mp4
ffmpeg -y -loglevel error -framerate 30 -i out/sim/frames/%05d.png -vf "fps=20,scale=760:-1:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
    docs/readme/collapse_two_storey.gif
ls -la docs/readme/collapse_two_storey.*
