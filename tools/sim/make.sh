#!/usr/bin/env bash
# Destruction demo: a wrecking ball through the two-storey building, simulated
# with Rapier on the baked asset's chunks, bonds and hulls (tools/sim), then
# rendered with `prefracture preview --sim` and captioned (compose.py).
#
# Needs: the baked building (benchmarks/configs/support_clusters.toml, see
# tools/explainer/make.sh), Python 3 with Pillow, and ffmpeg.
# Writes docs/readme/hero_collapse.gif (wide, minimal captions),
# docs/readme/collapse_closeup.gif (close-up camera, counters),
# docs/readme/collapse_two_storey.mp4 (wide, counters) and out/sim/ (frames).
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
# (REUSE_SIM=1 re-renders an existing out/sim/collapse.json without simulating)
if [ -z "${REUSE_SIM:-}" ] || [ ! -f out/sim/collapse.json ]; then
    tools/sim/target/release/frac-sim "$ASSET" out/sim/collapse.json "$@"
fi
# two cameras: the wide default, and a low close-up of the impact
R="./target/release/prefracture preview --input $ASSET --sim out/sim/collapse.json"
rm -rf out/sim/render out/sim/render_close && mkdir -p out/sim/render out/sim/render_close
$R --out out/sim/render/frame.png --width 960 --height 600
$R --out out/sim/render_close/frame.png --width 960 --height 600 --azimuth 55 --elevation 14 --fit=2,0,2.5,13,7,11
# frames are 60 per sim second; played at 30 fps = 2x slow motion
python3 tools/sim/compose.py out/sim/collapse.json out/sim/render out/sim/hero 2 --minimal
python3 tools/sim/compose.py out/sim/collapse.json out/sim/render out/sim/frames 2
python3 tools/sim/compose.py out/sim/collapse.json out/sim/render_close out/sim/close 2 --compact
gif() {  # frames dir, width, output (15 fps keeps the README light)
    ffmpeg -y -loglevel error -framerate 30 -i "$1/%05d.png" -vf "fps=15,scale=$2:-1:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" "$3"
}
gif out/sim/hero 720 docs/readme/hero_collapse.gif           # README hero (wide, minimal captions)
gif out/sim/close 680 docs/readme/collapse_closeup.gif       # README runtime section (close-up, counters)
ffmpeg -y -loglevel error -framerate 30 -i out/sim/frames/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 20 \
    -preset slow -movflags +faststart docs/readme/collapse_two_storey.mp4   # wide, counters
ls -la docs/readme/hero_collapse.gif docs/readme/collapse_closeup.gif docs/readme/collapse_two_storey.mp4
