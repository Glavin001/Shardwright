#!/usr/bin/env bash
# Support-graph views of the two-storey building's bonds (README + explainer):
#   1. the Level-0 support graph over a ghost of the parts;
#   2. the self-weight load path on the Level-2 chunks;
#   3. the same with the column the wrecking ball hits (tools/sim) removed:
#      chunks whose bonds go over capacity in red.
# Loads come from frac-sim --analyze (exact sparse stress solve of the bond
# network, design-checked like the sim). Writes out/why/render/support_*/.
set -euo pipefail
cd "$(dirname "$0")/../.."
A=out/support_clusters/two_storey_building.asset.json
SIM=tools/sim/target/release/frac-sim
P="./target/release/prefracture preview --input $A --width 760 --height 600 --fit=-0.4,0,-0.4,8.7,6.5,8.7"
G=out/why/graph
R=out/why/render
COLUMN=f0_col_22
cargo build --release --manifest-path tools/sim/Cargo.toml
rm -rf $G && mkdir -p $G out/why/data   # preview never overwrites (it adds -2, -3)
for l in 0 2; do
    $SIM $A /dev/null --level $l --analyze out/why/data/bonds_L$l.json --remove $COLUMN
    python3 tools/explainer/bond_graph.py $A out/why/data/bonds_L$l.json $G/l$l
done
echo '{"nodes": [], "edges": []}' > $G/empty.json
$P --graph $G/empty.json --out $G/empty.png
$P --graph $G/l0_graph.json --out $G/l0_graph.png
$P --sim $G/l0_rest.json --out $G/l0_solid.png
for v in load removed; do
    rm -rf $R/support_$v && mkdir -p $R/support_$v
    $P --sim $G/l2_$v.json --out $R/support_$v/frame.png
done
rm -rf $R/support_graph && mkdir -p $R/support_graph
python3 tools/explainer/ghost.py $G/l0_solid_0000.png $G/l0_graph.png $G/empty.png $R/support_graph/000.png 0.35

# every ground-floor column removed in turn (README: support-graph section)
rm -rf $R/support_columns && mkdir -p $R/support_columns
for c in f0_col_00 f0_col_01 f0_col_02 f0_col_10 f0_col_11 f0_col_12 f0_col_20 f0_col_21 f0_col_22; do
    $SIM $A /dev/null --level 2 --analyze out/why/data/bonds_L2_$c.json --remove $c
    python3 tools/explainer/bond_graph.py $A out/why/data/bonds_L2_$c.json $G/col_$c > /dev/null
    $P --sim $G/col_${c}_removed.json --out $R/support_columns/$c.png
done
# labelled README figures and the column GIF
python3 tools/explainer/support_figs.py
