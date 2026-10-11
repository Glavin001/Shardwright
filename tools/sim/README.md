# frac-sim: a Rapier destruction demo on a baked asset

`frac-sim` runs a rigid-body destruction simulation on a baked prefracture
asset. Shardwright's bonds feed a stress solver, and [Rapier](https://rapier.rs)
moves the pieces. It produces the README's collapse animation
(`docs/readme/collapse_two_storey.{gif,mp4}`).

It is a demo of what a runtime can do with the baked data. It is not part of
the library, and it builds on its own: it is excluded from the main workspace,
so the main build and CI do not compile Rapier.

```sh
tools/sim/make.sh                      # sim + render + captions + GIF/MP4
tools/sim/make.sh --velocity -20,0,0   # extra arguments go to frac-sim
```

`make.sh` builds `prefracture` and `frac-sim`, runs the sim on
`out/support_clusters/two_storey_building.asset.json` (baking it first if it
is missing), then:

1. renders every frame with `prefracture preview --sim`;
2. captions the frames with `compose.py` (legend, clock and live counters);
3. encodes the GIF and MP4 with ffmpeg.

It needs Python 3 with Pillow, and ffmpeg. A run takes about 2 minutes.

## What it simulates

| Piece | How |
|---|---|
| **Chunks** | The fragments of the starting level (`--level`, default 2: 1,182 solver chunks). Each chunk keeps its baked convex hulls and its baked mass. |
| **Islands** | Chunks joined by intact bonds form one rigid body, a compound of their hulls. An island bonded to the ground is fixed. The others are dynamic: they fall, tumble and collide. |
| **Stress solver** | For each fixed island, an exact sparse Cholesky solve (faer, AMD ordering) of the bond network under gravity and contact loads. Each bond has normal, shear, bending and torsion springs (the `Spec` model of `frac_validate::network`), built from the bond's exact area, second moments and materials. The factorization is cached while an island's topology does not change. |
| **Moving islands** | Solved free-free with inertial relief whenever they are hit with more than `--impact-g` times their weight. Their contact loads are balanced by the d'Alembert forces of the acceleration they cause, so free fall is stress-free but a hard landing breaks them. |
| **Breaking** | A bond's utilization is `max(Fn⁺/T + M/Mcap, Fn⁻/C, V/S)`. Capacities are the material's tensile and compressive strength × bond area × the bond's baked `strength_scale` (Weibull size effect and boundary weakening). Bending capacity is `ft · I / c`. In each round, only the bonds within 70% of an island's worst utilization break (`BREAK_FRACTION`). Then the island is re-solved, so the load re-routes before the next bonds go, as a crack advances. Islands split when their bonds break. |
| **Refinement** | Chunks in the projectile's path are replaced by their children at the next level, down to the leaf. So are moving chunks hit hard (`--refine-dv`). Their bonds are re-made through the bond hierarchy (`parent_bond` / `child_bonds`), so detail appears only where it breaks. Refined chunks are drawn orange. |

### The design check

A building designed from plain-concrete tensile strength would not stand
under its own weight: real columns and slabs carry rebar, which the bonds do
not model. So before the impact, `frac-sim` solves the intact structure under
gravity. Every bond loaded above `1 / --design-margin` of its capacity is
reinforced until it carries `--design-margin` × its self-weight load (default
2×). Child bonds inherit their parent's factor, and the interior bonds of a
refined chunk take the largest factor of its surface bonds.

The start-up log reports how many bonds were reinforced. With
`--design-margin 0`, nothing is reinforced and whatever cannot carry its own
weight breaks before the impact.

### Projectile load

The projectile is a dynamic Rapier ball. Its contacts load the structure
like any other moving body. Just before it hits, the chunks it is about to
touch also get its momentum over `--impact-time` as a load, split among
them. That is what breaks the column it hits.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--level` | 2 | starting level of the chunks |
| `--seconds` | 3.5 | simulated time |
| `--fps` | 60 | output frames per simulated second |
| `--substep-hz` | 240 | physics steps per second |
| `--projectile` | 13,1.4,8.15 | ball start (m) |
| `--velocity` | -15,0,0 | ball velocity (m/s) |
| `--radius` | 0.4 | ball radius (m) |
| `--density` | 7800 | ball density (kg/m³, steel: 2.1 t at r 0.4) |
| `--impact-time` | 0.02 | duration of the ball's momentum transfer (s) |
| `--strength-scale` | 1 | global multiplier on bond capacities |
| `--design-margin` | 2 | self-weight margin of the design check (0 = off) |
| `--refine-dv` | 3 | velocity change (m/s) that refines a hit moving chunk |
| `--rounds` | 1 | break/re-solve rounds per step under gravity alone |
| `--impact-rounds` | 8 | break/re-solve rounds per step while something hits the structure |
| `--impact-g` | 3 | contact force (× weight) above which a moving island is solved |

The environment variables `SIM_DIAG=1` (per-solve sizes and the first breaks
with their forces and capacities) and `SIM_ROUNDS=1` (breaks per round) print
diagnostics.

## Output

`OUT.json` holds `{fps, level, frames}`. Each frame has:

* `t`;
* `bodies`: `[fragment, tx, ty, tz, qx, qy, qz, qw]`, the pose of each
  fragment's centre of mass;
* `projectile`: `[x, y, z, r]`;
* the running counts `broken`, `refined` and `islands`.

`prefracture preview --input ASSET --sim OUT.json --out dir/frame.png` renders
it from a fixed camera. Use `--fit x0,y0,z0,x1,y1,z1` to frame another box.
