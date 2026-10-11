"""Procedural test models for the fracture-modes demo, in the spirit of the
figures in Sellán et al. 2023 (a bracket with a clear weakness, a long bone,
a many-armed creature, a chair). Built with manifold3d so every mesh
is watertight; written as OBJ + metadata to out/why/models/."""
import json, math, os
import numpy as np
from manifold3d import Manifold

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
D = ROOT + "/out/why/models"
os.makedirs(D, exist_ok=True)

def box(sx, sy, sz, x=0.0, y=0.0, z=0.0):
    """Axis-aligned box of size (sx, sy, sz) with its min corner's x/z centred
    and its bottom at y."""
    return Manifold.cube([sx, sy, sz]).translate([x - sx / 2, y, z - sz / 2])

def capsule(p0, p1, r0, r1, seg=24):
    """Tapered cylinder from p0 to p1 (radii r0, r1) with spherical caps."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0; L = np.linalg.norm(d)
    c = Manifold.cylinder(L, r0, r1, seg)                    # along +z from the origin
    z = np.array([0, 0, 1.0]); u = d / L
    ax = np.cross(z, u); s = np.linalg.norm(ax)
    if s > 1e-9:
        ang = math.degrees(math.atan2(s, z @ u)); ax /= s
        # rotate about ax by ang: manifold3d rotates by Euler angles, so use a matrix transform
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        a = math.radians(ang); Rm = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
        c = c.transform([[Rm[0, 0], Rm[0, 1], Rm[0, 2], p0[0]], [Rm[1, 0], Rm[1, 1], Rm[1, 2], p0[1]], [Rm[2, 0], Rm[2, 1], Rm[2, 2], p0[2]]])
    else:
        c = c.translate(p0.tolist())
    return c + Manifold.sphere(r0, seg).translate(p0.tolist()) + Manifold.sphere(r1, seg).translate(p1.tolist())

def save(name, m, material="concrete_c30", extra=None):
    mesh = m.to_mesh()
    V = np.asarray(mesh.vert_properties)[:, :3]; F = np.asarray(mesh.tri_verts)
    with open(f"{D}/{name}.obj", "w") as f:
        for v in V: f.write("v %.6f %.6f %.6f\n" % tuple(v))
        for t in F: f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))
    part = {"material": material}
    part.update(extra or {})
    json.dump({"parts": {"*": part}, "connections": [], "ground_height": None}, open(f"{D}/{name}.obj.meta.json", "w"))
    print(name, len(V), "verts", len(F), "tris", "volume %.4f" % m.volume())

# 1. bracket with a clear weakness: two plates with bosses joined by a slender
#    rod that has a narrow neck (off-centre, so a balanced split misses it)
rod, neck = 0.07, 0.03
m = (box(0.55, 0.07, 0.55) + box(0.2, 0.2, 0.2, y=0.07)                       # bottom plate + boss
     + box(rod, 0.32, rod, y=0.27) + box(neck, 0.04, neck, y=0.59)            # rod up to the neck
     + box(rod, 0.62, rod, y=0.63)                                            # rod above the neck
     + box(0.2, 0.2, 0.2, y=1.25) + box(0.55, 0.07, 0.55, y=1.45))            # top boss + plate
save("bracket", m)

# 2. long bone: a slender shaft with heavy ends (head + neck at the top,
#    two condyles at the bottom)
shaft = capsule([0, 0.12, 0], [0.02, 0.86, 0], 0.034, 0.03, 32)
head = Manifold.sphere(0.07, 40).translate([-0.07, 0.93, 0]) + capsule([0.02, 0.84, 0], [-0.06, 0.92, 0], 0.035, 0.045, 32)
troch = Manifold.sphere(0.05, 32).translate([0.06, 0.88, 0])
condyles = Manifold.sphere(0.065, 40).translate([-0.045, 0.06, 0]) + Manifold.sphere(0.065, 40).translate([0.05, 0.06, 0])
save("bone", shaft + head + troch + condyles, "ceramic")

# 3. many-armed creature: a round body with eight curling, tapering arms
body = Manifold.sphere(0.22, 48).translate([0, 0.3, 0])
arms = Manifold()
for k in range(8):
    a = 2 * math.pi * k / 8 + 0.2
    pts = []
    for j in range(7):
        t = j / 6
        r = 0.16 + 0.55 * t
        pts.append([r * math.cos(a + 0.5 * t * t), 0.22 - 0.2 * t + 0.08 * math.sin(3 * t + k), r * math.sin(a + 0.5 * t * t)])
    for j in range(6):
        r0 = 0.075 * (1 - j / 6.5) + 0.012; r1 = 0.075 * (1 - (j + 1) / 6.5) + 0.012
        arms += capsule(pts[j], pts[j + 1], r0, r1, 20)
save("creature", body + arms)

# 4. chair: seat, four legs, two back posts and a back rest
seat = box(0.46, 0.05, 0.46, y=0.45)
legs = sum((box(0.045, 0.45, 0.045, x=sx * 0.2, z=sz * 0.2) for sx in (-1, 1) for sz in (-1, 1)), Manifold())
posts = box(0.045, 0.5, 0.045, x=-0.2, y=0.5, z=-0.2) + box(0.045, 0.5, 0.045, x=0.2, y=0.5, z=-0.2)
back = box(0.46, 0.2, 0.04, y=0.78, z=-0.2)
save("chair", seat + legs + posts + back, "pine", {"grain": [1.0, 0.0, 0.0]})

# configs: bake.toml with cells sized for each model and the number of
# Level-1 pieces to show (2 = the first fracture mode)
cfg = open(ROOT + "/benchmarks/configs/bake.toml").read()
for name, density, pieces in (("bracket", 15000, 2), ("bone", 60000, 2), ("creature", 12000, 9), ("chair", 40000, 6)):
    open(f"{D}/{name}.toml", "w").write(cfg.replace("target_analysis_cells_per_m3 = 200", f"target_analysis_cells_per_m3 = {density}")
                                        .replace("fine_per_analysis = 8", "fine_per_analysis = 4")
                                        .replace("target_level1_fragments = 12", f"target_level1_fragments = {pieces}"))
