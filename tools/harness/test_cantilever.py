import numpy as np, sys
sys.path.insert(0, ".")
from kratos_fem import mesh_solid, solve_static
# box 2 x 0.2 x 0.2 cantilever, E=30e9
L, b = 2.0, 0.2
v = np.array([[x, y, z] for z in (0, b) for y in (0, b) for x in (0, L)], float)
# box triangles (outward)
idx = lambda x, y, z: x + 2 * y + 4 * z
quads = [[0,2,3,1],[4,5,7,6],[0,1,5,4],[2,6,7,3],[0,4,6,2],[1,3,7,5]]
tris = []
for q in quads:
    tris += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
coords, tets = mesh_solid(v, np.array(tris), 0.05)
E, nu = 30e9, 0.2
fixed = coords[:, 0] < 1e-9
end = np.nonzero(coords[:, 0] > L - 1e-9)[0]
P = 1000.0
f = np.zeros_like(coords)
f[end, 1] = -P / len(end)
d, S, C = solve_static(coords, tets, E, nu, 2400, fixed, f)
I = b ** 4 / 12
print("tets", len(tets), "tip defl", d[end, 1].mean(), "theory", -P * L**3 / (3 * E * I) * (1 + 0))
