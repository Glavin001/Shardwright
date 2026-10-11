// Voro++ reference oracle (test-only, BSD-style license; never shipped).
// Usage: voro_oracle xmin xmax ymin ymax zmin zmax < points.txt
// points.txt lines: "id x y z". Output per cell (17 significant digits):
// "id volume nverts x y z x y z ..."
#include "voro++.hh"
#include <cstdio>
#include <vector>
using namespace voro;
int main(int argc, char** argv) {
  if (argc < 7) return 1;
  double b[6];
  for (int i = 0; i < 6; i++) b[i] = atof(argv[i + 1]);
  std::vector<int> ids; std::vector<double> xs, ys, zs;
  int id; double x, y, z;
  while (scanf("%d %lf %lf %lf", &id, &x, &y, &z) == 4) { ids.push_back(id); xs.push_back(x); ys.push_back(y); zs.push_back(z); }
  int n = (int)ids.size();
  int nb = 1; while (nb * nb * nb < n / 4 + 1) nb++;
  container con(b[0], b[1], b[2], b[3], b[4], b[5], nb, nb, nb, false, false, false, 8);
  for (int i = 0; i < n; i++) con.put(ids[i], xs[i], ys[i], zs[i]);
  c_loop_all cl(con);
  voronoicell_neighbor c;
  if (cl.start()) do {
    if (con.compute_cell(c, cl)) {
      double px, py, pz; cl.pos(px, py, pz);
      std::vector<double> v; c.vertices(px, py, pz, v);
      printf("%d %.17g %d", cl.pid(), c.volume(), (int)(v.size() / 3));
      for (size_t k = 0; k < v.size(); k++) printf(" %.17g", v[k]);
      printf("\n");
    }
  } while (cl.inc());
  return 0;
}
