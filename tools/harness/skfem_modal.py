"""Independent modal oracle: quadratic tetrahedral linear elasticity
assembled with scikit-fem, generalized eigenproblem via shift-invert."""
import numpy as np


def modal_frequencies(coords, tets, E, nu, rho, fixed_mask, n=10):
    import skfem
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    from skfem.helpers import dot
    from scipy.sparse.linalg import eigsh
    corners = tets[:, :4]
    used = np.unique(corners)
    remap = -np.ones(len(coords), int)
    remap[used] = np.arange(len(used))
    m = skfem.MeshTet(coords[used].T, remap[corners].T)
    e = skfem.ElementVector(skfem.ElementTetP2())
    ib = skfem.Basis(m, e)
    K = skfem.asm(linear_elasticity(*lame_parameters(E, nu)), ib)

    @skfem.BilinearForm
    def mass(u, v, w):
        return rho * dot(u, v)

    M = skfem.asm(mass, ib)
    fixed_nodes = np.nonzero(fixed_mask[used])[0]
    D = ib.get_dofs(nodes=fixed_nodes).flatten() if len(fixed_nodes) else np.array([], int)
    # skfem: dofs for nodes include P2 edge dofs via facets; use condense
    I = ib.complement_dofs(D)
    Kc = K[I][:, I]
    Mc = M[I][:, I]
    k = min(n + (0 if len(D) else 6), Kc.shape[0] - 2)
    w, _ = eigsh(Kc, k=k, M=Mc, sigma=0.0, which="LM")
    w = np.sort(w[w > 1e-6 * max(1.0, abs(w).max())])
    return list(np.sqrt(w)[:n] / (2 * np.pi))
