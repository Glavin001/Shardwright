"""Linear-elastic FEM oracle on an unfractured solid using Kratos
Multiphysics (StructuralMechanicsApplication), meshed with gmsh
(quadratic tetrahedra).

Used by bond_fidelity.py. Validation-only tooling (never shipped)."""
import os
import numpy as np


def mesh_solid(verts, tris, h, order=2):
    """Tetrahedralize a closed triangle mesh with gmsh. Returns (nodes, tets)
    with tets of 10 nodes (order 2) in gmsh ordering."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("solid")
    # discrete surface from triangles
    tag = gmsh.model.addDiscreteEntity(2)
    ntags = np.arange(1, len(verts) + 1)
    gmsh.model.mesh.addNodes(2, tag, ntags, np.asarray(verts, float).ravel())
    gmsh.model.mesh.addElementsByType(tag, 2, [], (np.asarray(tris) + 1).ravel())
    gmsh.model.mesh.classifySurfaces(np.pi / 6, True, False, np.pi)
    gmsh.model.mesh.createGeometry()
    surfs = [s[1] for s in gmsh.model.getEntities(2)]
    loop = gmsh.model.geo.addSurfaceLoop(surfs)
    gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.3)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.model.mesh.generate(3)
    if order == 2:
        gmsh.model.mesh.setOrder(2)
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    etype = 11 if order == 2 else 4
    _, enodes = gmsh.model.mesh.getElementsByType(etype)
    npe = 10 if order == 2 else 4
    enodes = enodes.reshape(-1, npe)
    gmsh.finalize()
    index = {int(t): i for i, t in enumerate(node_tags)}
    tets = np.vectorize(index.get)(enodes)
    return coords, tets


def solve_static(coords, tets, E, nu, rho, fixed_mask, nodal_forces):
    """Linear static solve. Returns (displacements[N,3], element stress
    tensors [n_el, 3, 3] averaged over integration points, element centroids)."""
    import KratosMultiphysics as KM
    import KratosMultiphysics.StructuralMechanicsApplication as SMA  # noqa: F401
    model = KM.Model()
    mp = model.CreateModelPart("Structure")
    mp.SetBufferSize(2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for v in [KM.DISPLACEMENT, KM.REACTION, KM.VOLUME_ACCELERATION]:
        mp.AddNodalSolutionStepVariable(v)
    import KratosMultiphysics.StructuralMechanicsApplication as SMA
    mp.AddNodalSolutionStepVariable(SMA.POINT_LOAD)
    for i, c in enumerate(coords):
        mp.CreateNewNode(i + 1, float(c[0]), float(c[1]), float(c[2]))
    prop = mp.GetProperties()[1]
    prop.SetValue(KM.YOUNG_MODULUS, E)
    prop.SetValue(KM.POISSON_RATIO, nu)
    prop.SetValue(KM.DENSITY, rho)
    prop.SetValue(KM.CONSTITUTIVE_LAW, SMA.LinearElastic3DLaw())
    npe = tets.shape[1]
    ename = "SmallDisplacementElement3D10N" if npe == 10 else "SmallDisplacementElement3D4N"
    # gmsh 10-node ordering: 0..3 corners, 4:(0,1) 5:(1,2) 6:(2,0) 7:(0,3) 8:(2,3) 9:(1,3)
    # Kratos Tetrahedra3D10: 4:(0,1) 5:(1,2) 6:(2,0) 7:(0,3) 8:(1,3) 9:(2,3)
    perm = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8] if npe == 10 else [0, 1, 2, 3]
    for e, t in enumerate(tets):
        mp.CreateNewElement(ename, e + 1, [int(t[k]) + 1 for k in perm], prop)
    KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, KM.REACTION_X, mp)
    KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y, mp)
    KM.VariableUtils().AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z, mp)
    loads = []
    for i in np.nonzero(fixed_mask)[0]:
        n = mp.GetNode(int(i) + 1)
        n.Fix(KM.DISPLACEMENT_X)
        n.Fix(KM.DISPLACEMENT_Y)
        n.Fix(KM.DISPLACEMENT_Z)
    cid = 1
    for i, f in enumerate(nodal_forces):
        if np.any(f != 0.0):
            mp.CreateNewCondition("PointLoadCondition3D1N", cid, [i + 1], prop)
            n = mp.GetNode(i + 1)
            n.SetSolutionStepValue(SMA.POINT_LOAD, KM.Vector([float(f[0]), float(f[1]), float(f[2])]))
            cid += 1
    linear_solver = KM.SkylineLUFactorizationSolver()
    try:
        import KratosMultiphysics.LinearSolversApplication as LSA
        linear_solver = LSA.SparseLUSolver()
    except Exception:
        pass
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, False, False, False, False)
    strategy.SetEchoLevel(0)
    mp.CloneTimeStep(1.0)
    mp.ProcessInfo[KM.STEP] = 1
    strategy.Initialize()
    strategy.Solve()
    disp = np.array([[n.GetSolutionStepValue(KM.DISPLACEMENT)[k] for k in range(3)] for n in mp.Nodes])
    stresses = []
    cents = []
    for el in mp.Elements:
        sv = el.CalculateOnIntegrationPoints(KM.CAUCHY_STRESS_VECTOR, mp.ProcessInfo)
        s = np.mean(np.array([[v[k] for k in range(6)] for v in sv]), axis=0)
        # Kratos Voigt: xx, yy, zz, xy, yz, xz
        S = np.array([[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]])
        stresses.append(S)
        g = el.GetGeometry()
        cents.append(np.mean([[g[k].X, g[k].Y, g[k].Z] for k in range(4)], axis=0))
    return disp, np.array(stresses), np.array(cents)


def eigenfrequencies(coords, tets, E, nu, rho, fixed_mask, n):
    """Natural frequencies (Hz) with scipy on the Kratos-assembled... we use
    an independent scipy assembly of quadratic tets for the eigenproblem."""
    from skfem_modal import modal_frequencies
    return modal_frequencies(coords, tets, E, nu, rho, fixed_mask, n)
