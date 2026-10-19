"""
ert_common.py
==============
Shared pieces used by all five inversion scripts so they operate on the
EXACT SAME synthetic survey / mesh / noisy data and are directly
comparable:

    - ERTForward        : thin wrapper around pyGIMLi's ERTModelling
    - build_synthetic_example : one shallow conductor + one deep resistor,
                                 the classic case where a uniform-lambda
                                 L2 inversion under-recovers the deep body
    - roughness_matrix   : first-order (Laplacian-like) smoothness operator
                            built from mesh cell adjacency, used by the
                            conventional L2 script
    - plot_field         : quick scatter-based cell plotting helper

Run any of the five method scripts directly; each calls example() which
uses this module.
"""

import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert


# ----------------------------------------------------------------------
# Forward operator wrapper
# ----------------------------------------------------------------------
class ERTForward:
    """
    Wraps pyGIMLi's ERTModelling forward operator on a given mesh.
    Works in log-resistivity <-> log-apparent-resistivity throughout,
    which is the standard, better-conditioned parameterization for ERT.
    """

    def __init__(self, scheme, mesh):
        self.mesh = mesh
        self.scheme = scheme
        self.fop = ert.ERTModelling()
        self.fop.setData(scheme)
        self.fop.setMesh(self.mesh)
        # n_cells must match the paraDomain (non-background cells only),
        # because fop.response() and fop.jacobian() both operate in that
        # reduced parameter space. Using mesh.cellCount() (which includes
        # the background buffer region) causes a shape mismatch:
        #   J is (n_data, n_paraDomain)  but  res would be (n_mesh,).
        self.n_cells = self.fop.paraDomain.cellCount()
        self.n_data = scheme.size()

    def response(self, log_res):
        res = np.exp(log_res)
        return np.array(self.fop.response(pg.Vector(res)))

    def jacobian_loglog(self, log_res):
        """dlog(rhoa)/dlog(rho). Shape (n_data, n_cells)."""
        res = np.exp(log_res)
        # response() FIRST: pyGIMLi's createJacobian() reuses the potentials of the
        # most recent response() call, not the model passed to it.
        dcalc = self.response(log_res)
        self.fop.createJacobian(pg.Vector(res))
        J = np.array(self.fop.jacobian())
        return (J * res[None, :]) / dcalc[:, None], dcalc


# ----------------------------------------------------------------------
# Mesh-adjacency first-order roughness matrix (for conventional L2)
# ----------------------------------------------------------------------
def roughness_matrix(mesh, para_marker=None, z_weight=1.0):
    """
    Build a sparse first-order smoothness (finite-difference) operator C
    such that C @ m penalizes differences between neighbouring cells.
    One row per interior mesh boundary shared by two cells.

    Parameters
    ----------
    mesh : pg.Mesh
        The full mesh (e.g. para_mesh). Pass `para_marker` to restrict
        C to the paraDomain cells so its column count matches n_cells.
    para_marker : int or None
        If given, only boundaries where BOTH neighbours carry this marker
        are included, and cells are re-indexed to match paraDomain ordering.
        If None, uses all cells (legacy behaviour).
    z_weight : float
        Anisotropy (as pyGIMLi's zWeight). Rows for boundaries whose normal
        is vertical (i.e. penalising VERTICAL roughness) are scaled by
        z_weight; z_weight < 1 favours laterally elongated structure.
    """
    import numpy as np
    from scipy.sparse import coo_matrix

    if para_marker is not None:
        markers = np.array([c.marker() for c in mesh.cells()])
        para_global = np.where(markers == para_marker)[0]
        global_to_local = {int(gid): lid for lid, gid in enumerate(para_global)}
        n = len(para_global)
        rows, cols, vals = [], [], []
        r = 0
        for b in mesh.boundaries():
            lc = b.leftCell()
            rc = b.rightCell()
            if lc is None or rc is None:
                continue
            li, ri = int(lc.id()), int(rc.id())
            if li in global_to_local and ri in global_to_local:
                ll, rl = global_to_local[li], global_to_local[ri]
                ny_ = abs(b.norm().y())
                w = 1.0 - (1.0 - z_weight) * ny_
                rows += [r, r]; cols += [ll, rl]; vals += [w, -w]
                r += 1
        C = coo_matrix((vals, (rows, cols)), shape=(r, n)).tocsr()
    else:
        rows, cols, vals = [], [], []
        r = 0
        for b in mesh.boundaries():
            if b.leftCell() is not None and b.rightCell() is not None:
                i, j = b.leftCell().id(), b.rightCell().id()
                rows += [r, r]
                cols += [i, j]
                vals += [1.0, -1.0]
                r += 1
        C = coo_matrix((vals, (rows, cols)), shape=(r, mesh.cellCount())).tocsr()
    return C


# ----------------------------------------------------------------------
# Synthetic example shared by every script
# ----------------------------------------------------------------------
def irls_row_weights(Cm, eps=None, p=1.0, eps_rel=0.1):
    """
    IRLS weights turning ||C m||_2^2 into an approximate L1 ("blocky")
    norm of the model gradient: w_i = ((Cm)_i^2 + eps^2)^((p-2)/2), scaled
    (p=1: blocky/L1; p=0: minimum-gradient-support "focusing"),
    so mean(w) = 1 (keeps lambda comparable to the smooth L2 case).
    Returns sqrt(w) so that  C_w = diag(sqrt(w)) @ C.
    """
    Cm = np.asarray(Cm)
    if eps is None:
        eps = max(eps_rel * np.std(Cm), 1e-3)
    w = (Cm ** 2 + eps ** 2) ** ((p - 2.0) / 2.0)
    w /= np.mean(w)
    return np.sqrt(w)


def coverage_row_weights(C, J, Wd, gamma=0.5, w_min=0.02):
    """
    Per-row weights for the roughness operator from the data coverage
    cov_j = sqrt(sum_i (Wd_i J_ij)^2). Row weight = (cov_edge / median)^gamma,
    capped at 1: well-covered (shallow) regions keep full smoothing, poorly
    covered (deep) regions are smoothed less, counteracting the tendency of a
    uniform lambda to flatten everything the data only weakly constrain.
    """
    from scipy.sparse import csr_matrix
    cov = np.sqrt(np.sum((Wd[:, None] * J) ** 2, axis=0))
    cov = cov / np.median(cov)
    A = abs(csr_matrix(C))
    A.data[:] = 1.0
    # geometric mean of the two cells' coverage per row (via log-average)
    edge = np.exp((A @ np.log(cov + 1e-12)) / np.maximum(A.sum(axis=1).A1, 1))
    return np.clip(edge ** gamma, w_min, 1.0)


def build_synthetic_example(seed=0, noise_level=0.03):
    elec_x = np.linspace(0, 47, 48)
    scheme = ert.createData(elecs=elec_x, schemeName="dd")

    world = mt.createWorld(start=[-20, -25], end=[67, 0], worldMarker=False)
    for e in elec_x:
        world.createNode([e, 0])
    para_mesh = mt.createParaMesh(scheme, quality=33,
                                   paraMaxCellSize=2.0, boundaryMaxCellSize=50)

    fwd = ERTForward(scheme, para_mesh)

    # Cell centers must come from fop.paraDomain, not the full para_mesh.
    # The paraDomain excludes the background buffer region, so its cell
    # count matches fwd.n_cells (= fop.paraDomain.cellCount()).
    cell_centers = np.array([c.center() for c in fwd.fop.paraDomain.cells()])

    true_log_res = np.full(fwd.n_cells, np.log(100.0))
    shallow = (np.abs(cell_centers[:, 0] - 20) < 4) & (cell_centers[:, 1] > -4)
    deep = ((np.abs(cell_centers[:, 0] - 35) < 8)
            & (cell_centers[:, 1] < -10) & (cell_centers[:, 1] > -16))
    true_log_res[shallow] = np.log(15.0)
    true_log_res[deep] = np.log(400.0)

    dobs_clean = fwd.response(true_log_res)
    rng = np.random.default_rng(seed)
    dobs = dobs_clean * (1 + rng.normal(0, noise_level, size=dobs_clean.shape))
    derr = np.full_like(dobs_clean, noise_level)  # relative error: residuals are in log(rhoa)

    return dict(
        scheme=scheme, mesh=para_mesh,
        paraDomain=fwd.fop.paraDomain,  # non-background cells; size == fwd.n_cells
        fwd=fwd,
        cell_centers=cell_centers, true_log_res=true_log_res,
        dobs=dobs, derr=derr,
        # starting model: homogeneous half-space at the median apparent
        # resistivity (data-derived, NOT the true background)
        log_res0=np.full(fwd.n_cells, np.log(np.median(dobs))),
        shallow_mask=shallow, deep_mask=deep,
    )


def plot_field(ax, cell_centers, vals, title, cmap="viridis", **kw):
    sc = ax.scatter(cell_centers[:, 0], cell_centers[:, 1], c=vals,
                     s=8, cmap=cmap, **kw)
    ax.set_title(title)
    ax.set_aspect("equal")
    return sc
