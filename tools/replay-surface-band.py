"""Shadow surface-band replay against captured production fields.

Usage: python3.11 tools/replay-surface-band.py CAPTURE_ARM_DIRECTORY [--out DIR]
                  [--interp cubic|linear] [--sections 2,4,6,7,8,10,12,16,20]
                  [--width METRES] [--extension normal|approach]
Output: <out>/replay.json, phi-{A,B,C,D}-step-N.npy, section-step-N.png;
render a composite with tools/render-surface-band-replay.py <out>.

The capture directory comes from tools/capture-retained-visual-ab-dawn.ts run
with every step listed (--steps=0,1,2,...,N). This script never touches the
solver. It replays, on the CPU, the coupling proposed in
docs/HANDOFF_FLUID_SURFACE_REVIEW_2026-09-08.md against the unmodified
solver's own per-step native density and velocity:

  1. a fine-lattice signed distance phi seeded from the scene primitives,
  2. velocity taken from solver cells with density >= 0.5 and extended into
     the rest of the lattice along the band normal (constant along normals),
  3. semi-Lagrangian RK2 advection of phi with that velocity,
  4. redistancing with the Russo-Smereka subcell fix (interface kept),
  5. one constant shift of phi per native cell so the cell mean of the
     production amount model q = clamp(.5 - phi/w, 0, 1) (the seed ramp,
     w = one fine cell; phi tricubic at 4^3 sub-points per fine cell)
     equals the solver's conserved density (arms B, C),
  6. redistancing again.

Velocity extension: 'normal' takes each unknown cell's velocity from its
upwind neighbours along the band normal (closest interface); 'approach'
prefers set neighbours whose velocity points into the cell, so the air
between two approaching bodies follows the body that will arrive.

Arms: A = band only (no mass coupling); B = per-native-cell shift on every
active native cell whose band amount disagrees with its density; C = the same
shift restricted to native cells the band's interface passes through; D = no
per-cell shift at the interface: native cells the solver holds full (>= .995)
or empty (<= .005) are filled or drained, then one shift per connected liquid
body makes the body's amount equal the solver mass assigned to it (each fine
cell is assigned to the body its nearest liquid belongs to).

Metrics reuse tools/analyze-current-map-ray-field.py: half-level roots along
134 antipodal rays from the discrete free-fall centre, and pool heights on a
17x17 grid, against the same references the current-map analysis used.
"""
import argparse
import importlib.util
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    'ray_field', Path(__file__).with_name('analyze-current-map-ray-field.py'))
_ray = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ray)

SQRT3 = np.sqrt(3.0)


# ---------------------------------------------------------------- capture ---

def load_configuration(directory):
    config = json.loads((directory / 'configuration.json').read_text())
    scene = config['scene']
    sphere = scene['fluid']['initialLiquidVolumes'][0]
    assert sphere['shape'] == 'sphere'
    steps = config['steps']
    assert steps == list(range(len(steps))), 'replay needs every step captured'
    return dict(
        h=float(config['h']), dims=tuple(int(n) for n in config['dimensions']),
        origin=np.asarray(config['origin'], dtype=float), dt=float(config['dt']),
        center0=np.array([sphere['center_m']['x'], sphere['center_m']['y'], sphere['center_m']['z']]),
        radius=float(sphere['radius_m']),
        pool=float(scene['container']['height_m'] * scene['container']['fillFraction']),
        gravity=np.array([scene['fluid']['gravity_m_s2'][k] for k in 'xyz'], dtype=float),
        steps=steps, arm=config['arm'], scene_id=config['sceneId'])


def load_step(directory, step, dims):
    nx, ny, nz = dims
    base = directory / f'step-{step}'
    density = np.fromfile(base / 'density.bin', dtype='<f4').astype(float).reshape(nz, ny, nx).transpose(2, 1, 0)
    velocity = np.fromfile(base / 'velocity.bin', dtype='<f4').astype(float).reshape(nz, ny, nx, 4).transpose(2, 1, 0, 3)[..., :3]
    activity = json.loads((base / 'activity.json').read_text())
    receipt = json.loads((base / 'receipt.json').read_text())
    mesh = np.fromfile(base / 'mesh.bin', dtype='<f4').astype(float).reshape(-1, 8)
    return dict(density=np.ascontiguousarray(density), velocity=np.ascontiguousarray(velocity),
                activity=activity, receipt=receipt, mesh=mesh)


def native_layout(activity, dims):
    """Native cell id per fine cell (-1 outside active bricks) and width per id."""
    nid = np.full(dims, -1, dtype=np.int64)
    widths = []
    k = 0
    for brick in activity['bricks']:
        if not brick['active']:
            continue
        s, r = int(brick['spanBricks']), int(brick['acceptedResolution'])
        w = 8 * s // r
        bx, by, bz = brick['coordinate']
        for i in range(r):
            for j in range(r):
                for l in range(r):
                    x0, y0, z0 = 8 * s * bx + i * w, 8 * s * by + j * w, 8 * s * bz + l * w
                    sl = (slice(max(x0, 0), min(x0 + w, dims[0])), slice(max(y0, 0), min(y0 + w, dims[1])),
                          slice(max(z0, 0), min(z0 + w, dims[2])))
                    if any(a.start >= a.stop for a in sl):
                        continue
                    assert (nid[sl] == -1).all(), 'active bricks overlap'
                    nid[sl] = k
                    widths.append(w)
                    k += 1
    return nid, np.asarray(widths, dtype=float)


def native_means(field, nid, count):
    valid = nid >= 0
    sums = np.bincount(nid[valid], weights=field[valid], minlength=count)
    cells = np.bincount(nid[valid], minlength=count)
    return sums / np.maximum(cells, 1), cells


# --------------------------------------------------------------- lattice ---

class Lattice:
    def __init__(self, dims, origin, h):
        self.dims, self.origin, self.h = dims, origin, h
        axes = [origin[a] + (np.arange(dims[a]) + 0.5) * h for a in range(3)]
        self.centers = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        self.lower = origin
        self.upper = origin + np.asarray(dims) * h

    def fractional(self, points):
        u = (np.asarray(points, dtype=float) - self.origin) / self.h - 0.5
        return np.clip(u, 0.0, np.asarray(self.dims) - 1.0 - 1e-9)

    def trilinear(self, field, points):
        u = self.fractional(points)
        i0 = np.floor(u).astype(int)
        f = u - i0
        i1 = np.minimum(i0 + 1, np.asarray(self.dims) - 1)
        out = 0.0
        for dx, wx in ((0, 1 - f[:, 0]), (1, f[:, 0])):
            ix = i1[:, 0] if dx else i0[:, 0]
            for dy, wy in ((0, 1 - f[:, 1]), (1, f[:, 1])):
                iy = i1[:, 1] if dy else i0[:, 1]
                for dz, wz in ((0, 1 - f[:, 2]), (1, f[:, 2])):
                    iz = i1[:, 2] if dz else i0[:, 2]
                    w = wx * wy * wz
                    out = out + (w[:, None] * field[ix, iy, iz] if field.ndim == 4 else w * field[ix, iy, iz])
        return out

    def tricubic(self, field, points):
        """Catmull-Rom on cell centres with clamped indices."""
        u = self.fractional(points)
        i0 = np.floor(u).astype(int)
        t = u - i0
        weights, indices = [], []
        for a in range(3):
            ta = t[:, a]
            w = np.stack([-0.5 * ta ** 3 + ta ** 2 - 0.5 * ta,
                          1.5 * ta ** 3 - 2.5 * ta ** 2 + 1.0,
                          -1.5 * ta ** 3 + 2.0 * ta ** 2 + 0.5 * ta,
                          0.5 * ta ** 3 - 0.5 * ta ** 2], 1)
            idx = np.stack([np.clip(i0[:, a] + d, 0, self.dims[a] - 1) for d in (-1, 0, 1, 2)], 1)
            weights.append(w)
            indices.append(idx)
        out = np.zeros(len(u))
        for a in range(4):
            for b in range(4):
                wab = weights[0][:, a] * weights[1][:, b]
                for c in range(4):
                    out += wab * weights[2][:, c] * field[indices[0][:, a], indices[1][:, b], indices[2][:, c]]
        return out


def gradient(phi, h):
    return np.stack(np.gradient(phi, h), -1)


def unit_normal(phi, h):
    g = gradient(phi, h)
    mag = np.linalg.norm(g, axis=-1)
    return g / np.maximum(mag, 1e-12)[..., None], mag


def shifted(field, axis, delta):
    """field at index+delta along axis, and validity mask (no wrap)."""
    out = np.roll(field, -delta, axis=axis)
    valid = np.ones(field.shape[:3], dtype=bool)
    idx = [slice(None)] * 3
    if delta > 0:
        idx[axis] = slice(field.shape[axis] - delta, None)
    else:
        idx[axis] = slice(0, -delta)
    valid[tuple(idx)] = False
    return out, valid


# --------------------------------------------------------------- physics ---

def extend_velocity(velocity, known, phi, h, iterations=90, mode='normal'):
    """Constant-along-normal extension of the known velocity into unknown cells.

    Unknown cells take the |n_a|-weighted average of their upwind neighbours,
    upwind meaning toward decreasing phi; known cells are Dirichlet data.
    Iterated Jacobi until nothing changes.
    """
    normal, mag = unit_normal(phi, h)
    isotropic = mag < 1e-9
    U = np.where(known[..., None], velocity, 0.0)
    filled = known.copy()
    for _ in range(iterations):
        num = np.zeros_like(U)
        den = np.zeros(U.shape[:3])
        num_app = np.zeros_like(U)
        den_app = np.zeros(U.shape[:3])
        for a in range(3):
            for delta in (-1, 1):
                nb, valid = shifted(U, a, delta)
                nbset, _ = shifted(filled, a, delta)
                # upwind: toward decreasing phi. n_a>0 => phi decreases toward -1.
                w = np.where(isotropic, 1.0, np.where((normal[..., a] > 0) == (delta < 0), np.abs(normal[..., a]), 0.0))
                w = w * valid * nbset
                num += w[..., None] * nb
                den += w
                if mode == 'approach':
                    # neighbour velocity component pointing from the neighbour into this cell
                    approaching = (-delta * nb[..., a] > 1e-3) & valid & nbset
                    wa = np.where(approaching, np.abs(normal[..., a]) + 0.05, 0.0)
                    num_app += wa[..., None] * nb
                    den_app += wa
        if mode == 'approach':
            use = den_app > 0
            num = np.where(use[..., None], num_app, num)
            den = np.where(use, den_app, den)
        update = (~known) & (den > 0)
        new = np.where(update[..., None], num / np.maximum(den, 1e-300)[..., None], U)
        change = np.abs(new - U).max() if update.any() else 0.0
        U = new
        filled |= update
        if change < 1e-12 and filled.all():
            break
    return U, filled


def advect(phi, U, lattice, dt, interp):
    X = lattice.centers
    u1 = lattice.trilinear(U, X)
    Xm = X - 0.5 * dt * u1
    um = lattice.trilinear(U, Xm)
    Xd = X - dt * um
    return interp(phi, Xd).reshape(lattice.dims)


def redistance(phi, h, iterations=24, dtau=0.4, lipschitz=1.5):
    """Godunov reinitialisation with the Russo-Smereka subcell fix.

    Three guards on the fix, each measured on the quarter capture at steps 6
    and 7 (the last two before contact) against the exact reference distance:
      - an axis whose two neighbours are both across the interface (a gap
        thinner than two cells) uses the larger one-sided difference instead
        of the central difference, whose cancellation blew D up to ~1 m;
      - |D| is clamped to h: a cell with a face neighbour across the interface
        is at most one cell from it;
      - an air-side interface cell whose value differs from an across-interface
        neighbour by more than `lipschitz` cells is stale (it still holds the
        distance to the body it was nearest to before the other body arrived)
        and is rebuilt from the liquid side instead of being pinned.
    Without the guards the band error in |phi| < 2h at step 6 was 9.2 mm RMS /
    218 mm max and 10.2 / 152 at step 7; with them 1.4 / 8 and 2.1 / 25
    (steps 1-5 unchanged, 1.3-1.7 mm RMS).
    """
    phi0 = phi.copy()
    S = phi0 / np.sqrt(phi0 ** 2 + h ** 2)
    positive = phi0 > 0
    interface = np.zeros(phi.shape, dtype=bool)
    stale = np.zeros(phi.shape, dtype=bool)
    G2 = np.zeros(phi.shape)
    for a in range(3):
        fwd, vf = shifted(phi0, a, 1)
        bwd, vb = shifted(phi0, a, -1)
        dp = np.where(vf, (fwd - phi0) / h, 0.0)
        dm = np.where(vb, (phi0 - bwd) / h, 0.0)
        central = np.where(vf & vb, (fwd - bwd) / (2 * h), dp + dm)
        cross_f = vf & ((fwd > 0) != positive)
        cross_b = vb & ((bwd > 0) != positive)
        G2 += np.where(cross_f & cross_b, np.maximum(np.abs(dp), np.abs(dm)), central) ** 2
        interface |= cross_f | cross_b
        stale |= (cross_f & (np.abs(phi0 - fwd) > lipschitz * h)) | (cross_b & (np.abs(phi0 - bwd) > lipschitz * h))
    interface &= ~(stale & positive)
    D = np.clip(phi0 / np.maximum(np.sqrt(G2), 1e-6), -h, h)
    sgn0 = np.sign(phi0)
    for _ in range(iterations):
        G2 = np.zeros(phi.shape)
        for a in range(3):
            fwd, vf = shifted(phi, a, 1)
            bwd, vb = shifted(phi, a, -1)
            dp = np.where(vf, (fwd - phi) / h, 0.0)
            dm = np.where(vb, (phi - bwd) / h, 0.0)
            pos = np.maximum(np.maximum(dm, 0.0) ** 2, np.minimum(dp, 0.0) ** 2)
            neg = np.maximum(np.minimum(dm, 0.0) ** 2, np.maximum(dp, 0.0) ** 2)
            G2 += np.where(S > 0, pos, neg)
        G = np.sqrt(G2)
        bulk = phi - dtau * h * S * (G - 1.0)
        fix = phi - dtau * (sgn0 * np.abs(phi) - D)
        phi = np.where(interface, fix, bulk)
    return phi


def plane_fractions(phi, normal, h):
    """Liquid fraction of each cell under the plane n.(x-xc)+phi=0 (exact)."""
    absn = np.abs(normal)
    L1 = absn.sum(-1)
    degenerate = L1 < 1e-9
    m = absn / np.maximum(L1, 1e-300)[..., None]
    m = np.maximum(m, 1e-4)
    m = m / m.sum(-1, keepdims=True)
    alpha = np.clip(0.5 - phi / (h * np.maximum(L1, 1e-9)), 0.0, 1.0)
    cube = lambda x: np.maximum(x, 0.0) ** 3
    V = cube(alpha)
    for a in range(3):
        V = V - cube(alpha - m[..., a]) + cube(alpha - 1.0 + m[..., a])
    V = V / (6.0 * m[..., 0] * m[..., 1] * m[..., 2])
    V = np.clip(V, 0.0, 1.0)
    linear = np.clip(0.5 - phi / h, 0.0, 1.0)
    return np.where(degenerate, linear, V)


_GAUSS = np.polynomial.legendre.leggauss(8)
_SUB = 4
_SUB_OFFSETS = None
_CLASSES = 'topology'  # fill/drain class test: 'topology' (sign) or 'ramp' (sub-cell); set from --classes


def amount_samples(phi, lattice):
    """phi at 4^3 sub-points of every fine cell (tricubic), shape (cells, 64)."""
    global _SUB_OFFSETS
    if _SUB_OFFSETS is None:
        offs = (np.arange(_SUB) + 0.5) / _SUB - 0.5
        ox, oy, oz = np.meshgrid(offs, offs, offs, indexing='ij')
        _SUB_OFFSETS = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], -1)
    out = np.empty((len(lattice.centers), len(_SUB_OFFSETS)))
    for k, o in enumerate(_SUB_OFFSETS):
        out[:, k] = lattice.tricubic(phi, lattice.centers + o * lattice.h)
    return out


def amount_from_samples(samples, shift, w):
    """Cell mean of the ramp q = clamp(.5 - (phi + shift)/w, 0, 1) from sub-samples."""
    if w <= 0:
        return ((samples + shift[:, None]) <= 0).mean(1)
    return np.clip(0.5 - (samples + shift[:, None]) / w, 0.0, 1.0).mean(1)


def amount_field(phi, lattice, w):
    return amount_from_samples(amount_samples(phi, lattice), np.zeros(len(lattice.centers)), w).reshape(phi.shape)


def ramp_fractions(phi, normal, h, w):
    """Cell mean of the production amount model q = clamp(.5 - phi/w, 0, 1).

    Integrating the ramp over a cell equals averaging the sharp plane-cut
    volume over level shifts in [-w/2, w/2]; the average is taken by Gauss
    quadrature. With w = 0 this is the sharp fraction.
    """
    if w <= 0:
        return plane_fractions(phi, normal, h)
    nodes, weights = _GAUSS
    out = np.zeros(phi.shape)
    for node, weight in zip(nodes, weights):
        out += 0.5 * weight * plane_fractions(phi - 0.5 * w * node, normal, h)
    return out


def correct_mass(phi, lattice, target, nid, widths, cells, h, w, mode, iterations=40):
    """One constant shift of phi per native cell so its ramp amount matches target."""
    count = len(widths)
    valid = nid >= 0
    ids = nid[valid]
    samples = amount_samples(phi, lattice)[valid.reshape(-1)]

    def native_fraction(shift_native):
        V = amount_from_samples(samples, shift_native[ids], w)
        return np.bincount(ids, weights=V, minlength=count) / np.maximum(cells, 1)

    F0 = native_fraction(np.zeros(count))
    need = np.abs(F0 - target) > 1e-4
    if mode == 'interface':
        need &= (F0 > 1e-3) & (F0 < 1 - 1e-3)
    halo = need & (F0 <= 1e-3) & (target > 1e-3)
    deficit = need & (F0 >= 1 - 1e-3) & (target < 1 - 1e-3)
    lo = -(widths + 1.0) * h * SQRT3
    hi = (widths + 1.0) * h * SQRT3
    F_lo = native_fraction(lo)
    F_hi = native_fraction(hi)
    solvable = (F_lo >= target) & (F_hi <= target)
    unreachable = need & ~solvable
    need &= solvable
    halo &= solvable
    deficit &= solvable
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        F = native_fraction(mid)
        too_wet = F > target
        lo = np.where(too_wet, mid, lo)
        hi = np.where(too_wet, hi, mid)
    shift = np.where(need, 0.5 * (lo + hi), 0.0)
    corrected = phi + np.where(valid, shift[np.maximum(nid, 0)], 0.0)
    F1 = native_fraction(shift)
    volume = cells * h ** 3
    stats = dict(
        correctedCells=int(need.sum()), interfaceCells=int(((F0 > 1e-3) & (F0 < 1 - 1e-3)).sum()),
        shift=_ray.statistics(shift[need] * 1e3) if need.any() else None,
        haloCells=int(halo.sum()), haloMass_m3=float((target * volume)[halo].sum()),
        haloShift_mm=_ray.statistics(shift[halo] * 1e3) if halo.any() else None,
        deficitCells=int(deficit.sum()), deficitMass_m3=float(((1 - target) * volume)[deficit].sum()),
        residualBefore=_ray.statistics((F0 - target) * volume) if len(F0) else None,
        residualAfter=_ray.statistics((F1 - target) * volume) if len(F1) else None,
        unreachableCells=int(unreachable.sum()),
        unreachableMismatch_m3=float(np.abs((F0 - target) * volume)[unreachable].sum()),
        ignoredMismatch_m3=float(np.abs((F0 - target) * volume)[~need].sum()))
    return corrected, shift, stats


def label_bodies(liquid, phi, iterations=400):
    """Connected liquid bodies (min-label flooding) and nearest-body label of every cell."""
    big = np.iinfo(np.int64).max
    lab = np.where(liquid, np.arange(liquid.size, dtype=np.int64).reshape(liquid.shape), big)
    for _ in range(iterations):
        new = lab.copy()
        for a in range(3):
            for delta in (-1, 1):
                nb, valid = shifted(lab, a, delta)
                new = np.where(liquid & valid & (nb < new), nb, new)
        if (new == lab).all():
            break
        lab = new
    ids = np.unique(lab[liquid])
    remap = {int(v): i for i, v in enumerate(ids)}
    body = np.full(liquid.shape, -1, dtype=np.int64)
    body[liquid] = [remap[int(v)] for v in lab[liquid]]
    # air cells adopt the label of the neighbour with the smallest phi, until all labelled
    nearest = body.copy()
    for _ in range(iterations):
        unl = nearest < 0
        if not unl.any():
            break
        best_phi = np.full(liquid.shape, np.inf)
        best_lab = np.full(liquid.shape, -1, dtype=np.int64)
        for a in range(3):
            for delta in (-1, 1):
                nb_lab, valid = shifted(nearest, a, delta)
                nb_phi, _ = shifted(phi, a, delta)
                take = unl & valid & (nb_lab >= 0) & (nb_phi < best_phi)
                best_phi = np.where(take, nb_phi, best_phi)
                best_lab = np.where(take, nb_lab, best_lab)
        nearest = np.where(unl & (best_lab >= 0), best_lab, nearest)
    return body, nearest, len(ids)


def correct_bodies(phi, lattice, density, nid, h, w, iterations=40):
    """Fill/drain cells the solver holds full or empty, redistance, then one shift per body (last)."""
    valid = nid >= 0
    count = int(nid.max()) + 1
    cells = np.bincount(nid[valid], minlength=count)
    ids = nid[valid]
    samples = amount_samples(phi, lattice)[valid.reshape(-1)]

    def native_fraction(shift_native):
        V = amount_from_samples(samples, shift_native[ids], w)
        return np.bincount(ids, weights=V, minlength=count) / np.maximum(cells, 1)
    F0 = native_fraction(np.zeros(count))
    rho = np.bincount(nid[valid], weights=density[valid], minlength=count) / np.maximum(cells, 1)
    target = np.clip(rho, 0, 1)
    if _CLASSES == 'topology':
        # Fill/drain only on a topological disagreement: the solver holds the
        # cell full/empty while the band puts its centre on the other side.
        # A sharp solver density (HEAD: exactly 1.0 up to a cell face and 0.0
        # above) holds every interface-adjacent cell at >= 0.995 / <= 0.005
        # while the ramp amount model puts them at 0.875 / 0.125, so the
        # sub-cell comparison fires on the whole pool surface every step.
        fill = (target >= 0.995) & (F0 < 0.5)
        drain = (target <= 0.005) & (F0 > 0.5)
    else:
        fill = (target >= 0.995) & (F0 < 0.995)
        drain = (target <= 0.005) & (F0 > 0.005)
    need = fill | drain
    widths = np.cbrt(np.maximum(cells, 1))
    lo = -(widths + 1.0) * h * SQRT3
    hi = (widths + 1.0) * h * SQRT3
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        F = native_fraction(mid)
        too_wet = F > target
        lo = np.where(too_wet, mid, lo)
        hi = np.where(too_wet, hi, mid)
    shift = np.where(need, 0.5 * (lo + hi), 0.0)
    phi = phi + np.where(valid, shift[np.maximum(nid, 0)], 0.0)
    stats = dict(filledCells=int(fill.sum()), drainedCells=int(drain.sum()),
                 bulkShift=_ray.statistics(shift[need] * 1e3) if need.any() else None)
    # Redistance after the fill/drain, BEFORE the per-body shift: a constant
    # shift keeps |grad phi| = 1, whereas a redistance after the shift moves
    # the interface and broke the mass agreement by up to 6.9 L (0.6 %) at
    # step 11; the shift must be the last operation of the step.
    phi = redistance(phi, h, iterations=12)
    # per-body shift
    body, nearest, bodies = label_bodies(phi < 0, phi)
    sel = nearest >= 0
    samples = amount_samples(phi, lattice)[sel.reshape(-1)]
    labels = nearest[sel]
    target_body = np.bincount(labels, weights=density[sel], minlength=bodies)

    def body_amount(shift_body):
        return np.bincount(labels, weights=amount_from_samples(samples, shift_body[labels], w), minlength=bodies)
    before = body_amount(np.zeros(bodies))
    lo = np.full(bodies, -3 * h)
    hi = np.full(bodies, 3 * h)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        A = body_amount(mid)
        too_wet = A > target_body
        lo = np.where(too_wet, mid, lo)
        hi = np.where(too_wet, hi, mid)
    bshift = 0.5 * (lo + hi)
    phi = phi + np.where(sel, bshift[np.maximum(nearest, 0)], 0.0)
    stats.update(bodies=int(bodies), bodyShift_mm=[round(1e3 * v, 3) for v in bshift],
                 bodyTarget_m3=[float(v * h ** 3) for v in target_body],
                 bodyAmountBefore_m3=[float(v * h ** 3) for v in before])
    return phi, stats


# --------------------------------------------------------------- metrics ---

def reference_center(cfg, step):
    return cfg['center0'] + cfg['gravity'] * cfg['dt'] ** 2 * step * (step - 1) / 2


def reference_fraction(lattice, cfg, step, sub=6):
    """Fine-cell liquid fraction of the reference sphere-and-pool by subsampling."""
    center = reference_center(cfg, step)
    h = lattice.h
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    ox, oy, oz = np.meshgrid(offs, offs, offs, indexing='ij')
    offsets = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], -1) * h
    frac = np.zeros(len(lattice.centers))
    for o in offsets:
        p = lattice.centers + o
        inside = (np.linalg.norm(p - center, axis=1) <= cfg['radius']) | (p[:, 1] <= cfg['pool'])
        frac += inside
    return (frac / len(offsets)).reshape(lattice.dims)


def surface_metrics(cfg, lattice, field_fn, step, directions):
    center = reference_center(cfg, step)
    sphere = _ray.sphere_metrics(field_fn, center, cfg['radius'], cfg['pool'], lattice.lower, lattice.upper,
                                 directions, 65, 1e-8)
    pool = _ray.pool_metrics(field_fn, cfg['pool'], center, cfg['radius'], lattice.lower, lattice.upper,
                             lattice.h, 17, 65, 1e-8)
    if 'samples' in sphere:
        sphere['radialErrorByDirection_mm'] = [None if s['radialError_m'] is None else round(1e3 * s['radialError_m'], 3)
                                               for s in sphere.pop('samples')]
    if 'samples' in pool:
        pool['heightErrorByPoint_mm'] = [None if s['heightError_m'] is None else round(1e3 * s['heightError_m'], 3)
                                         for s in pool.pop('samples')]
    return dict(sphere=sphere, pool=pool)


def mesh_metrics(cfg, mesh, step):
    center = reference_center(cfg, step)
    gap = center[1] - cfg['radius'] - cfg['pool']
    if gap <= 0:
        return dict(referenceScope='contact')
    separator = 0.5 * (cfg['pool'] + center[1] - cfg['radius'])
    p = mesh[:, :3]
    on_sphere = p[:, 1] > separator
    on_pool = (p[:, 1] <= separator) & (p[:, 1] >= cfg['pool'] - 2 * cfg['h'])
    radial = np.linalg.norm(p[on_sphere] - center, axis=1) - cfg['radius']
    height = p[on_pool, 1] - cfg['pool']
    return dict(sphereVertices=int(on_sphere.sum()), radialError_m=_ray.statistics(radial),
                poolVertices=int(on_pool.sum()), heightError_m=_ray.statistics(height))


def body_volumes(cfg, lattice, fraction, step):
    """Liquid volume above and below the reference separator (pre-contact)."""
    center = reference_center(cfg, step)
    gap = center[1] - cfg['radius'] - cfg['pool']
    total = float(fraction.sum() * lattice.h ** 3)
    if gap <= 0:
        return dict(total_m3=total)
    separator = 0.5 * (cfg['pool'] + center[1] - cfg['radius'])
    above = lattice.centers[:, 1].reshape(lattice.dims) > separator
    return dict(total_m3=total, sphere_m3=float(fraction[above].sum() * lattice.h ** 3),
                pool_m3=float(fraction[~above].sum() * lattice.h ** 3))


# --------------------------------------------------------------- imaging ---

def write_png(path, rgb):
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    height, width = rgb.shape[:2]
    raw = b''.join(b'\x00' + rgb[r].tobytes() for r in range(height))

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')
    Path(path).write_bytes(png)


def section_image(cfg, lattice, density, fields, step, pixels_per_cell=6, scale=3):
    """x-y section at z=0: density grey, coloured zero contours."""
    nx, ny = lattice.dims[0], lattice.dims[1]
    W, H = nx * pixels_per_cell, ny * pixels_per_cell
    xs = lattice.origin[0] + (np.arange(W) + 0.5) * lattice.h / pixels_per_cell
    ys = lattice.origin[1] + (np.arange(H) + 0.5) * lattice.h / pixels_per_cell
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], -1)
    grey = np.clip(lattice.trilinear(density, pts).reshape(W, H), 0, 1)
    img = np.repeat((40 + 150 * grey)[..., None], 3, -1)

    def contour(values, colour):
        v = values.reshape(W, H)
        edge = np.zeros((W, H), dtype=bool)
        edge[:-1, :] |= (v[:-1, :] > 0) != (v[1:, :] > 0)
        edge[:, :-1] |= (v[:, :-1] > 0) != (v[:, 1:] > 0)
        img[edge] = colour
    center = reference_center(cfg, step)
    ref = np.minimum(np.linalg.norm(pts - center, axis=1) - cfg['radius'], pts[:, 1] - cfg['pool'])
    contour(0.5 - lattice.trilinear(density, pts), (255, 255, 255))
    for values, colour in fields:
        contour(-values, colour)
    contour(-ref, (255, 220, 0))
    # native cell faces of width > 1 as faint lines are omitted; image is index-aligned.
    img = np.transpose(img, (1, 0, 2))[::-1]
    img = np.repeat(np.repeat(img, scale, 0), scale, 1)
    return img


# ------------------------------------------------------------------ main ---

def replay(directory, out, interp_name, section_steps, width, extension):
    cfg = load_configuration(directory)
    dims, h, dt = cfg['dims'], cfg['h'], cfg['dt']
    w = h if width is None else width
    lattice = Lattice(dims, cfg['origin'], h)
    interp = lattice.tricubic if interp_name == 'cubic' else lattice.trilinear
    directions = _ray.antipodal_directions(134)
    out.mkdir(parents=True, exist_ok=True)

    # Seed: exact signed distance of sphere-and-pool union.
    seed = np.minimum(np.linalg.norm(lattice.centers - cfg['center0'], axis=1) - cfg['radius'],
                      lattice.centers[:, 1] - cfg['pool']).reshape(dims)
    arms = {'A': dict(mode=None), 'B': dict(mode='all'), 'C': dict(mode='interface'), 'D': dict(mode='bodies')}
    for arm in arms.values():
        arm['phi'] = seed.copy()
    report = dict(capture=str(directory), arm=cfg['arm'], scene=cfg['scene_id'], interpolation=interp_name,
                  h=h, dt=dt, dims=dims, amountRampWidth_m=w, velocityExtension=extension, steps=[])
    previous = load_step(directory, 0, dims)
    for step in range(1, cfg['steps'][-1] + 1):
        current = load_step(directory, step, dims)
        nid, widths = native_layout(current['activity'], dims)
        rho_native, cells = native_means(current['density'], nid, len(widths))
        # The diagnostic density is the native mean expanded over its fine cells.
        spread, _ = native_means(np.abs(current['density'] - rho_native[np.maximum(nid, 0)]) * (nid >= 0), nid, len(widths))
        assert spread.max() < 1e-5, f'density not piecewise constant per native cell at step {step}: {spread.max()}'
        target = np.clip(rho_native, 0.0, 1.0)
        known = previous['density'] >= 0.5
        entry = dict(step=step, time_s=step * dt, nativeCells=int(len(widths)),
                     solverAmount_m3=float(current['density'].sum() * h ** 3),
                     solverMassAboveOne_m3=float(np.maximum(current['density'] - 1, 0).sum() * h ** 3),
                     refSphereCenter_m=reference_center(cfg, step).tolist())
        # Model-free diagnostic: solver native means vs the reference amount
        # (the seed ramp integrated over the exact reference geometry).
        ref_center = reference_center(cfg, step)
        ref_phi = np.minimum(np.linalg.norm(lattice.centers - ref_center, axis=1) - cfg['radius'],
                             lattice.centers[:, 1] - cfg['pool']).reshape(dims)
        exact = amount_field(ref_phi, lattice, w)
        exact_native, _ = native_means(exact, nid, len(widths))
        interfacial = (exact_native > 1e-3) & (exact_native < 1 - 1e-3)
        volume = cells * h ** 3
        contact = reference_center(cfg, step)[1] - cfg['radius'] - cfg['pool'] <= 0
        entry['solverVsReference'] = dict(referenceScope='contact; reference geometry invalid') if contact else dict(
            interfacialNativeCells=int(interfacial.sum()),
            fractionError=_ray.statistics((rho_native - exact_native)[interfacial]) if interfacial.any() else None,
            absoluteVolumeError_m3=float(np.abs((rho_native - exact_native) * volume).sum()),
            massOutsideReference_m3=float(((rho_native * volume)[exact_native <= 1e-3]).sum()),
            referenceGap_m=float(reference_center(cfg, step)[1] - cfg['radius'] - cfg['pool']))
        entry['solverNativeTrilinear'] = surface_metrics(
            cfg, lattice, lambda pts: lattice.trilinear(current['density'], pts), step, directions)
        entry['shippingMesh'] = mesh_metrics(cfg, current['mesh'], step)
        entry['arms'] = {}
        for name, arm in arms.items():
            phi = arm['phi']
            U, filled = extend_velocity(previous['velocity'], known, phi, h, mode=extension)
            advected = advect(phi, U, lattice, dt, interp)
            redistanced = redistance(advected, h)
            result = dict(velocityFilled=bool(filled.all()))
            if arm['mode'] is None:
                final = redistanced
            elif arm['mode'] == 'bodies':
                corrected, stats = correct_bodies(redistanced, lattice, current['density'], nid, h, w)
                result['correction'] = stats
                final = corrected  # correct_bodies redistances before its shift; the shift is last
            else:
                corrected, shift, stats = correct_mass(redistanced, lattice, target, nid, widths, cells, h, w, arm['mode'])
                result['correction'] = stats
                final = redistance(corrected, h, iterations=12)
            np.save(out / f'phi-{name}-step-{step}.npy', final.astype(np.float32))
            normal_f, _ = unit_normal(final, h)
            fraction = plane_fractions(final, normal_f, h)
            amount = amount_field(final, lattice, w)
            F_native, _ = native_means(amount, nid, len(widths))
            mismatch = (F_native - target) * volume
            result['bandVolume'] = body_volumes(cfg, lattice, fraction, step)
            result['bandAmount'] = body_volumes(cfg, lattice, amount, step)
            result['bandMinusSolver_m3'] = float(amount.sum() * h ** 3 - current['density'].sum() * h ** 3)
            result['nativeMismatchAfterStep'] = _ray.statistics(mismatch)
            result['liquidInInactiveCells_m3'] = float(fraction[nid < 0].sum() * h ** 3)
            result['surface'] = surface_metrics(cfg, lattice, lambda pts: 0.5 - lattice.trilinear(final, pts) / h,
                                                step, directions)
            entry['arms'][name] = result
            arm['phi'] = final
        entry['solverVolume'] = body_volumes(cfg, lattice, np.clip(current['density'], 0, 1), step)
        report['steps'].append(entry)
        if step in section_steps:
            points = _section_points(lattice)
            fields = [(lattice.trilinear(arms['A']['phi'], points), (0, 220, 255)),
                      (lattice.trilinear(arms['C']['phi'], points), (0, 255, 90)),
                      (lattice.trilinear(arms['D']['phi'], points), (255, 0, 200))]
            write_png(out / f'section-step-{step}.png', section_image(cfg, lattice, current['density'], fields, step))
        previous = current
        print(json.dumps(_summary_line(entry)))
        sys.stdout.flush()
    (out / 'replay.json').write_text(json.dumps(report, indent=1))
    return report


def _section_points(lattice, pixels_per_cell=6):
    nx, ny = lattice.dims[0], lattice.dims[1]
    W, H = nx * pixels_per_cell, ny * pixels_per_cell
    xs = lattice.origin[0] + (np.arange(W) + 0.5) * lattice.h / pixels_per_cell
    ys = lattice.origin[1] + (np.arange(H) + 0.5) * lattice.h / pixels_per_cell
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], -1)


def _mm(stats, key='maxAbs'):
    return None if stats is None else round(1e3 * stats[key], 3)


def _summary_line(entry):
    sv = entry['solverVsReference'].get('fractionError')
    line = dict(step=entry['step'], gap_mm=round(1e3 * entry['solverVsReference'].get('referenceGap_m', float('nan')), 1))
    line['solverFracErrMax'] = None if sv is None else round(sv['maxAbs'], 4)
    for name, arm in entry['arms'].items():
        s = arm['surface']['sphere']
        p = arm['surface']['pool']
        line[f'{name}_sphere_rms_mm'] = _mm(s.get('radialError_m'), 'rms')
        line[f'{name}_sphere_max_mm'] = _mm(s.get('radialError_m'))
        line[f'{name}_pool_max_mm'] = _mm(p.get('heightError_m'))
        line[f'{name}_missing'] = s.get('missingRoots')
        if 'correction' in arm and 'shift' in arm['correction']:
            line[f'{name}_shift_max_mm'] = None if arm['correction']['shift'] is None else round(arm['correction']['shift']['maxAbs'], 2)
            line[f'{name}_halo'] = arm['correction']['haloCells']
        elif 'correction' in arm:
            line[f'{name}_bodyShift_mm'] = arm['correction']['bodyShift_mm']
            line[f'{name}_filled'] = arm['correction']['filledCells']
        line[f'{name}_dV_ml'] = round(1e6 * arm['bandMinusSolver_m3'], 2)
    m = entry['shippingMesh']
    line['mesh_sphere_rms_mm'] = _mm(m.get('radialError_m'), 'rms')
    line['mesh_sphere_max_mm'] = _mm(m.get('radialError_m'))
    line['mesh_pool_max_mm'] = _mm(m.get('heightError_m'))
    n = entry['solverNativeTrilinear']
    line['native_sphere_rms_mm'] = _mm(n['sphere'].get('radialError_m'), 'rms')
    line['native_sphere_max_mm'] = _mm(n['sphere'].get('radialError_m'))
    line['native_pool_max_mm'] = _mm(n['pool'].get('heightError_m'))
    return line


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--interp', choices=('cubic', 'linear'), default='cubic')
    parser.add_argument('--sections', default='2,4,6,7,8,10,12,16,20')
    parser.add_argument('--width', type=float, default=None,
                        help='amount ramp width in metres (default: one fine cell, the production seed width)')
    parser.add_argument('--extension', choices=('normal', 'approach'), default='normal')
    parser.add_argument('--classes', choices=('topology', 'ramp'), default='topology',
                        help='arm D fill/drain class test: topology (default) = solver full/empty while the band '
                             'puts the cell centre on the other side (F0 vs 0.5); ramp = band sub-cell fraction '
                             'vs 0.995/0.005 (the 8 September afternoon rule; fires on every interface-adjacent '
                             'cell of a sharp density, see the handoff)')
    args = parser.parse_args()
    _CLASSES = args.classes
    sections = {int(s) for s in args.sections.split(',') if s}
    replay(args.directory, args.out or (args.directory / f'replay-{args.extension}'), args.interp, sections,
           args.width, args.extension)
