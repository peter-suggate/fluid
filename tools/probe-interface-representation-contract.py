"""CPU research oracle for interface information and zero-time remapping.

This does not execute CM12, infer normals, advect liquid, or validate GPU cost.
It asks whether a retained plane survives conservative repartitioning, and
whether volume plus centroid can distinguish a single body from two bodies.
Run: python3 tools/probe-interface-representation-contract.py --output <json>
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


def plane_fraction(normal, alpha, lo, hi):
    """Exact box volume fraction for normal.x <= alpha (float64 arithmetic).

    Inclusion-exclusion integrates the positive orthant simplex after mapping
    the box to independent uniform coordinates. Zero normal components reduce
    its dimension. No spatial quadrature participates in the remapping.
    """
    weights = [abs(n) * (b - a) for n, a, b in zip(normal, lo, hi) if n != 0]
    t = alpha - sum(min(n * a, n * b) for n, a, b in zip(normal, lo, hi))
    if t <= 0:
        return 0.0
    if t >= sum(weights):
        return 1.0
    # Complement avoids cancellation near the fully occupied endpoint.
    if t > sum(weights) / 2:
        return 1 - plane_fraction(tuple(-n for n in normal), -alpha, lo, hi)
    terms = []
    for mask in itertools.product((0, 1), repeat=len(weights)):
        shift = sum(w * bit for w, bit in zip(weights, mask))
        terms.append((-1) ** sum(mask) * max(0, t - shift) ** len(weights))
    value = math.fsum(terms) / (math.factorial(len(weights)) * math.prod(weights))
    assert -1e-12 <= value <= 1 + 1e-12, value
    return min(1.0, max(0.0, value))


def volume(lo, hi):
    return math.prod(b - a for a, b in zip(lo, hi))


def recover_alpha(normal, fraction, lo, hi):
    lower = sum(min(n * a, n * b) for n, a, b in zip(normal, lo, hi))
    upper = sum(max(n * a, n * b) for n, a, b in zip(normal, lo, hi))
    for _ in range(64):
        mid = (lower + upper) / 2
        if plane_fraction(normal, mid, lo, hi) < fraction:
            lower = mid
        else:
            upper = mid
    return (lower + upper) / 2


def children(lo, hi):
    middle = tuple((a + b) / 2 for a, b in zip(lo, hi))
    for bits in itertools.product((0, 1), repeat=3):
        yield (tuple(middle[k] if bits[k] else lo[k] for k in range(3)),
               tuple(hi[k] if bits[k] else middle[k] for k in range(3)))


def remap(normal, alpha, lo, hi, depth, mixed):
    """Return leaf mass, local conservation error, and plane displacement.

    Refined children carry fraction and the parent's retained normal. Their
    plane offset is recovered from those data, never from the authored alpha.
    Mixed leaves stop one level earlier on the right half of the root cube.
    """
    fraction = plane_fraction(normal, alpha, lo, hi)
    mass = fraction * volume(lo, hi)
    if depth == 0 or fraction == 0 or fraction == 1:
        return mass, 0.0, 0.0
    if mixed and lo[0] >= 0.5 and depth == 1:
        return mass, 0.0, 0.0
    recovered = recover_alpha(normal, fraction, lo, hi)
    displacement = abs(recovered - alpha) / math.sqrt(sum(n * n for n in normal))
    receipts = [remap(normal, recovered, a, b, depth - 1, mixed)
                for a, b in children(lo, hi)]
    leaf_mass = math.fsum(r[0] for r in receipts)
    error = max(abs(leaf_mass - mass), *(r[1] for r in receipts))
    return leaf_mass, error, max(displacement, *(r[2] for r in receipts))


def planar_experiment():
    lo, hi = (0.0,) * 3, (1.0,) * 3
    normals = [(1, 0, 0), (0, 1, 0), (0, 0, -1), (1, 1, 0),
               (1, 1, 1), (1, -2, 3), (-3, 1, 2), (2, 3, -1)]
    max_mass_error = max_displacement = max_quadrature_error = 0.0
    max_copy_l1 = 0.0
    runs = 0
    for normal in normals:
        extent = sum(abs(n) for n in normal)
        minimum = sum(min(n, 0) for n in normal)
        for position in (0.13, 0.29, 0.5, 0.71, 0.87):
            alpha = minimum + extent * position
            fraction = plane_fraction(normal, alpha, lo, hi)
            # Independent indicator quadrature is only a sanity check of the
            # closed-form integration, with its separate discretization error.
            samples = 24
            count = sum(sum(n * (q + 0.5) / samples for n, q in zip(normal, xyz))
                        <= alpha for xyz in itertools.product(range(samples), repeat=3))
            max_quadrature_error = max(max_quadrature_error, abs(count / samples**3 - fraction))
            copy_l1 = math.fsum(abs(plane_fraction(normal, alpha, a, b) - fraction)
                               * volume(a, b) for a, b in children(lo, hi))
            max_copy_l1 = max(max_copy_l1, copy_l1)
            for mixed in (False, True):
                current_alpha = alpha
                for _ in range(32):
                    mass, error, displacement = remap(normal, current_alpha, lo, hi, 3, mixed)
                    current_alpha = recover_alpha(normal, mass, lo, hi)
                    max_mass_error = max(max_mass_error, error, abs(mass - fraction))
                    max_displacement = max(max_displacement, displacement,
                        abs(current_alpha - alpha) / math.sqrt(sum(n * n for n in normal)))
                runs += 1
    # These budgets concern this float64 oracle only, not production GPU tests.
    assert max_mass_error < 1e-10
    assert max_displacement < 1e-9
    assert max_quadrature_error < 0.035
    assert max_copy_l1 >= 0.49
    return dict(configurations=runs, cycles_per_configuration=32, maximum_depth=3,
                retained_normal_max_local_or_global_mass_error=max_mass_error,
                retained_normal_max_plane_displacement=max_displacement,
                copied_mean_max_child_fraction_integrated_L1=max_copy_l1,
                independent_24_cubed_indicator_max_fraction_error=max_quadrature_error)


def moments(intervals):
    # Extrude each x interval over y,z in [0,1]; these integrals are analytic.
    mass = math.fsum(b - a for a, b in intervals)
    mx = math.fsum((b * b - a * a) / 2 for a, b in intervals)
    return [mass, mx, mass / 2, mass / 2]


def ambiguity_experiment():
    single = [(0.375, 0.625)]
    double = [(0.125, 0.25), (0.75, 0.875)]
    assert moments(single) == moments(double)
    assert all(b <= c or d <= a for a, b in single for c, d in double)
    symmetric_difference_volume = moments(single)[0] + moments(double)[0]
    return dict(single_body_x_intervals=single, two_body_x_intervals=double,
                shared_mass_and_first_moments=moments(single),
                connected_components=[1, 2],
                symmetric_difference_volume=symmetric_difference_volume,
                conclusion="Volume and first moments do not uniquely determine arbitrary interface topology.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = dict(schema=1, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="CPU float64 representation oracle; no production solver, normal estimator, transport, or GPU validation",
                  planes=planar_experiment(), ambiguity=ambiguity_experiment())
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
