# Twin-dam coarse-surface settling investigation

1 October 2026. Disabled in production at the user's request after review.
The UI toggle and runtime parameter have been removed; saved experimental
settings are ignored. Dynamic coarsening keeps liquid surface tiles fine.
The low-level experimental implementation and diagnostic evidence below remain
for investigation. The candidate did not complete validation: the still-pond
regression remained unresolved. This is not a validated settling fix.

## Reproduction and isolation

`twin-dam-collision`, Uniform Geometric balanced, dynamic ownership, 0.5h
surface tolerance, fixed 1/60 s. All GPU runs are serial under the repository
lease, with no browser simulation running.

Per-step coarse redistancing changes the sampled interface even without any
advection or force. A frozen curved-surface fixture moves an edge crossing by
0.151487h in one rebuild. The error is spatial, but the old schedule applies
it once per small timestep. The subsequent global surface-volume correction
can restore total represented volume without restoring its distribution.
This repeatedly separates the surface from transported liquid and drives
volume repair during settling. Small pressure residuals do not diagnose it.

The diagnostic records canonical GPU ownership, volume after transport and
sharpening, geometry, and velocity after advection, forces, coarse projection,
and fine projection. Its owner-weighted positive-face kinetic measure is a
proxy, not an exact MAC kinetic-energy budget. Positive centre phi alone is
not proof of orphaned liquid: partially filled cells can legitimately have
air centres. Excess above cell capacity is reported separately.

In the isolated cadence A/B, mean excess volume over seconds 20–30 falls
from 9.600% to 0.246% of the water. Over seconds 50–60 it falls from 9.793%
to 0.377%. The cadence arm also completes 100 simulated seconds; mean excess
over seconds 20–100 is 0.352%. Motion remains, without adding damping.

## Candidate

Accumulate actual characteristic travel at coarse vertices in owner-cell
units, in the existing advection kernels. Rebuild nearby coarse interface
distance after one cell traversal and retain the fractional remainder.
Fine vertices reset their aligned slots. Far-air reconstruction/retirement
keeps its old schedule because residency relies on those certificates.
The existing smooth-surface toggle selects this behavior; disabling it
selects the original force/reconstruction behavior.

The new GPU regression verifies that stationary coarse surfaces remain fixed
across repeated calls, and that actual characteristic travel still triggers
rebuilding. The retained long-dam test keeps its three-slot impact check,
mass/front assertions and residency checks.

Cadence alone is insufficient for the cost constraint: an initial throughput
pair measured 11.973 versus 11.670 ms/step. Correcting the surface changes the
amount and arrangement of mixed-boundary work. A companion cache reuses
owner normals and curvature across incident faces, borrowing dead scratch
memory. It preserves the force formulas, not necessarily float32 trajectory
bits. The inline fallback and still-pond regression must pass independently.

## Rejected experiments

Permanently preserving coarse cut-cell corners greatly reduces compression
but changes the long-dam impact's pressure demand. A per-step correction clamp,
wall-contact changes, cubic-gradient sampling, and a wall-gradient change
were insufficient. They are not production changes. A first travel-cadence
version delayed far-air retirement and failed a residency certificate; the
candidate retains that cleanup. No assertions or timing limits were relaxed.

## Reproduction tools

`tools/probe-uniform-settling-dawn.ts` accepts `--scene=twin-dam-collision`,
`--dt=0.016666666666666666`, `--frames`, `--every`, and `--quality-every`.
`--control=every-step` restores the old reconstruction and inline forces;
`--control=cadence-inline` isolates the scheduling change;
`--control=every-step-cached` isolates force caching. Default is the candidate.
Pass `--values='{"timeStep":"scene"}'` for the fixed scene step.

`tools/probe-uniform-stage-scaling-dawn.ts --scene=twin-dam-collision
--throughput --warmup=1200 --frames=600` measures seconds 20–30 with two frames
in flight, no timestamps/readbacks in the timed interval, and no rendering.
The old arm adds `--rebuild-every-step --inline-curvature`. Captures record
source fingerprints before and after; source-changing runs are excluded.

Raw captures, rejected patches and logs are retained in
`artifacts/uniform-settling-ab/` (generated and ignored).
