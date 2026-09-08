# Production surface follow-up — 8 September 2026

The handoff is **not fully implemented**. General current-field transport is
still absent: production continues to transport native means and fit
`a*q_seed+b`. No translation carrier, shape fitting, mesh smoothing, projected
surface, or analytic animation was introduced.

## Production correction

`cm12RetainedDensityPhiAtFine` now delegates to a tested density companion.
Strictly wet/dry support ranges retain their sign shortcuts. When a range
endpoint equals half density, it evaluates the actual clamped seed density at
the queried point. An endpoint equality no longer declares the whole support
to be surface. Real half-density plateaus remain zero. The interior expression
forms the density residual before division to preserve the tested exact roots.

This repairs the third defect in the handoff. It does not repair center-mean
transport diffusion or missing spatial transport into previously dry supports.

## Reviewed real-gravity capture

`tools/capture-retained-falling-velocity-dawn.ts` remains read-only with respect
to physics. Its adapter now:

- Captures VEX accepted depth and requires depth 0–8 plus effective validity.
- Reads destination density and the destination collocated velocity bank at
  projection. Stage captures precede frame commit, so both use their respective
  source parity XOR 1. Scalar and face parity are captured independently.
- Excludes invalid velocity samples from fits and local neighbor derivatives;
  reports geometric and excluded counts. Projection statistics cover wet
  cells only, not a purported freshly extended projected air halo.
- Measures group mass independently of velocity eligibility and records total
  native liquid volume. These are native means times native support volume.
- Refreshes native ownership and GPU buffers after deferred topology work,
  checks unique IDs and nonoverlapping physical support, and requires exactly
  one VEX and one projection snapshot per encoded step.
- Verifies accepted native width one for the fully fine arm, including runtime
  leaf enumeration. Runtime pages are represented over their full support;
  the diagnostic is scoped to the quarter scene, not a general clipped-domain
  or solid-volume integration adapter.

The exact quarter-scene query is `0_0_0_25_66.6667_100_8_8`, with balanced,
coarse-first and paper timestep defaults. The fine arm imposes min=max=1 over
the physical domain. Gravity is authored production gravity, not prescribed
velocity. Six steps and twelve snapshots per arm were captured in
`artifacts/retained-falling-velocity-reviewed-final/quarter/{fine,coarse}/`.

At projection step six, the maximum sphere-bulk velocity residual against the
discrete gravity reference was approximately 0.000012 m/s fully fine and
0.033 m/s adaptive. Maximum selected pool speed was 0.00237 m/s fully fine and
0.150 m/s adaptive. Fully fine projected sphere-bulk velocity follows the
gravity reference throughout all six steps. Later valid halo samples deviate;
bulk and halo measurements must remain separate. Final reviewed capture times
were 20.570 s fully fine and 18.472 s adaptive. Total native liquid volume
changed from 1.089777600 to 1.089778082 m³ fully fine and from 1.089777572
to 1.089777177 m³ adaptive between projection steps one and six. Near-constant
global mass is not evidence of correct spatial geometry.

This narrows the diagnosis: incorrect bulk acceleration does not explain the
early fully fine density transport defect, while the adaptive arm also has a
measurable physical velocity discrepancy. These velocity measurements do not
establish correct evolved geometry or pool flatness.

## Shipping geometry and UI

Both shipping GPU mesh captures completed at steps 0, 1, 2 and 6: 21.720 s
fully fine and 24.597 s adaptive. The fully fine mesh capture also verified
accepted native widths. Saved triangles are in
`artifacts/retained-surface-endpoint-fix/quarter/{fine,coarse}/`.
The [section comparison](../artifacts/retained-surface-endpoint-fix/quarter/section-comparison.png)
still shows distorted evolved spheres in both arms at 0.2 s, with additional
adaptive asymmetry. This is a failed geometry acceptance, not a solved scene.

The existing QA renderer's sphere reference incorrectly used continuous
`g*t²/2`. It now uses `g*dt²*n*(n-1)/2` at completed production steps, consistent
with transport before the gravity kick. Only the QA reference is changed;
saved production triangles are untouched. The reference was checked at
steps 0, 1, 2 and 6.

The fully fine scene was opened in the actual UI after all Dawn work finished.
At 99 s the browser still displayed “Finalize sparse presentation resources”,
7/10 tasks, and disabled simulation controls. The temporary tab was closed to
release the GPU. No successful interactive UI acceptance is claimed.

## Validation

- Production companion GPU test: 5/5 tests, including 414 exact sign/zero cases
  and the captured half-drained sphere support counterexample.
- Existing retained scene and motion negative controls: 25/25 CPU tests.
- Reviewed velocity analysis: 5/5 CPU tests, including affine/non-affine fields,
  mixed native offsets, independent parities and rank-deficient coverage.
- Unchanged canonical Dawn suite: **failed**, 180.037 s, five lanes passed,
  six timed out, six were not run. Timeout lanes: symmetric expansion, topology
  page budget, hydrostatic adaptivity, mini32 correctness, mini32 performance,
  mini64 performance. Same lane outcome as the handoff baseline; no ceiling
  changes. Log: `/tmp/fluid-surface-canonical.log`.
- Repository-wide TypeScript check fails in existing unrelated files; the
  focused velocity analysis/test strict type check passes.

The remaining handoff work is the coupled spatial density/momentum transport
replacement, consistent gamma/sharpening/solid/injection behavior, and its
fully fine/adaptive UI acceptance. The endpoint correction must not be treated
as that cutover.
