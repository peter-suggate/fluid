# Stage 2 Browser Shell — Tests and Acceptance Criteria

These tests were defined before the Stage 2 implementation. Stage 2 contains no
fluid solver and makes no physics-validity claim.

## Functional gates

| ID | Claim | Automated acceptance |
|---|---|---|
| S2-01 | Default scene is valid SI input | schema validator reports no issues |
| S2-02 | Scene JSON round-trips | canonical serialized bytes are identical |
| S2-03 | Camera reset is deterministic | pose equals the documented default |
| S2-04 | Camera basis is orthonormal | dot errors `< 1e-10`, norm errors `< 1e-10` |
| S2-05 | Seeded generator is reproducible | first 1,000 values are byte-identical |
| S2-06 | Invalid physical input is rejected | negative dimensions/density and invalid fill fail |
| S2-07 | WebGPU capability is explicit | supported or unavailable state is visible; never silently falls back |
| S2-08 | Render loop survives resize | positive canvas backing dimensions and no uncaught error |
| S2-09 | Controls support pause/reset/step | state transitions are deterministic |
| S2-10 | Export includes reproducibility metadata | schema/build/browser/precision keys present |

## Interaction gates

- Orbit, pan, zoom, reset, and front/side/top camera actions are available.
- Eulerian, particle, and comparison presentation modes are selectable.
- Scientific and presentation views are distinct.
- Container, fill, gravity, density, viscosity, and nominal resolution controls
  show SI units.
- Save, load, reset, and metric export actions are accessible by keyboard.
- The layout remains usable at 1440x900 and 1024x768.
- Reduced-motion preference disables nonessential animation.

## Diagnostic gates

- Adapter state, frame time, presentation time, canvas resolution, build ID,
  precision mode, and simulation state are visible.
- Validation results show measured value, threshold, and pass/fail.
- Unsupported WebGPU and device loss produce an actionable status rather than a
  blank canvas.

## Stage gate

Stage 2 passes only when the production build, deterministic unit suite, and
browser interaction checks pass. GPU visual output is evidence for renderer
operation only, never for fluid correctness.
