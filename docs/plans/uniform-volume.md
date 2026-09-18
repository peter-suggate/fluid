# Uniform geometric volume method

The new `uniform-volume` method uses the scene's finest lattice throughout.
It retains independent shared-vertex signed distance (phi) and conservative
cell liquid volume (V). It is an explicit picker option, not automatic scene
switching. The adaptive method and its in-progress changes remain intact.

## Storage and execution

- Two `(nx+1)(ny+1)(nz+1)` r32float vertex fields; direct trilinear sampling.
- Existing dense cell volume, MAC velocity and multigrid pressure textures.
- A fixed nine-entry receiver stencil: eight translated-box donors and one
  uncovered-donor identity fallback. No hash tables, incidence graphs, topology
  generations, sparse pages, hanging constraints or indirect cell lists.
- RK2 backtraced vertex phi and receiver boxes; three receiver/donor capacity
  normalization rounds ending in donor normalization. V may exceed capacity.
- Bounded closest-point phi redistancing. Phi is never reconstructed from V.
- Symmetric six-neighbor conservative sharpening toward phi-implied capacity;
  closed solid faces block transfer, and sharpening never writes phi.
- Phi-based pressure/extension membership, ghost-fluid fractions, and the
  adaptive method's capped excess-volume release, `min(.5 excess, capacity)/dt`.
- The existing dense backend supplies velocity extension, multigrid, rigid
  resources, diagnostics and density-encoded phi presentation.

This is a dense specialization, not bitwise parity: the uniform pressure
hierarchy, bounded regular stencil and local sharpening differ from adaptive
execution. Preserve invariants and compare physical results before claiming
numerical equivalence. In particular test closed-box conservation, zero-flow
identity, translation, hydrostatics, walls and disconnected interfaces.

## Validation and performance

Use Dawn under the repository WebGPU lease, sequentially. Add focused numerical
GPU tests, a reproducible small-scene benchmark, type/module checks and run the
unchanged canonical Sparse CM12 regression matrix after integration. Report
warmup separately from steady-state wall time; fence timed GPU work. Compare
matched scene, timestep, pressure quality and simulated duration. A speedup is
an observation to measure, not an acceptance threshold to manufacture.

Dense storage/work scale with domain volume, so this method targets small
scenes. Do not silently reduce resolution to fit device limits.

## Mini32 wall-contact diagnosis (2026-09-18)

The first dense implementation omitted adaptive-volume's closed-wall phi
continuation. Zero normal wall velocity leaves a wall vertex tracing along the
wall, so an initially positive vertex cannot acquire arriving interior liquid.
This creates false air at impact and traps conservative V outside the phi
pressure region. The missing rule now runs inside `uvAdvectPhi`: trace one cell
inside every incident closed plane, continue negative old phi onto the wall,
and retain the existing released-wall air rule. No dispatch or buffer was added.

At 30 frames (1 s), represented-volume drift changed from approximately -79%
to -1.9%; raw V drift was 7e-7. At 90 frames, maximum cell V was 1.021 and raw
V drift 2.2e-6. Phi drift was still -8.1%: numerical equivalence with adaptive
is not established. The extra MacCormack phi passes, piecewise-trace experiment,
and reordered extra extension were removed after they did not resolve the issue.

Nine focused Dawn checks pass, including closed-wall arrival, 90-frame mini32,
independent dense V/phi slice sampling, conservation, translation, sharpening,
and hydrostatics. Five CPU checks pass. The full sparse regression invocation
was refused because another task held the repository WebGPU lease. The existing
sparse overlay readback test previously failed on a missing capacity sample;
that remains separate from the passing dense overlay check.
