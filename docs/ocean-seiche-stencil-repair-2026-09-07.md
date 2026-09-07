# Default ocean-seiche transport-stencil repair

The supplied balanced-default configuration halted at zero-based frame 35
(time 1.2 s), in `prepareSparseCM12InteriorFaceTiles`, at (56, 39, 63).
Its accepted 32/16/4/2-width junction makes the projected Newton iteration
cycle. The final proposal can lie inside the parameter cube despite a large
geometric residual. Nearby queries also require more than eight primal-vertex
crossings, or more than eight Newton iterations near a collapsed corner.

The ordinary eight-hop/eight-iteration locator remains the first search.
Only failed queries retry with a width-dependent traversal budget, sixteen
Newton iterations, and the exit proposal associated with the closest iterate.
Positive shape weights, exact cell-centre identity, and the existing first-moment
acceptance tolerance remain unchanged. Genuine unlocatable geometry still halts.

## Verification

The focused shader tests include the captured junction, the later collapsed
corner, deterministic nearby samples, positive partition and affine reproduction,
exact nodal weights, and bit-exact preservation of successful ordinary stencils.

The supplied scene is retained in
`tests/fixtures/sparse-cm12-ocean-seiche-ui-stencil.json`. The production-method
regression executes all 36 steps through the reported halt, checks every failure
receipt, and checks finite velocity, nonnegative density, and mass drift below
0.1%. It passed with the corrected shader. Run it with:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --max-old-space-size=12288 --import tsx --test tests/sparse-cm12-ocean-seiche-stencil-dawn.test.ts
```

`FLUID_OCEAN_STEPS=600` requests the full authored duration. This longer trajectory
has **not** been validated: the attempted run was stopped after ten minutes in
topology preparation. Node's default heap also proved insufficient for that
investigation; the larger heap changes the test process, not simulation settings.

The canonical Dawn gate was run without changing any thresholds. It is not green:
the horizontal density-symmetry check also fails with the pre-repair shader in
an isolated baseline run, and other lanes encounter timing limits. Browser
rendering was stopped at the start of the final focused reproduction and gate,
but the tab was subsequently resumed with another running scene. The timing
results therefore cannot establish isolated performance.
