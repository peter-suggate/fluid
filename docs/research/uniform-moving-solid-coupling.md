# Uniform Geometric: moving solids and live voxel displacement

2026-09-23

## Reproduction

Run exclusively (no browser GPU scene or other Dawn process):

```sh
npm run test:dawn:uniform-moving-solid
```

The production Uniform Geometric options run a 16³, 0.8 m tank half full,
zero gravity/viscosity/surface tension, at 1/120 s. One fixture lowers a
0.3 × 0.2 × 0.3 m prescribed box from air into water; another fills the
same submerged 6 × 4 × 6 block through the public live voxel edit API.
Both continue for 48 steps with the production 1e-3-cell dust floor,
accounting for its separately reported discarded mass and quantization bound.
The submerged edit is then undone with dust removal disabled for an exact
conservation check. A further 32 lateral body steps measure advance cost
without field readbacks. These checks inspect the actual conserved field, rather
than just a rendered surface or total-volume diagnostic.

Before the fix:

| Fixture | Initial water (cells) | Final water | Maximum loss | Flow outside edited region |
| --- | ---: | ---: | ---: | ---: |
| Descending box | 2048 | 1998.6010 | 2.4121% | 0.5305 m/s peak |
| Submerged voxel fill | 2048 | 1904 | 7.03125% | exactly zero |

The voxel case erased exactly its 144 covered water cells. The previous
live-solid-edit test only placed a wall ahead of a dam front in dry space.
It established obstruction/removal, not displacement of existing water.

## References and coupling contract

The saved primary sources are:

- [BBB07, Fast Variational Solid–Fluid Coupling](../papers/BBB07_Fast_Variational_Solid_Fluid_Coupling.txt), §§2–4, especially Eqs. 6, 10, 13 and 15.
- [CM11a, A Multigrid Fluid Pressure Solver Handling Separating Solid Boundary Conditions](../papers/A_Multigrid_Fluid_Pressure_Solver_Handling_Separat.txt), §3.1.
- [CM12, Mass-Conserving Eulerian Liquid Simulation](../papers/massConservingLiquids.txt), §§3.6–3.7.

Public originals: [BBB07 author repository](https://uwspace.uwaterloo.ca/items/f5cb877f-1b56-4ff5-81d9-f7473a5e7f1a/full),
[CM12 author PDF](https://matthias-research.github.io/pages/publications/masscon_sca.pdf).

Pressure must enforce wall-relative nonpenetration, allowing separation
through nonnegative solid-contact pressure. Transport face aperture and
pressure face-centred dual volume are different quantities. The existing
solver already distinguishes them and has a separating pressure active set.

For prescribed rigid motion the pressure RHS must vanish when the liquid
and rigid body share the same translation (and for rigid rotation away
from external walls). Written with non-solid dual fraction W and cell
capacity C, each positive/negative face contributes

```
D = Σaxis [ W+ u+ + (C − W+) us+ − W− u− − (C − W−) us− ] / haxis
```

This is `div(W (u−us)) + C div(us)`; rigid velocity has zero divergence.
The previous implementation used `(W−C) us` with a plus sign and omitted
`1/h` on just the solid term. That drives the wrong wall-relative pressure
response and makes its strength depend on world units. CM11a's printed
Eqs. 9–10 are misleading in this respect; the correction follows BBB07's
prescribed-solid variational formulation and the co-moving null-mode test,
not a literal transcription of that line.

Pressure alone cannot recover volume that transport has already deleted.
CM12 §3.6 separately moves excess water out of newly covered solid cells.
Uniform Geometric disabled the older density solver's correction. Its
fallback identity edge was then multiplied by zero receiver capacity,
so closed donors vanished during normalized gathering.

## Changes

- Reconcile covered water on the GPU before work classification, extension,
  and geometric transport. This ensures receivers reached by a thick edit
  are included in the same step's work lists.
- For geometric mode, redistribute excess to the nearest open axial shell,
  splitting equal-distance directions. This works for voxel edits without
  an analytic solid SDF and for multiple-cell body entry. Integer remainder
  assignment preserves each rounded donor's total. Open cells keep their
  existing volume; excess in ordinary fluid remains the pressure solver's
  responsibility.
- Run this work while rigid bodies are present or after a solid-mask edit;
  static body-free scenes incur no extra displacement dispatches.
- Correct the wall-relative pressure RHS, including cell-size scaling.
- Sample rigid velocity at the same MAC location as fluid velocity. The
  dual-volume quadrature previously averaged velocity at covered sample
  points, shifting it toward the solid centroid. An off-axis co-rotation
  test reproduced a spurious 0.20000076 s⁻¹ divergence from this mismatch;
  coverage quadrature now determines volume/ownership only.
- Sample prescribed pose motion at simulation advances. Held/static bodies
  get translation and angular velocity from successive poses; stopping the
  pointer stops the wall source even if its last event carried a nonzero
  velocity. Rotating a held body now supplies angular wall velocity even
  though the controller's pose command clears angular velocity. Free GPU
  bodies and the first grab retain their explicit command/state.
- If every axial escape is sealed, retain and report the unresolved water
  rather than erase it. Subsequent solid edits retry reconciliation.

## Limits: displacement is not a turbulence model

The relocation is a discrete edit repair, not an exact swept-volume remap.
Nearest axial exits are an approximation to the solid-distance gradient;
large instantaneous fills have no unique physical motion path. The search
has a domain-sized bound and only runs for cells with covered water.
An edit leaving no accessible capacity cannot have an incompressible
solution; retained unresolved water is diagnostic storage, not a valid
fluid state inside a rigid body.

The normal boundary condition does not impose tangential no-slip drag.
For example, an inviscid spinning sphere with no normal surface motion
need not stir surrounding water. Wakes/splashes depend on resolved geometry,
resolution, velocity advection and viscosity; arbitrary vorticity injection
would not repair broken displacement.

Free dynamic bodies still use the existing approximate reaction/drag and
buoyancy integration. This is not BBB07's simultaneous fluid/body pressure
solve with the `Jᵀ M⁻¹ J` contribution. The prescribed-motion fixes do not
establish added-mass stability or exact two-way momentum conservation for
light freely floating objects. That remains a distinct solver extension.

## Validation

The focused moving-solid Dawn run passes the translation invariant (exactly
zero measured divergence), body entry, submerged voxel insertion, removal,
and a fully sealed/unsealed edit. Conserved volume remaining after 48 steps:

| Fixture | Water remaining | Reported dust removal | Dust quantization bound | Unexplained loss beyond bound |
| --- | ---: | ---: | ---: | ---: |
| Descending held box | 2045.459060 | 2.483578 | 0.165125 | 0 |
| Submerged voxel fill | 2044.036663 | 3.861203 | 0.280047 | 0 |

Both fixtures leave zero water in the submerged block and drive external
fluid (peak speeds 0.6495 and 2.7960 m/s respectively). The zero-gravity,
instantaneous voxel insertion is an impulsive test, not a settled-height
benchmark. Dust-accounted loss and water creation must remain below 1e-4
of initial volume; the original block-deletion defect cannot pass this gate.
Undo and sealed-reservoir recovery are additionally checked with the dust
sink disabled. A direct GPU regression splits four fixed-point units across
six open neighbors, verifying exact conservation and nonnegative deposits
when independently rounded shares would otherwise overdraw the donor.

The 32 subsequent lateral body advances average **12.13 ms/advance** on the
local Dawn/Metal device at 16³, including completion fencing but excluding
field readbacks, rendering and startup compilation. This is a small-fixture
measurement, not a large-world performance guarantee. Thick instantaneous
fills deserve separate workload budgets because the remap searches farther
than ordinary sub-cell body motion.

All six existing uniform geometric boundary scenarios passed, including
figure 8's curved shell and figure 12 rebound. The existing live solid edit
Dawn test passed. Three prescribed-motion unit tests pass. The additional off-axis co-rotation invariant passes at 1.97e-6 s⁻¹
(previously 0.20000076 s⁻¹). The canonical Sparse CM12 suite was run with
unchanged limits: its initial attempt hit existing lane timeouts in symmetric
expansion (20 s), topology page budget (30 s), and hydrostatic adaptivity
(45 s); another task then acquired the GPU lease between lanes and blocked
later lanes from starting. The second run also remained red (4/17 lanes passed): multiple process
timeouts, mini64 median frame cost 212.47 ms against the unchanged 110 ms
ceiling, a terrain topology halt, and exhaustion of the 480 s suite budget.
The terrain halt was `MISSING_COMPILED_TOPOLOGY_FACE` at frame 17,
generation 18, owner 595, operands `22,1,1,1`. An isolated checkout of original commit `78ef5c2ded0f021fdf4a740ac9c251b003df8c2c`
also fails the terrain lane, with a different halt: `GEOMETRIC_VOLUME_TRANSPORT`
in `addWholeFrameUncoveredDonorFallbacks` at frame 49, generation 50,
owner 44383, operands `21,0,27.87582778930664,64`. This establishes that the
terrain lane was already failing, not that the two errors are identical.
The temporary control worktree was removed. These results are not a clean
regression gate and must not be represented as one.

The machine-readable measurements and both full-suite receipts are in
[uniform-moving-solid-coupling-results.json](uniform-moving-solid-coupling-results.json).

Repository-wide `npm run check:types` reports errors in unrelated sparse
tests/tools; none point to these coupling
changes. No existing regression assertion or timing ceiling was changed.
