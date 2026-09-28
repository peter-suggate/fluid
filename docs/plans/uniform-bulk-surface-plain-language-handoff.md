# Handoff: current Uniform Geometric defaults

This describes the working-tree implementation inspected on 28 September 2026, including staged and unstaged changes. It replaces the earlier description of an independent h surface overlay over 4h bulk. It is a code review of the current defaults, not a new simulation or timing result.

## The approach now

**Solve the large-scale pressure at 4h everywhere, then solve the surface detail at h in a narrow band.** Keep the bulk away from the surface coarse. Keep the surface tiles themselves fine.

Here, h is the smallest cell spacing. A 4h cell is four times wider in every direction and covers the space of 64 h cells. Pressure corrects velocity so that water does not compress. The level set describes the water/air boundary; a separate volume field accounts for how much water there is.

Think of the pressure calculation as two passes over the same water:

1. **Get the overall flow right on the large grid.** A global 4h solve handles the whole domain and supplies the amount of flow across coarse faces.
2. **Refine the surface answer locally.** Selected h surface tiles solve again, keeping the flow imposed at their boundary by the first pass. They can improve local surface velocities without choosing a different flow into or out of the surrounding bulk.

These are not two independent bodies of water. They are two resolutions of the pressure calculation. The fine pass starts from an interpolated coarse pressure answer and writes its corrected velocities into the simulation field.

## What changed from the previous handoff

The previous proposal tried to keep the moving front's bulk at 4h while retaining an independent h surface shape and a separate h velocity overlay. That experiment did not establish correct dam-front motion. The present defaults take a more conservative route:

- **Surface tiles remain h.** In dynamic mode, band pressure overrides the shape and speed exceptions that would otherwise allow surface tiles to coarsen. Fast motion does not make those surface tiles 4h.
- **The surface field follows mixed-grid ownership again.** Fine tiles have fine surface samples; coarse tiles use coarse corner information. Advection and redistancing write the samples owned by that grid, and a resolve pass reconstructs missing samples at mixed boundaries. An h-sized texture does not mean every texel is an independently maintained h detail.
- **The retained h velocity overlay is bypassed in the default pressure mode.** Ordinary mixed-grid velocity extension and sampling remain active. The local h pressure solve supplies fine velocities directly where the simulation is fine.
- **The default no longer builds a graded pressure layout midway through each frame.** Its global pressure grid stays all-4h; the GPU constructs a list of fine surface tiles for the local pass.

This is a material compromise against the original request: **the default is coarse bulk with a fine surface band, not a 4h moving front carrying an independent h overlay.** The added band reach is still zero, but the surface tiles themselves are fine.

## How it differs from 2h seams

| Aspect | Graded pressure option | Default: 4h + h band |
| --- | --- | --- |
| Overall pressure calculation | One coupled mixed-resolution solve | Global 4h solve followed by a local h solve |
| Transition | h → 2h → 4h pressure cells | No 2h pressure transition cells |
| How fine and coarse communicate | Through connections inside the coupled solve | Coarse projected face flow becomes the fine band's boundary condition |
| Moving pressure layout | Rebuild/adopt the graded layout, including a mid-frame CPU readback | Fixed coarse grid; build the band list and rows on the GPU |
| Fine correction's influence | Part of the coupled answer | Does not feed back into a second global coarse solve in the same frame |
| Static solids | Supported path | Not supported; automatically uses Graded |

There is still an interface between fine and coarse work. The default removes the **2h pressure seam machinery**, not the need to make fine and coarse answers agree. Bulk transport, surface sampling, and relayout still contain mixed-grid handling.

Also, “no mid-frame pressure-layout readback” does not mean “no CPU waits anywhere.” End-of-frame receipts and dynamic bulk-layout adoption still involve readback.

## What one default frame does

The simulation extends velocity near the water, moves the surface and conservative water volume, and updates momentum and forces. Surface rebuilding uses the mixed grid. Total surface-volume correction is On by default: it moves the existing surface to match total conservative volume. That is a global amount constraint, not proof that the water is in the right places.

Before the coarse pressure transfer overwrites temporary fields, the GPU identifies eligible h surface tiles and prepares their fine pressure equations from the current geometry and forced velocities. Neighbor/face classifications and coefficients are prepared once for this local solve and reused by its iterations.

The global 4h solve then runs with convergence checks. Its projected velocities are transferred back to the simulation grid. The fine band starts from the coarse pressure, performs **eight red-black relaxation sweeps**, and projects local velocities while retaining coarse boundary fluxes.

The band is made from existing h simulation tiles with a recognized liquid/air surface face. It does not independently introduce h pressure inside an arbitrary 4h simulation tile. This is why the current default keeps surface tiles fine.

The fine band's air classification also follows the coarse solve's interpretation. Air pockets or wall slivers that the coarse grid treats as liquid are not automatically given zero air pressure in the fine pass. The code restricts which fine air cells count as open air to avoid releasing pressure against boundary flows computed for liquid. That protects consistency, but means the local solve cannot freely recover every feature the coarse solve missed.

## Defaults and UI caveats

| Setting | Current default/effective behavior |
| --- | --- |
| Coarsening | Dynamic |
| Pressure solve | **4h + h band**; static-solid scenes use Graded |
| Coarsening band reach / hysteresis | 0 / 0 |
| Coarse surface tolerance / fast surface travel | UI values 0.5 / 4, but both effectively zero for dynamic band pressure: surface tiles stay fine |
| Extended h surface velocity | UI default On, but bypassed for band pressure; applies to the Graded path |
| Total surface volume | On |
| Surface-deficit balancing | Off |

Changing **Pressure solve** rebuilds the solver and resets simulation time. The old **Extended h surface velocity** switch is not an On/Off comparison of the current default algorithm. Its label and hint do not currently make that limitation clear. Some older coarsening help text also still mentions a 2h collar without distinguishing pressure modes.

Saved scene URLs and explicit parameters can override defaults. Select the pressure mode explicitly when comparing runs, and check whether the scene contains static solids.

## Main gaps

1. **The original 4h-front goal is not delivered.** Surface pressure is only refined where simulation tiles are already h. Allowing fast surface tiles to remain 4h again would require a working independent fine correction and surface representation over them; simply restoring the speed exception leaves those tiles without the local h solve.

2. **The fine correction cannot repair a wrong global coarse flow.** Its boundary flow is fixed by the coarse answer, and there is no same-frame feedback solve. Thin fronts, wall sheets, and small air features are therefore important validation cases. This is a structural limitation to investigate, not a claim that a particular current scene has been measured failing.

3. **Fine-band convergence is measured but not enforced.** The frame checks global pressure acceptance and band capacity overflow. It records the band's residual, but does not reject a large or nonfinite band residual before publishing the local projection. Eight fixed sweeps are not evidence that every band has converged. A successful global pressure receipt alone is not enough to establish the quality of the final velocity field.

4. **Correct motion and the 1–2% timing target still need current evidence.** The earlier handoff's 76/149 dynamic front measurements belonged to the superseded overlay experiment. They must not be presented as current-default results. This documentation update runs no GPU tests and establishes no new pass or failure. Compare front position, water in the leading strips, wall behavior, conservation, and full-frame timing on the current code.

5. **Scope and cost remain limited.** Static solids fall back to Graded. The local band has capacity for at most 8,192 tiles and throws on overflow. Fixed GPU dispatches avoid a band-list readback, but still do work; fine surface tiles, bulk relayout, global volume correction, and remaining mixed-grid operations also cost time. Removing 2h pressure seams does not by itself prove the one-tile penalty is gone.

6. **The UI does not yet fully describe the effective algorithm.** The overlay switch can say On while being bypassed, and two coarsening settings are overridden in band mode. The grid overlay must be interpreted by stage: bulk ownership, global pressure, and the local pressure band are different grids.

## Next handoff

First validate the current defaults as they stand, keeping the original acceptance bounds. Explicitly compare **band versus graded**, with all-fine bulk, the reported small coarse corner, and dynamic coarsening. Even all-fine *bulk* still uses a global 4h pressure stage when Band is selected, so label that control accurately. Do not mix pressure modes silently or use old overlay results as its baseline.

Inspect both global and band residuals alongside the actual front and leading-strip water. Establish a proper acceptance policy for the final local correction. Then measure full-frame cost, including dynamic classification and remap. Keep GPU runs isolated from browser simulations and other Dawn processes.

The next architectural decision is whether this fine-surface-band compromise is acceptable. If the requirement remains a moving front whose bulk is 4h, that is additional unresolved work; do not describe the present default as achieving it. Do not widen the band or loosen correctness/timing limits to hide failures.

Source map for the next engineer:

- `lib/methods/uniform/uniform-geometric-parameters.ts` and `uniform-geometric-options.ts`: UI defaults and option mapping.
- `lib/methods/uniform/webgpu-uniform-reference.ts`: static-solid fallback and effective dynamic surface policy.
- `lib/methods/uniform/uniform-mixed-frame.ts`: two-stage ordering, overlay bypass, eight-sweep band setup, receipts and acceptance.
- `lib/methods/uniform/uniform-pressure-band.ts`: band membership, open-air rules, prepared rows, coarse boundary fluxes, relaxation and projection.
- `lib/methods/uniform/uniform-mixed-surface.ts`, `uniform-mixed-phi-resolve.ts`, and `uniform-mixed-remap.ts`: surface ownership and reconstruction.
- `tools/probe-uniform-pressure-band-dawn.ts`: explicit band/graded scene probe, including band residual and front/strip measurements. A probe is not itself an acceptance gate.
- `tests/uniform-long-dam-front-dawn.test.ts` and `tools/benchmark-uniform-bulk-surface-dawn.ts`: original scene bounds and full-frame performance checks; verify their effective options before attributing results to a mode.

The older [surface-overlay handoff](uniform-h-surface-overlay-handoff.md) and [pressure-access handoff](uniform-pressure-access-handoff.md) remain experiment history. Their claims about live defaults are superseded by this document.
