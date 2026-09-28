# Plan: take the host out of the Uniform mixed advance loop

Status: proposal, 28 September 2026, written against `8a80095f`. It comes from reading the code only. Nothing has been measured or run.

## Problem

Peter's standing rule (28 September) is that the simulation stays GPU resident: no blocking readbacks on the advance path, and diagnostics only, non-blocking. The mixed Uniform Geometric frame breaks that rule every frame. Nothing waits in the middle of a frame, and no readback carries simulation fields. The problem is the end-of-frame wait, which gates the next frame:

1. **Pressure receipt (every frame).** `uniform-mixed-frame.ts:445-450` copies state, reductions, plan and band receipt into a 120-byte buffer, submits, and awaits `mapAsync`. `webgpu-uniform-reference.ts:2222` (`advanceTo`) returns early while `framePending` is set, so the whole CPU→GPU→CPU round trip sits between frames. The host uses the receipt for three things:
   - **Fail fast (`:452-456`):** a rejected or unconverged solve, band capacity overflow (8,192 tiles), or a failed band solid certificate throws.
   - **Next frame's cycle plan (`:459`, read at `:415`):** `pressurePlan = {vCycles, fullCycles}` decides how many gated slots the host encodes and which kind each one is (`schedule.begin(plan)` writes the control buffer). The slots themselves already launch indirectly on GPU gates, but the slot count is a host feedback loop.
   - **Diagnostics:** residual, cycles, dust totals, band tiles and band residual go into `executionInfo`.
2. **Dynamic census and layout build (every frame when coarsening is Dynamic, the default).**
   - The classifier (`uniform-mixed-dynamic.ts:559`) and the builder (`uniform-mixed-layout-builder.ts:259`) map on the same submission as the receipt, so the wait is shared. The builder maps the full tile topology (n words).
   - `updateMixedDynamic` (`webgpu-uniform-reference.ts:1410`) runs inside the pending frame and awaits both reads. It rebuilds the layout on the CPU (`uniformMixedLayoutFromTiles`, a receipt consistency check), then submits `adoptBuiltLayout`, which does the remap and phi resolve. The host also computes `pressureMatchesSimulation` from the layout words and keeps `mixedDynamicFine` (used by `promoteMixedDrop`).
   - So every relayout frame adds CPU layout work to the critical path.

These are off the advance path and not in scope:
- `readStats` (a diagnostics readback that waits for the frame first);
- `readCoarsestCapture` (QA);
- the readbacks in `webgpu-uniform-reference.ts` around lines 2000, 2160, 2539 and 2580, which come after the mixed branch returns at `:2319` and belong to dense Uniform only;
- the h pressure band, which builds its tile list on the GPU.

`UniformMixedFrame.receipt()` (`:337`) is dead code and should be deleted.

## Goal

The host can encode and submit frame N+1 without waiting for frame N's receipt. Every check still fails loudly, possibly one or two frames late. No simulation behaviour changes, and a relayout needs no CPU layout work before the next frame.

## Constraints to respect

- **Fail fast, no recovery.** A late receipt must still throw and set `simulationPipelineError`. No rollback. The thrown message should name the frame it belongs to. Rejected projections are already withheld on the GPU (the gate enables projection only on acceptance), so a late throw publishes nothing wrong.
- **Empty indirect dispatches are not free.** They cost about 12 µs each on Dawn/Metal, and gating the whole 7-slot superset cost +20 ms. "Encode the maximum schedule and gate on the GPU" is therefore not free: measure it against the round trip it replaces.
- **Layout-general.** Everything must survive dynamic coarsening, with no fast-path hacks.
- **Dense Uniform stays untouched.**
- **Frame time over bit identity.** Ulp-level drift is acceptable.

## Work packages

### WP1: measure first (small)

Instrument one scene: the 128³ long dam, dynamic coarsening. Record the host gap between the frame submission and the next `advanceTo` encode, i.e. map latency plus the `updateMixedDynamic` CPU time. Split it into receipt-only frames and relayout frames. If the gap is small next to GPU frame time and the GPU never idles, stop here and record why. This matches the one-scene, small-matrix rule: one Dawn run, after a CPU shader preflight.

### WP2: deferred receipt (medium)

- Give the receipt a ring of 2–3 map buffers. Each frame copies into its own slot and starts `mapAsync` without awaiting it. `pendingFrame` resolves on submission, not on the map.
- Check each receipt when its map resolves. A failure throws through the same `pressureFrameFailure` / `simulationPipelineError` path, with the frame index attached.
- `advanceTo` refuses to run more than K frames (2 or 3) ahead of the newest checked receipt. That bounds how late a failure can surface and stops unbounded queueing.
- Diagnostics (`executionInfo`) update from the newest resolved receipt.

### WP3: GPU-resident cycle plan (medium, decide after WP1/WP2)

The host must not need `mapped[20..21]` before encoding the next frame. There are two options; measure both on the target scene:
- **(a) Lagged plan.** Encode from the newest *resolved* plan (1–2 frames stale). This is cheap and needs no new GPU work, but the plan lags.
- **(b) GPU-written plan.** The acceptance stage writes the next plan straight into the schedule's control buffer, and the host encodes the maximum slot count gated by that plan. That costs empty-dispatch overhead for the unused slots (see the constraint above). It only pays if the maximum is small or the gate skips whole passes.

Start with (a).

### WP4: host-free relayout (large)

- Adopt the built layout on the GPU in the same or the next submission. `remap.applyBuilt` and the phi resolve take their workgroup counts from the builder's counts buffer through indirect dispatch, instead of from host layout arrays.
- Audit every host consumer of `layout` / `ownership.layout` on the frame path: ownership derivation, dispatch sizes and seam lists in `uniform-mixed-ownership.ts`, `pressureMatchesSimulation`, `mixedDynamicFine`, stage grids, and diagnostics. Each one either becomes GPU-sized, or reads a lagged host mirror updated from a non-blocking readback. The mirror is allowed only where lag is harmless: diagnostics, drop promotion (verify), overlay.
- `pressureMatchesSimulation` currently selects the split versus non-split pressure encode on the host. That needs a GPU gate, or both paths encoded behind gates (again, mind empty-dispatch cost).
- The builder's receipt consistency checks move to the deferred receipt as fail-fast checks.

This is the largest package. It touches the layout builder, remap, ownership and the frame. fluid-d1 has been working in the layout and pressure files; coordinate before starting.

## Acceptance

- With `FLUID_GPU_COMPILATION_CONCURRENCY=1`, one Dawn run on the 128³ long dam (dynamic) shows no host wait between frames, and a lower full frame time than the WP1 baseline.
- `npm run test:dawn:sparse-cm12` is unchanged (canonical gate), and the long-dam front lane still passes. No lane is weakened and no ceiling raised.
- A forced pressure failure still throws within K frames and publishes no projection.

## Order and size

WP1 (hours) → WP2 (about a day) → WP3a (small) → decide on WP3b and WP4 from the measurements. WP4 is multi-day and overlaps fluid-d1's area.

## WP1 result (fluid-a1, 28 September 2026, HEAD 4e1f9ddc)

- **Setup:** one Dawn run on sparse-cm12-long-dam-break (192×96×32) with dynamic coarsening, which relayouts every frame. Medians are taken over 30 untraced frames (steps 30–90, alternate frames traced). Probe: a scratch script (per-frame queue.submit timestamps, advance and updateMixedDynamic wrappers), run from a `git archive` tree.
- **Per frame (ms):**
  - Wall: 20.05.
  - CPU encode up to the main submit: 3.0, with the GPU idle.
  - Main submit → receipt resolved: 15.9.
  - GPU span of the traced stages up to pressure projection: 13.3.
  - `updateMixedDynamic` CPU: 0.33.
  - Relayout submit → frame done: 0.92.
- **GPU busy:** about 13.3 of stages, plus about 1.5–2 of census tail (extension, census, builder), plus the relayout. The GPU idles **about 4 ms per 20 ms frame**. Most of that is the 3 ms CPU encode, which cannot overlap anything while the next advance waits for the receipt. The rest is map latency plus the relayout step.
- **Verdict:** WP2 is worth doing (about 20% of frame time). Separately, a 3 ms CPU encode per frame is itself a lead.
- **Sequencing:** WP2 edits are held until fluid-d1's pressure-path perf round ends, because the receipt, `encodePlanCopy` and `framePending` sit in its blast radius.
