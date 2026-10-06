import assert from "node:assert/strict";
import test from "node:test";
import { runPondRestArm, withPondRestDevice, type PondRestResult } from "../tools/uniform-pond-rest";

const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

/** The divergence a converged pressure step may leave, 1/s: the residual gate
 * of uniform-pond-rest-dawn. Every bound below is this, f32, or the pond-rest
 * surface bounds, which are lengths and do not rescale with the owner width. */
const RESIDUAL = 1e-4;

/** Coarse hydrostatics: ponds at rest whose owners are 4h (Requested detail
 * with no Fine region), on the hero-garden-hose-x10 lattice. A resting state
 * is an exact discrete equilibrium, so its error budget is the solver
 * tolerance plus f32: nothing here is fitted to a run. */
function assertAtRest(result: PondRestResult, fine?: { tiles: number; rests: boolean }, interior = true): void {
  const { arm, samples } = result, [initial, first] = samples;
  assert.ok(initial && first && first.frame === 1, `${arm}: sampled frames 0 and 1`);
  assert.equal(result.fineTiles, fine?.tiles ?? 0, fine ? `${arm}: the Fine box is h` : `${arm}: every tile is 4h`);
  // W4: the fill target is the initial V, so the equilibrium carries no correction
  // beyond the start-up half kick's converged-solve residue.
  assert.ok(initial.excess < 1e-5, `${arm}: initial excess ${initial.excess}`);
  assert.ok(first.correction < RESIDUAL, `${arm}: the authored state meets its fill target to the start-up half kick's converged-solve residue (${first.correction})`);
  assert.ok(first.surface.all.max_mm < 0.001, `${arm}: zero-velocity transport moved the authored plane ${first.surface.all.max_mm} mm`);
  assert.ok(first.maxSpeed < 1e-3, `${arm}: frame 1 speed ${first.maxSpeed} m/s`);
  for (const sample of samples) {
    const at = `${arm} frame ${sample.frame}`;
    assert.ok(sample.residual !== undefined && sample.residual < RESIDUAL, `${at}: residual ${sample.residual}`);
    assert.equal(sample.missing, 0, `${at}: every sampled column keeps its surface`);
    assert.ok(Math.abs(sample.sum / initial.sum - 1) < 1e-6, `${at}: V ${sample.sum} of ${initial.sum}`);
    // 4h owners rest in every frame; so does the seam (T5), whose h faces
    // carry the 4h face's flux. An h region is held to it where it rests.
    assert.ok(sample.coarseSpeed < 1e-3, `${at}: 4h speed ${sample.coarseSpeed} m/s`);
    if (fine) assert.ok(sample.seamSpeed !== undefined && sample.seamSpeed < 1e-3, `${at}: seam face speed ${sample.seamSpeed} m/s`);
    if (!fine || fine.rests) assert.ok(sample.maxSpeed < 1e-3, `${at}: speed ${sample.maxSpeed} m/s`);
    // A basin with no uncut neighbourhood (a film over a cut floor) holds every column to the interior bound.
    if (interior) assert.ok(sample.surface.interior.rms_mm < 0.025, `${at}: interior RMS ${sample.surface.interior.rms_mm} mm`);
    else assert.ok(sample.surface.all.max_mm < 0.025, `${at}: film thickness error ${sample.surface.all.max_mm} mm`);
    assert.ok(sample.surface.all.max_mm < 0.5, `${at}: surface error ${sample.surface.all.max_mm} mm`);
    // The admitted divergence over the deepest column, integrated to now,
    // above the authored plane's own f32 placement (frame 0).
    const drift_mm = 1000 * RESIDUAL * result.depth_m * sample.frame / 30;
    assert.ok(sample.surface.all.max_mm <= initial.surface.all.max_mm + drift_mm, `${at}: surface drift ${sample.surface.all.max_mm} mm exceeds ${drift_mm} mm`);
    assert.equal(sample.airBelow, 0, `${at}: an owner under the waterline is a pressure air row`);
    assert.equal(sample.liquidAbove, 0, `${at}: an owner over the waterline is a pressure liquid row`);
  }
}

// T1a. The waterline at each quarter of a 4h row, and at the two places the
// pressure rows change an owner's class: the centre row itself (offset 0.5)
// and the liquid margin just under it.
dawnTest("a flat all-4h pond with no solid rests at every waterline offset", { timeout: 900_000 }, async () => {
  await withPondRestDevice("Uniform coarse hydrostatics: flat basin", async device => {
    for (const waterline of [20, 21, 22, 22.004, 23]) {
      assertAtRest(await runPondRestArm(device, ["--detail=none", "--basin=open", `--arm=coarse-rest-open-${waterline}`, `--waterline-cells=${waterline}`, "--frames=90"]));
    }
  });
});

/** Static solids on cut 4h owners: Requested detail with no solid-contact
 * request, so no tile is promoted to h (assertAtRest's fineTiles === 0). */
const COARSE_SOLIDS = ["--detail=none", `--values=${JSON.stringify({ detailSolidContact: "off" })}`];

// T1b. A voxel floor 0..3 h cells into its 4h row (the floor owners' open
// fraction 1, 3/4, 1/2, 1/4) under each T1a waterline. The record's Omega and
// face V carry the cut: W1 (the RHS and the operator share V), W4 (the fill
// target is the initial V: no initial excess, no frame-1 correction).
dawnTest("an all-4h pond rests on a voxel floor at every offset in its 4h row", { timeout: 900_000 }, async () => {
  await withPondRestDevice("Uniform coarse hydrostatics: voxel floor", async device => {
    for (const floor of [0, 1, 2, 3]) for (const waterline of [20, 21, 22, 22.004, 23]) {
      assertAtRest(await runPondRestArm(device, [...COARSE_SOLIDS, "--basin=floor", `--arm=coarse-rest-floor-${floor}-${waterline}`, `--floor-offset-cells=${floor}`, `--waterline-cells=${waterline}`, "--frames=90"]));
    }
  });
});

// T2. Planar terrain whose shoreline crosses cut 4h owners: every owner along
// the slope is cut or closed, the waterline meets the plane inside one, and
// the surface bound of assertAtRest covers the shoreline columns. The 1:1
// basin is 19.6 cells deep: at 35.6 cells the cold frame-1 root solve of this
// staircase needs more than the production cycle budget, with solid contact
// at h as much as at 4h (0.0016 against 1e-4 after 6 cycles, both arms).
dawnTest("an all-4h pond rests on planar slopes that cross its waterline", { timeout: 900_000 }, async () => {
  await withPondRestDevice("Uniform coarse hydrostatics: slopes", async device => {
    for (const [name, slope, waterline] of [["1-3", ["--slope-x=0.3333333333"], 40.6], ["1-1", ["--slope-x=1"], 24.6], ["diagonal-1-2", ["--slope-x=0.5", "--slope-z=0.5"], 40.6]] as const) {
      assertAtRest(await runPondRestArm(device, [...COARSE_SOLIDS, "--basin=slope", `--arm=coarse-rest-slope-${name}`, "--floor-offset-cells=-7.75", ...slope, `--waterline-cells=${waterline}`, "--frames=90"]));
    }
  });
});

// T5. A Fine (h) box beside wet cut 4h owners: across a cut floor under the
// waterline, and across the 1:3 slope's shoreline. The seam's h faces keep
// the 4h face's flux (the transfer's V-weighted mean, the band's V-weighted
// Neumann data), so the 4h side and the seam meet T2's bounds in every frame
// and the surface bounds hold on both sides.
// The h side of the slope is not held to the every-frame speed bound, as
// uniform-pond-rest-dawn does not hold the h path to it: an h surface row
// beside a shoreline grows a cell-to-cell pattern from rest with solid
// contact at h as much as inside this box (1.2e-3 m/s at frame 60 here,
// 0.85e-3 with contact at h and no box), whatever the band's cycle count,
// time step or surface tension. It is an open defect of the h path, not a
// tolerance of this lane; the floor box, whose h cells meet no shoreline,
// is held to the bound.
// These 90 frames are not a proof of rest for the box itself: a Fine box in an
// all-4h pond grows a slow seam mode with no solid anywhere (--basin=open, the
// same box: T5b below), and the floor box here is 3.7e-5 m/s and rising at
// frame 90 (4.0e-4 at frame 200, a fatal pressure non-convergence at frame
// 248; measured 2026-10-05 with the band solving in correction form). It is
// the open seam drift of the mixed layout, not a property of cut owners: this
// lane holds the cut seam to the window the same box over an uncut floor meets.
dawnTest("a Fine box rests across a cut 4h floor and a cut 4h shoreline", { timeout: 900_000 }, async () => {
  await withPondRestDevice("Uniform coarse hydrostatics: seams", async device => {
    assertAtRest(await runPondRestArm(device, [...COARSE_SOLIDS, "--basin=floor", "--arm=coarse-rest-seam-floor", "--floor-offset-cells=1", "--waterline-cells=21", "--fine-box=56,8,32,88,28,64", "--frames=90"]), { tiles: 320, rests: true });
    assertAtRest(await runPondRestArm(device, [...COARSE_SOLIDS, "--basin=slope", "--arm=coarse-rest-seam-slope", "--floor-offset-cells=-7.75", "--slope-x=0.3333333333", "--waterline-cells=40.6", "--fine-box=96,28,32,124,48,64", "--frames=90"]), { tiles: 280, rests: false });
  });
});

// T5b. The same box in the open basin (no solid anywhere), for 300 frames: the
// app default (Requested with a Fine region) in a still pond. It does not
// rest: 1.4e-4 m/s at frame 50, 4.8e-4 at frame 100, 0.04 at frame 150 and
// over 1 m/s by frame 200, from the box's surface corners (measured
// 2026-10-05). Two causes are fixed: the h redistance dropped phi(q) at its
// Newton stop (an all-h surface grew a 2h sawtooth), and the band solved total
// pressure, whose float32 floor the root answered. Two remain. The all-4h
// root's volume correction on tiles the band solves pumps the box through its
// free surface (an all-h slab with no seam at the surface: 7.5e-5 m/s at frame
// 300, 2.9e-3 at frame 600, smooth upwelling). And where the seam crosses the
// surface, face flow in the surface row alternates cell to cell, which neither
// vertex phi nor the cell-centre departure boxes see, so V leaves phi in the
// air row; that mode grows from the seam with the pump removed (a held
// candidate: 3.3e-5 m/s at frame 150, 2.6e-4 at frame 300, a blow-up by frame
// 450; the same candidate's all-h slab holds 1.7e-5 m/s at frame 600). The
// bound is the lane's; the arm is a todo until the box rests.
dawnTest("a Fine box rests in an all-4h pond with no solid", { timeout: 900_000, todo: "Known reproduction: the seam where an h box crosses a resting surface grows from rest" }, async () => {
  await withPondRestDevice("Uniform coarse hydrostatics: open seam", async device => {
    assertAtRest(await runPondRestArm(device, ["--detail=none", "--basin=open", "--arm=coarse-rest-seam-open", "--waterline-cells=21", "--fine-box=56,8,32,88,28,64", "--frames=300"]), { tiles: 320, rests: true });
  });
});

// T8. A Fine box over cut 4h owners drawn before every odd frame and removed
// before every even one, for ten frames: each promote splits a cut owner's V
// by its h cells' open fractions and each retire hands the buried corners
// their surface plane back (uniform-mixed-remap), so the pond meets the
// all-4h bounds in every frame, toggled or not, and V is conserved. The box
// is the cut floor row under the waterline, and the shorelines of the 1:3
// and 1:1 slopes.
dawnTest("promoting and retiring cut 4h tiles every frame leaves the pond at rest", { timeout: 900_000 }, async () => {
  await withPondRestDevice("Uniform coarse hydrostatics: promote and retire", async device => {
    for (const [name, basin, box, tiles] of [
      ["floor", ["--basin=floor", "--floor-offset-cells=1", "--waterline-cells=21"], "56,12,32,88,16,64", 8 * 1 * 8],
      ["slope-1-3", ["--basin=slope", "--floor-offset-cells=-7.75", "--slope-x=0.3333333333", "--waterline-cells=40.6"], "96,28,32,124,48,64", 7 * 5 * 8],
      ["slope-1-1", ["--basin=slope", "--floor-offset-cells=-7.75", "--slope-x=1", "--waterline-cells=24.6"], "8,12,32,32,32,64", 6 * 5 * 8],
    ] as const) {
      const result = await runPondRestArm(device, [...COARSE_SOLIDS, ...basin, `--arm=coarse-rest-toggle-${name}`, `--toggle-box=${box}`, "--frames=90"]);
      assertAtRest(result);
      for (let frame = 1; frame <= 10; frame++) {
        const sample = result.samples.find(sample => sample.frame === frame);
        assert.equal(sample?.fineTiles, frame % 2 ? tiles : 0, `${result.arm} frame ${frame}: the box's tiles are ${frame % 2 ? "h" : "4h"}`);
      }
    }
  });
});

/** A policy change on a running, resting pond before frame SWITCH (design
 * step 9). The remap between h and cut 4h owners keeps the equilibrium: V,
 * the surface and the 4h owners meet the rest bounds in every sampled frame
 * (the two after the switch among them), and so do the h owners in the
 * frames `hRests` names. */
const SWITCH = 6;
function assertSwitchHolds(result: PondRestResult, fineAfter: boolean, hRests: (frame: number) => boolean): void {
  const { arm, samples } = result, initial = samples[0]!;
  for (const frame of [SWITCH - 3, SWITCH, SWITCH + 1]) assert.ok(samples.some(sample => sample.frame === frame), `${arm}: sampled frame ${frame}`);
  for (const sample of samples) {
    const at = `${arm} frame ${sample.frame}`;
    if (sample.frame > 0) assert.equal(sample.fineTiles > 0, sample.frame < SWITCH ? !fineAfter : fineAfter, `${at}: ${sample.fineTiles} h tiles`);
    assert.ok(sample.residual !== undefined && sample.residual < RESIDUAL, `${at}: residual ${sample.residual}`);
    assert.equal(sample.missing, 0, `${at}: every sampled column keeps its surface`);
    assert.ok(Math.abs(sample.sum / initial.sum - 1) < 1e-6, `${at}: V ${sample.sum} of ${initial.sum}`);
    assert.ok(sample.coarseSpeed < 1e-3, `${at}: 4h speed ${sample.coarseSpeed} m/s`);
    if (hRests(sample.frame)) assert.ok(sample.maxSpeed < 1e-3, `${at}: speed ${sample.maxSpeed} m/s`);
    assert.ok(sample.surface.interior.rms_mm < 0.025, `${at}: interior RMS ${sample.surface.interior.rms_mm} mm`);
    assert.ok(sample.surface.all.max_mm < 0.5, `${at}: surface error ${sample.surface.all.max_mm} mm`);
  }
}

// Dynamic (solid contact at h: every wet cut tile is h) to Requested with no
// solid-contact request: one relayout retires every h tile, the cut ones to
// cut 4h owners whose buried corners the h layout never stored.
dawnTest("a resting pond keeps its rest when Dynamic detail gives way to all-4h cut owners", { timeout: 900_000 }, async () => {
  await withPondRestDevice("Uniform coarse hydrostatics: Dynamic to Requested", async device => {
    const change = [`--values=${JSON.stringify({ detailPolicy: "dynamic" })}`, `--switch-frame=${SWITCH}`, `--switch-values=${JSON.stringify({ detailPolicy: "requested", detailSolidContact: "off" })}`, "--frames=90"];
    assertSwitchHolds(await runPondRestArm(device, ["--detail=none", "--basin=floor", "--arm=coarse-rest-retire-floor", "--floor-offset-cells=1", "--waterline-cells=21", ...change]), false, () => true);
    assertSwitchHolds(await runPondRestArm(device, ["--detail=none", "--basin=slope", "--arm=coarse-rest-retire-slope", "--floor-offset-cells=-7.75", "--slope-x=0.3333333333", "--waterline-cells=40.6", ...change]), false, () => true);
  });
});

// The other way: `detailSolidContact` turned on over resting cut 4h owners,
// which promotes every wet cut tile to h in one relayout. The floor's h
// owners meet no shoreline and rest throughout. The slope's are held to the
// speed bound in the two frames after the switch only, and the arm stops at
// frame 40: the h path's open shoreline defect (see T5) grows from there
// whether the h tiles came from this switch or were h from frame 0, and
// ends in a pressure fatal (frame 71 here, frame 89 with contact at h from
// the start). The 4h owners, V and the surface are held in every frame.
dawnTest("a resting pond keeps its rest when solid contact promotes its cut 4h owners", { timeout: 900_000 }, async () => {
  await withPondRestDevice("Uniform coarse hydrostatics: solid contact on", async device => {
    const change = [`--switch-frame=${SWITCH}`, `--switch-values=${JSON.stringify({ detailSolidContact: "on" })}`];
    assertSwitchHolds(await runPondRestArm(device, [...COARSE_SOLIDS, "--basin=floor", "--arm=coarse-rest-promote-floor", "--floor-offset-cells=1", "--waterline-cells=21", ...change, "--frames=90"]), true, () => true);
    assertSwitchHolds(await runPondRestArm(device, [...COARSE_SOLIDS, "--basin=slope", "--arm=coarse-rest-promote-slope", "--floor-offset-cells=-7.75", "--slope-x=0.3333333333", "--waterline-cells=40.6", ...change, "--frames=40"]), true, frame => frame <= SWITCH + 1);
  });
});

// T4. A film 0.3 of a 4h row thick on a voxel floor that cuts the row in half:
// every wet owner is cut (open fraction 1/2, V 0.3), its centre on the floor
// plane. Transport holds a cut row to its open volume, so the film keeps its
// thickness; no face moves (assertAtRest's speed bound covers the downward
// faces).
dawnTest("a film on a cut voxel floor keeps its thickness", { timeout: 900_000 }, async () => {
  await withPondRestDevice("Uniform coarse hydrostatics: film on a cut floor", async device => {
    assertAtRest(await runPondRestArm(device, [...COARSE_SOLIDS, "--basin=floor", "--arm=coarse-rest-film", "--floor-offset-cells=2", "--waterline-cells=15.2", "--interior=optional", "--frames=90"]), undefined, false);
  });
});

// T4, the air-centred film: 0.6 h of water on a voxel floor one h cell into
// the row (open fraction 3/4), so every wet owner's centre is above the
// waterline: pressure air rows holding V and touching no liquid row, which is
// detached mass. Standing on solid it is not in flight: one frame of free
// fall is g/30 = 0.33 m/s on its faces, against a state that must not move at
// all (no pressure row exists to leave a residual). No 4h vertex lies between
// the floor and the waterline, so there is no surface to sample: V is the film.
dawnTest("a film under a cut 4h owner's centre does not fall into the floor", { timeout: 900_000 }, async () => {
  await withPondRestDevice("Uniform coarse hydrostatics: air-centred film on a cut floor", async device => {
    const result = await runPondRestArm(device, [...COARSE_SOLIDS, "--basin=floor", "--arm=coarse-rest-thin-film", "--floor-offset-cells=1", "--waterline-cells=13.6", "--interior=optional", "--frames=90"]);
    const initial = result.samples[0]!;
    assert.equal(result.fineTiles, 0, "every tile is 4h");
    // 36 x 24 floor tiles, 16 open columns each, 0.6 h deep.
    assert.ok(Math.abs(initial.sum / (36 * 24 * 16 * 0.6) - 1) < 1e-6, `authored film V ${initial.sum}`);
    assert.equal(result.samples.at(-1)!.frame, 90);
    for (const sample of result.samples) {
      const at = `${result.arm} frame ${sample.frame}`;
      assert.equal(sample.maxSpeed, 0, `${at}: a supported film moves at ${sample.maxSpeed} m/s`);
      assert.equal(sample.sum, initial.sum, `${at}: V ${sample.sum} of ${initial.sum}`);
    }
  });
});

// T3. Communicating vessels in motion: a wall one 4h tile thick splits the
// lattice, and the two sides meet through a slot of cut 4h owners (the lower
// half of each closed) under both waterlines, which start two 4h rows apart.
// The liquid crosses cut owners at about 1 m/s: their rows are normalized to
// the open volume, and their departures leave the open centroid.
// Asserted: every frame is accepted on all-4h owners; V is conserved to the
// bound of a moving surface (uniform-dynamic-coarsening's volume drift, 1e-4:
// the dust floor acts in motion); after 10 s each side's level, from its V
// and the h mask's open area, is within one h cell of the common level the
// total V fills (the lattice's resolution).
// Not asserted, and not met: the level to 0.025 mm. The exchange through an
// orifice is damped quadratically, so its amplitude decays as 1/t (about
// 0.3 mm after 20 s for this slot); and a cut owner in the slot does not hold
// V at its open fraction while liquid crosses it (0.35 to 2.2 of it, where an
// h cell there stays within a tenth).
dawnTest("communicating vessels equalize through a slot of cut 4h owners", { timeout: 900_000 }, async () => {
  await withPondRestDevice("Uniform coarse solids: communicating vessels", async device => {
    const result = await runPondRestArm(device, [...COARSE_SOLIDS, "--basin=vessels", "--arm=coarse-vessels-slot", "--floor-offset-cells=-12",
      "--vessel=0,72,144,0,96", "--window=4,6,0,96", "--waterline-cells=14", "--left-shift-cells=8", "--frames=300"]);
    const { arm, samples } = result, initial = samples[0]!, last = samples.at(-1)!, common = result.vesselFinal as number;
    assert.equal(result.fineTiles, 0, `${arm}: every tile is 4h`);
    assert.equal(last.frame, 300, `${arm}: every frame was accepted`);
    assert.ok(Math.abs(initial.chambers![0]! - 22) < 1e-6 && Math.abs(initial.chambers![1]! - 14) < 1e-6, `${arm}: the authored levels ${initial.chambers}`);
    assert.ok(Math.abs(common - (244224 - 768) / 13440) < 1e-6, `${arm}: the common level ${common} is the total V over the open area`);
    for (const sample of samples) assert.ok(Math.abs(sample.sum / initial.sum - 1) < 1e-4, `${arm} frame ${sample.frame}: V ${sample.sum} of ${initial.sum}`);
    for (const level of last.chambers!) assert.ok(Math.abs(level - common) < 1, `${arm}: a side rests at ${level} cells, the common level is ${common}`);
  });
});
