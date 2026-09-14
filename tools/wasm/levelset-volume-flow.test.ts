import assert from "node:assert/strict";
import test from "node:test";
import { createTallCellsHillsideDamBreakScene, findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument } from "../../lib/core/scene-definition";
import { decodePhysicsPublication } from "../../lib/physics-wasm/publication";
import { advanceRowX, advanceRowY, createAdvanceView, type AdvanceGraph,
  type AdvanceView } from "../../lib/physics-wasm/advance-view";
import { parsePhysicsReceipt } from "../../lib/physics-wasm/protocol";
import { loadFluidWasmForNode } from "./load-module.mjs";

function assertRedistancingReceipt(receipt: Readonly<Record<string, unknown>>): void {
  const value = receipt.levelSetVolume;
  assert.ok(value && typeof value === "object", "the selected transport returns its own receipt");
  const levelSetVolume = value as Readonly<Record<string, unknown>>;
  assert.ok(Number(levelSetVolume.redistancedSamples) > 0,
    "redistancing evaluates the accepted contour at cell centres for derived geometry");
  assert.ok(Number(levelSetVolume.redistanceSegmentCount) > 0,
    "redistancing indexes the accepted shared-interface contour");
  assert.ok(Number.isInteger(levelSetVolume.redistanceFallbackSamples)
    && Number(levelSetVolume.redistanceFallbackSamples) >= 0,
  "redistancing publishes its fallback-sample count");
  assert.equal(Number(levelSetVolume.redistanceNanoseconds), 0,
    "Wasm receipts expose the stage while leaving host timing disabled");
  assertSharpeningReceipt(levelSetVolume);
}

function assertSharpeningReceipt(levelSetVolume: Readonly<Record<string, unknown>>): void {
  const value = levelSetVolume.sharpening;
  assert.ok(value && typeof value === "object", "level-set volume publishes sharpening diagnostics");
  const sharpening = value as Readonly<Record<string, unknown>>;
  const number = (key: string) => {
    const result = Number(sharpening[key]);
    assert.ok(Number.isFinite(result), `sharpening.${key} is finite`);
    return result;
  };
  const tolerance = 1e-10 * Math.max(1, number("initialBandAbsoluteMismatch"));
  assert.ok(number("finalBandAbsoluteMismatch") <= number("initialBandAbsoluteMismatch") + tolerance,
    "sharpening does not increase mismatch in the eligible interface band");
  assert.ok(number("finalDistanceWeightedMismatch")
    <= number("initialDistanceWeightedMismatch") + tolerance,
  "sharpening moves conservative volume toward the immutable phi surface");
  assert.ok(number("finalOverCapacityVolume") <= number("initialOverCapacityVolume") + tolerance,
    "sharpening never creates additional over-capacity volume");
  assert.equal(number("crossComponentPairCount"), 0,
    "sharpening never transfers volume between phi components");
  assert.equal(number("boundViolationCount"), 0,
    "sharpening preserves donor and receiver bounds");
  assert.ok(Math.abs(number("globalConservationResidual")) <= 1e-10,
    "sharpening conserves global volume before the f32 field commit");
  assert.ok(number("maximumComponentConservationResidual") <= 1e-10,
    "sharpening conserves every labelled phi component");
}

function assertFlatHydrostaticSurface(view: ReturnType<typeof createAdvanceView>): void {
  const expectedY = 15.25;
  assert.ok(view.rdf.segmentsFine.length > 0, "hydrostatic level set publishes its free surface");
  for (let at = 0; at + 3 < view.rdf.segmentsFine.length; at += 4) {
    assert.ok(Math.abs(view.rdf.segmentsFine[at + 1]! - expectedY) <= 2e-4
      && Math.abs(view.rdf.segmentsFine[at + 3]! - expectedY) <= 2e-4,
    `hydrostatic segment ${at / 4} moved from y=${expectedY}`);
  }
}

function maximumHydrostaticLiquidFaceSpeed(view: AdvanceView): number {
  const expectedY = 15.25;
  let maximum = 0;
  for (const row of view.graph.rows) {
    if (!row.terms.some(term => view.graph.cells[term.cellId]!.center[1]! <= expectedY)) continue;
    let value = 0;
    if (row.axis === 0) {
      const x = Math.round(row.center[0]!);
      const y = view.ny - 1 - Math.floor(row.center[1]!);
      if (x >= 0 && x <= view.nx && y >= 0 && y < view.ny)
        value = view.faceVelocityXFine[advanceRowX(view, x, y)] ?? 0;
    } else {
      const x = Math.floor(row.center[0]!);
      const y = view.ny - Math.round(row.center[1]!);
      if (x >= 0 && x < view.nx && y >= 0 && y <= view.ny)
        value = view.faceVelocityYFine[advanceRowY(view, x, y)] ?? 0;
    }
    maximum = Math.max(maximum, Math.abs(value));
  }
  return maximum;
}

for (const artifact of ["scalar", "simd"] as const) {
  test(`${artifact} Wasm advances and publishes level-set-plus-volume through a resolution edit`, async () => {
    const wasm = await loadFluidWasmForNode(undefined, { artifact });
    const definition = findSceneDefinition("water-box-dam-break");
    assert.ok(definition);
    const scene = { id: definition.id, label: definition.name, document: sceneDocument(definition) };
    const world = wasm.FluidWorld.from_scene(JSON.stringify(scene.document), JSON.stringify({
      runEpoch: 1, commandSequence: 0, pressureIterations: 64, tracerBudget: 0,
      transportExperiment: "level-set-volume",
      production: { dtS: 1 / 30, timeStep: "paper" },
    }));
    try {
      const initial = parsePhysicsReceipt(world.receipt());
      const initialVolume = Number(initial.liquidMeasure);
      assert.ok(initialVolume > 0);
      let sequence = 0;
      for (let frame = 1; frame <= 3; frame++) {
        if (frame === 2) {
          const edit = parsePhysicsReceipt(world.apply_command(JSON.stringify({
            type: "set-refinement-regions", commandSequence: ++sequence, runEpoch: 1,
            regions: [{ minimumFine: [0, 0], maximumFine: [8, 8],
              minimumCellWidth: 2, maximumCellWidth: 2 }],
          })));
          assert.equal(edit.commandSequence, sequence);
        }
        const receipt = parsePhysicsReceipt(world.advance(++sequence, 1 / 30));
        assert.equal(receipt.frame, frame);
        assert.equal(receipt.microsteps, 0);
        assertRedistancingReceipt(receipt);
        assert.equal(receipt.cellwiseRemap, undefined);
        assert.ok(Math.abs(Number(receipt.liquidMeasure) - initialVolume) / initialVolume < 1e-6,
          "topology changes and publication retain conservative material");
        const bytes = world.snapshot(0xf).slice();
        const decoded = decodePhysicsPublication({ id: frame, revision: receipt, bytes, release() {} });
        try {
          const view = createAdvanceView(decoded, undefined, scene);
          assert.equal(view.revision.frame, frame);
          assert.ok(view.rdf.vertexPhiFine.some(Number.isFinite));
          assert.ok(view.rdf.segmentsFine.length > 0, "the level set publishes a visible surface");
        } finally { decoded.release(); }
      }
    } finally { world.free(); }
  });
}

for (const [sceneId, frames] of [["cm12-figure-7", 30], ["coarse-first-pool-impact-half", 10]] as const) {
  test(`SIMD level-set UI defaults publish ${frames} frames of ${sceneId}`, async () => {
    const wasm = await loadFluidWasmForNode(undefined, { artifact: "simd" });
    const definition = findSceneDefinition(sceneId);
    assert.ok(definition);
    const scene = { id: sceneId, label: definition.name, document: sceneDocument(definition) };
    const world = wasm.FluidWorld.from_scene(JSON.stringify(scene.document), JSON.stringify({
      runEpoch: 1, commandSequence: 0, pressureIterations: 256,
      pressureRelativeTolerance: 1e-6, transportExperiment: "level-set-volume",
      production: { dtS: 1 / 30, timeStep: "paper" },
    }));
    try {
      const initialVolume = Number(parsePhysicsReceipt(world.receipt()).liquidMeasure);
      let graph: AdvanceGraph | undefined;
      const observedBrickKeys = new Set<number>();
      let coarseFirstEmptyAllocations = 0;
      let unconstrainedEmptyAllocations = 0;
      for (let frame = 1; frame <= frames; frame++) {
        const receipt = parsePhysicsReceipt(world.advance(frame, 1 / 30));
        assert.equal(receipt.frame, frame);
        assert.equal(receipt.fault, null);
        assert.equal(receipt.microsteps, 0);
        assertRedistancingReceipt(receipt);
        assert.ok(Math.abs(Number(receipt.liquidMeasure) - initialVolume) / initialVolume < 1e-6);
        const decoded = decodePhysicsPublication({ id: frame, revision: receipt,
          bytes: world.snapshot(0xf).slice(), release() {} });
        try {
          const view = createAdvanceView(decoded, graph, scene);
          graph = view.graph;
          if (sceneId === "cm12-figure-7" && frame === 6) {
            assert.ok(view.graph.cells.some(cell => cell.widths[0] === 4),
              "Figure 7 retains width-4 interior cells under bulk falling motion");
          }
          const vacatedTrailingKeys = sceneId === "cm12-figure-7"
            ? frame === 16 ? [2246, 2249] : (frame === 17 || frame === 30) ? [2229, 2234] : []
            : [];
          for (const key of vacatedTrailingKeys) {
            const trailing = view.graph.bricks.find(brick => brick.key === key);
            assert.ok(!trailing || !trailing.active || trailing.resolution <= 1,
              `frame ${frame}: vacated trailing brick ${key} is coarsest or retired`);
          }
          const resolution = decoded.metadata.resolution as {
            bricks?: Array<{ brickKey: number; acceptedResolution: number;
              reasons: number; requestedResolution: number; planReasons: number }>;
          } | undefined;
          const policyByKey = new Map((resolution?.bricks ?? []).map(brick => [brick.brickKey, brick]));
          if (sceneId === "cm12-figure-7") {
            const newlyAllocatedEmptySupport = view.graph.bricks.filter(brick => {
              const policy = policyByKey.get(brick.key);
              const volume = view.lattice.cells.filter(cell => cell.brick === brick.key)
                .reduce((sum, cell) => sum + cell.volume, 0);
              return !observedBrickKeys.has(brick.key) && brick.active && volume === 0
                && policy?.planReasons === 0x8000_0001;
            });
            coarseFirstEmptyAllocations += newlyAllocatedEmptySupport.length;
            assert.ok(newlyAllocatedEmptySupport.every(brick =>
              policyByKey.get(brick.key)?.acceptedResolution === 1),
            `frame ${frame}: newly allocated empty support must start at the coarsest rung`);
            const unconstrainedEmptySupport = newlyAllocatedEmptySupport
              .filter(brick => policyByKey.get(brick.key)?.requestedResolution === 1);
            unconstrainedEmptyAllocations += unconstrainedEmptySupport.length;
            assert.ok(unconstrainedEmptySupport.every(brick => brick.resolution < 8),
            `frame ${frame}: unconstrained empty support must not become finest`);
          }
          for (const brick of view.graph.bricks) observedBrickKeys.add(brick.key);
          if (sceneId === "cm12-figure-7" && frame === 23) {
            const segments = view.rdf.segmentsFine;
            const directions: Array<readonly [number, number]> = [];
            for (let at = 0; at + 3 < segments.length; at += 4) {
              const dx = segments[at + 2]! - segments[at]!;
              const dy = segments[at + 3]! - segments[at + 1]!;
              const length = Math.hypot(dx, dy);
              if (length > 1e-6) directions.push([dx / length, dy / length]);
            }
            assert.ok(directions.some((a, index) => directions.slice(index + 1)
              .some(b => Math.abs(a[0] * b[1] - a[1] * b[0]) > 0.25)),
              "Figure 7 frame 23 has a curved shared zero contour");
            const curvedFine = view.graph.bricks.some(brick => {
              const curvatureFloor = Number(policyByKey.get(brick.key)?.reasons ?? 0) >>> 16;
              if (!brick.active || curvatureFloor <= 1 || brick.resolution < curvatureFloor) return false;
              const x0 = brick.coordinate[0]! * 8, y0 = brick.coordinate[1]! * 8;
              const x1 = x0 + brick.spanBricks * 8, y1 = y0 + brick.spanBricks * 8;
              for (let at = 0; at + 3 < segments.length; at += 4) {
                const x = 0.5 * (segments[at]! + segments[at + 2]!);
                const y = 0.5 * (segments[at + 1]! + segments[at + 3]!);
                if (x >= x0 && x <= x1 && y >= y0 && y <= y1) return true;
              }
              return false;
            });
            assert.ok(curvedFine,
              "a curved shared-contour brick satisfies its published curvature floor");
          }
          assert.ok(view.rdf.vertexPhiFine.some(Number.isFinite));
          assert.ok(view.rdf.segmentsFine.length > 0, `frame ${frame}: visible RDF surface`);
        } finally { decoded.release(); }
      }
      if (sceneId === "cm12-figure-7") assert.ok(coarseFirstEmptyAllocations > 0,
        "Figure 7 must exercise newly allocated empty support to cover coarse-first allocation");
      if (sceneId === "cm12-figure-7") assert.ok(unconstrainedEmptyAllocations > 0,
        "Figure 7 must exercise unconstrained empty support to guard against finest defaults");
    } finally { world.free(); }
  });
}

test("SIMD hillside level-set volume crosses the frame-29 topology transition without a velocity burst", async () => {
  const wasm = await loadFluidWasmForNode(undefined, { artifact: "simd" });
  const document = createTallCellsHillsideDamBreakScene();
  const scene = { id: document.sceneId, label: "Tall Cells hillside dam break", document };
  const dt = 1 / 30;
  const world = wasm.FluidWorld.from_scene(JSON.stringify(document), JSON.stringify({
    runEpoch: 1, commandSequence: 0, pressureIterations: 256,
    pressureRelativeTolerance: 1e-6, tracerBudget: 0,
    transportExperiment: "level-set-volume",
    production: { dtS: dt, timeStep: "paper" },
  }));
  try {
    const initialVolume = Number(parsePhysicsReceipt(world.receipt()).liquidMeasure);
    let graph: AdvanceGraph | undefined;
    let earlyMaximumVelocity = 0;
    let lateMaximumVelocity = 0;
    let sawPhiSeamComparison = false;
    for (let frame = 1; frame <= 40; frame++) {
      const receipt = parsePhysicsReceipt(world.advance(frame, dt));
      assert.equal(receipt.fault, null);
      assert.equal(receipt.microsteps, 0);
      assertRedistancingReceipt(receipt);
      assert.ok(Math.abs(Number(receipt.liquidMeasure) - initialVolume) / initialVolume < 1e-6,
        `frame ${frame}: topology changes retain the hillside's conservative material`);
      const levelSetVolume = receipt.levelSetVolume as Readonly<Record<string, unknown>>;
      for (const key of ["phiImpliedLiquidVolume", "signedPhiVolumeMismatch",
        "absolutePhiVolumeMismatch", "insideBandAbsolutePhiVolumeMismatch",
        "outsideBandAbsolutePhiVolumeMismatch", "maximumAbsolutePhiVolumeMismatch",
        "maximumNormalizedPhiVolumeMismatch"] as const) {
        assert.ok(Number.isFinite(Number(levelSetVolume[key])), `frame ${frame}: ${key}`);
      }
      assert.ok(Number(levelSetVolume.absolutePhiVolumeMismatch) >= 0);
      assert.ok(Number(levelSetVolume.maximumAbsolutePhiVolumeMismatch) >= 0);
      const seams = receipt.interfaceSeams as Readonly<Record<string, unknown>>;
      sawPhiSeamComparison ||= Number(seams.comparisonCount) > 0;
      const courant = Number(levelSetVolume.maximumTraceCourant);
      assert.ok(Number.isFinite(courant) && courant <= 5,
        `frame ${frame}: whole-step characteristic crossed ${courant} finest cells`);

      const decoded = decodePhysicsPublication({ id: frame, revision: receipt,
        bytes: world.snapshot(0xf).slice(), release() {} });
      try {
        const view = createAdvanceView(decoded, graph, scene);
        graph = view.graph;
        assert.ok(view.liquidVolumeFine.every(Number.isFinite));
        assert.ok(view.capacityFine.every(Number.isFinite));
        assert.ok(view.faceVelocityXFine.every(Number.isFinite));
        assert.ok(view.faceVelocityYFine.every(Number.isFinite));
        assert.ok(view.rdf.vertexPhiFine.every(Number.isFinite));
        assert.ok(view.rdf.segmentsFine.length > 0, `frame ${frame}: visible RDF surface`);
        const maximumVelocity = Number(receipt.maxVelocity);
        assert.ok(Number.isFinite(maximumVelocity));
        if (frame <= 20) earlyMaximumVelocity = Math.max(earlyMaximumVelocity, maximumVelocity);
        else lateMaximumVelocity = Math.max(lateMaximumVelocity, maximumVelocity);
      } finally { decoded.release(); }
    }
    assert.ok(earlyMaximumVelocity > 0);
    assert.ok(sawPhiSeamComparison,
      "the mixed-rung hillside publishes a phi-based seam comparison");
    assert.ok(lateMaximumVelocity <= 1.25 * earlyMaximumVelocity,
      `late topology transitions caused a velocity burst: early=${earlyMaximumVelocity}, late=${lateMaximumVelocity}`);
  } finally { world.free(); }
});

test("SIMD level-set hydrostatic offset stays flat through asymmetric refinement", async () => {
  const wasm = await loadFluidWasmForNode(undefined, { artifact: "simd" });
  const definition = findSceneDefinition("hydrostatic-power-large-offset");
  assert.ok(definition);
  const scene = { id: definition.id, label: definition.name, document: sceneDocument(definition) };
  const world = wasm.FluidWorld.from_scene(JSON.stringify(scene.document), JSON.stringify({
    runEpoch: 1, commandSequence: 0, pressureIterations: 256,
    pressureRelativeTolerance: 1e-6, transportExperiment: "level-set-volume",
    production: { dtS: 1 / 30, timeStep: "paper" },
  }));
  try {
    const initial = parsePhysicsReceipt(world.receipt());
    const initialVolume = Number(initial.liquidMeasure);
    let graph: AdvanceGraph | undefined;
    let sequence = 0;
    let sawMixedRungs = false;
    for (let frame = 0; frame <= 30; frame++) {
      if (frame === 4) {
        parsePhysicsReceipt(world.apply_command(JSON.stringify({
          type: "set-refinement-regions", commandSequence: ++sequence, runEpoch: 1,
          regions: [{ minimumFine: [0, 0], maximumFine: [16, 24],
            minimumCellWidth: 1, maximumCellWidth: 1 }],
        })));
      }
      const receipt = frame === 0 ? initial
        : parsePhysicsReceipt(world.advance(++sequence, 1 / 30));
      assert.equal(receipt.fault, null);
      assert.ok(Math.abs(Number(receipt.liquidMeasure) - initialVolume) <= 2e-5,
        `frame ${frame}: hydrostatic volume changed`);
      const decoded = decodePhysicsPublication({ id: frame, revision: receipt,
        bytes: world.snapshot(0xf).slice(), release() {} });
      try {
        const view = createAdvanceView(decoded, graph, scene);
        graph = view.graph;
        const liquidSpeed = maximumHydrostaticLiquidFaceSpeed(view);
        assert.ok(liquidSpeed <= 1e-3,
          `frame ${frame}: liquid-touching hydrostatic speed ${liquidSpeed}`);
        assertFlatHydrostaticSurface(view);
        const activeRungs = new Set(view.graph.bricks
          .filter(brick => brick.active).map(brick => brick.resolution));
        sawMixedRungs ||= activeRungs.size > 1;
      } finally { decoded.release(); }
    }
    assert.ok(sawMixedRungs, "the asymmetric edit must exercise a mixed-rung topology");
  } finally { world.free(); }
});

test("SIMD level-set answers the lab's drop gesture through the command boundary", async () => {
  const wasm = await loadFluidWasmForNode(undefined, { artifact: "simd" });
  const definition = findSceneDefinition("water-box-dam-break");
  assert.ok(definition);
  const scene = { id: definition.id, label: definition.name, document: sceneDocument(definition) };
  const world = wasm.FluidWorld.from_scene(JSON.stringify(scene.document), JSON.stringify({
    runEpoch: 1, commandSequence: 0, pressureIterations: 256,
    pressureRelativeTolerance: 1e-6, transportExperiment: "level-set-volume",
    production: { dtS: 1 / 30, timeStep: "paper" },
  }));
  try {
    let sequence = 0;
    let graph: AdvanceGraph | undefined;
    const publish = (receipt: ReturnType<typeof parsePhysicsReceipt>): AdvanceView => {
      const decoded = decodePhysicsPublication({ id: sequence, revision: receipt,
        bytes: world.snapshot(0xf).slice(), release() {} });
      try {
        const view = createAdvanceView(decoded, graph, scene);
        graph = view.graph;
        return view;
      } finally { decoded.release(); }
    };
    /** Whether the published zero contour has a segment inside this box. */
    const contourWithin = (view: AdvanceView, centre: readonly [number, number],
      radius: number): boolean => {
      for (let at = 0; at + 3 < view.rdf.segmentsFine.length; at += 4) {
        const x = 0.5 * (view.rdf.segmentsFine[at]! + view.rdf.segmentsFine[at + 2]!);
        const y = 0.5 * (view.rdf.segmentsFine[at + 1]! + view.rdf.segmentsFine[at + 3]!);
        if (Math.hypot(x - centre[0], y - centre[1]) <= radius) return true;
      }
      return false;
    };

    const before = publish(parsePhysicsReceipt(world.receipt()));
    /* Dry air above the dam, where only the drop can put an interface. */
    const centre = [before.nx * 0.75, before.ny * 0.8] as const;
    const radius = Math.max(2, Math.round(before.ny / 12));
    assert.ok(!contourWithin(before, centre, 2 * radius),
      "the authored scene already has a surface where the ball is aimed");
    const initialVolume = Number(before.receipt.liquidMeasure);

    const dropped = parsePhysicsReceipt(world.apply_command(JSON.stringify({
      type: "inject-liquid", commandSequence: ++sequence, runEpoch: 1,
      drop: { centreFine: centre, radiusFine: radius },
    })));
    const after = publish(dropped);
    const injection = after.receipt.lastInjection as
      { accepted?: boolean; cellsWetted?: number; areaAdmittedFine?: number } | null;
    assert.equal(injection?.accepted, true, "the level-set lane refused the drop");
    assert.ok(Number(injection?.cellsWetted) > 0);
    assert.equal(after.revision.injections, 1);
    assert.equal(after.revision.frame, 0, "a drop is an intervention, not a step");
    assert.ok(Math.abs(Number(after.receipt.liquidMeasure) - initialVolume
      - Number(injection?.areaAdmittedFine)) < 1e-4,
    "the published measure does not account for the reader's added mass");
    assert.ok(contourWithin(after, centre, 2 * radius),
      "the ball reached the volume but not the shared level set");

    /* The ball has to survive the stage that reads it: redistancing, pressure
     * geometry and the plan all run off the surface the drop republished. */
    for (let frame = 1; frame <= 4; frame++) {
      const receipt = parsePhysicsReceipt(world.advance(++sequence, 1 / 30));
      assert.equal(receipt.fault, null, `frame ${frame} faulted after the drop`);
      assertRedistancingReceipt(receipt);
      assert.ok(Math.abs(Number(receipt.liquidMeasure) - Number(after.receipt.liquidMeasure))
        / initialVolume < 1e-5, `frame ${frame}: the dropped ball's volume drifted`);
      publish(receipt);
    }
  } finally { world.free(); }
});
