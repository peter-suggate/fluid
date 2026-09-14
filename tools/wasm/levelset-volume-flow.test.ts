import assert from "node:assert/strict";
import test from "node:test";
import { findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument } from "../../lib/core/scene-definition";
import { decodePhysicsPublication } from "../../lib/physics-wasm/publication";
import { createAdvanceView, type AdvanceGraph } from "../../lib/physics-wasm/advance-view";
import { parsePhysicsReceipt } from "../../lib/physics-wasm/protocol";
import { loadFluidWasmForNode } from "./load-module.mjs";

function assertRedistancingReceipt(receipt: Readonly<Record<string, unknown>>): void {
  const value = receipt.levelSetVolume;
  assert.ok(value && typeof value === "object", "the selected transport returns its own receipt");
  const levelSetVolume = value as Readonly<Record<string, unknown>>;
  assert.ok(Number(levelSetVolume.redistancedSamples) > 0,
    "redistancing evaluates the accepted contour at trace landings and cell centres");
  assert.ok(Number(levelSetVolume.redistanceSegmentCount) > 0,
    "redistancing indexes the accepted shared-interface contour");
  assert.ok(Number.isInteger(levelSetVolume.redistanceFallbackSamples)
    && Number(levelSetVolume.redistanceFallbackSamples) >= 0,
  "redistancing publishes its fallback-sample count");
  assert.equal(Number(levelSetVolume.redistanceNanoseconds), 0,
    "Wasm receipts expose the stage while leaving host timing disabled");
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
            const resolution = decoded.metadata.resolution as {
              bricks?: Array<{ brickKey: number; reasons: number }>;
            } | undefined;
            const policyByKey = new Map((resolution?.bricks ?? []).map(brick => [brick.brickKey, brick]));
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
          if (sceneId === "cm12-figure-7" && frame === 25) {
            const resolution = decoded.metadata.resolution as {
              bricks?: Array<{ brickKey: number; reasons: number; planReasons: number }>;
            } | undefined;
            const policyByKey = new Map((resolution?.bricks ?? []).map(brick => [brick.brickKey, brick]));
            const unconstrainedNewSupport = view.graph.bricks.filter(brick => {
              if (!brick.active || brick.coordinate[1]! < 5) return false;
              const policy = policyByKey.get(brick.key);
              const volume = view.lattice.cells.filter(cell => cell.brick === brick.key)
                .reduce((sum, cell) => sum + cell.volume, 0);
              return volume === 0 && policy?.reasons === 0 && policy.planReasons === 0x8000_0001;
            });
            assert.ok(unconstrainedNewSupport.length > 0,
              "Figure 7 allocates empty look-ahead support above the falling liquid");
            assert.ok(unconstrainedNewSupport.every(brick => brick.resolution < 8),
              "unconstrained empty support pages do not default to the finest rung");
          }
          assert.ok(view.rdf.vertexPhiFine.some(Number.isFinite));
          assert.ok(view.rdf.segmentsFine.length > 0, `frame ${frame}: visible RDF surface`);
        } finally { decoded.release(); }
      }
    } finally { world.free(); }
  });
}
