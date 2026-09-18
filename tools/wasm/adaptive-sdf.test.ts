import assert from "node:assert/strict";
import test from "node:test";
import { findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument } from "../../lib/core/scene-definition";
import { decodePhysicsPublication } from "../../lib/physics-wasm/publication";
import { createAdvanceView } from "../../lib/physics-wasm/advance-view";
import { parsePhysicsReceipt } from "../../lib/physics-wasm/protocol";
import { loadFluidWasmForNode } from "./load-module.mjs";

for (const artifact of ["scalar", "simd"] as const) {
  test(`${artifact}: adaptive SDF defaults on; off retains the fine-grid comparison`, async () => {
    const wasm = await loadFluidWasmForNode(undefined, { artifact });
    const definition = findSceneDefinition("coarse-first-pool-impact-half-slab");
    assert.ok(definition);
    const scene = { id: definition.id, label: definition.name, document: sceneDocument(definition) };
    const finalFields: Float32Array[] = [];
    for (const adaptive of [true, false]) {
      const world = wasm.FluidWorld.from_scene(JSON.stringify(scene.document), JSON.stringify({
        runEpoch: 1, commandSequence: 0, pressureIterations: 256,
        pressureRelativeTolerance: 1e-6, transportExperiment: "level-set-volume", tracerBudget: 0,
        // Omit the on value to exercise the same default used by old links.
        ...(adaptive ? {} : { adaptiveSdf: false }),
        production: { dtS: 0.009, timeStep: "paper" },
      }));
      try {
        const volume = Number(parsePhysicsReceipt(world.receipt()).liquidMeasure);
        for (let frame = 0; frame <= 9; frame++) {
          const receipt = parsePhysicsReceipt(frame ? world.advance(frame, 0.009) : world.receipt());
          assert.equal(receipt.frame, frame);
          assert.equal(receipt.fault, null);
          assert.ok(Math.abs(Number(receipt.liquidMeasure) - volume) / volume < 1e-6);
          const decoded = decodePhysicsPublication({ id: frame, revision: receipt,
            bytes: world.snapshot(0xf).slice(), release() {} });
          try {
            const view = createAdvanceView(decoded, undefined, scene);
            const sdf = decoded.metadata.sdf as { adaptive: boolean; vertexCount: number; constrainedVertexCount: number };
            assert.equal(sdf.adaptive, adaptive);
            assert.ok(view.rdf.vertexPhiFine.every(Number.isFinite));
            assert.ok(view.rdf.segmentsFine.length > 0);
            if (adaptive) {
              const corners = new Set(view.graph.cells.flatMap(cell => [
                `${cell.minimum[0]},${cell.minimum[1]}`, `${cell.maximum[0]},${cell.minimum[1]}`,
                `${cell.minimum[0]},${cell.maximum[1]}`, `${cell.maximum[0]},${cell.maximum[1]}`,
              ]));
              assert.equal(sdf.vertexCount, corners.size, "only accepted shared corners own phi");
              assert.ok(sdf.vertexCount < view.rdf.vertexPhiFine.length, "fine raster is only a publication");
              assert.ok(sdf.constrainedVertexCount > 0, "mixed cells constrain hanging vertices");
            } else {
              assert.equal(sdf.vertexCount, view.rdf.vertexPhiFine.length);
              assert.equal(sdf.constrainedVertexCount, 0);
            }
            if (frame) {
              const step = receipt.levelSetVolume as Record<string, unknown>;
              assert.equal(step.adaptiveSdf, adaptive);
              assert.ok(Number(adaptive ? step.redistanceSeedCount : step.redistanceSegmentCount) > 0);
            }
            if (frame === 9) finalFields.push(view.rdf.vertexPhiFine.slice());
          } finally { decoded.release(); }
        }
      } finally { world.free(); }
    }
    assert.ok(finalFields[0]!.some((value, i) => Math.abs(value - finalFields[1]![i]!) > 0.01),
      "switch changes the evolved representation, not just its drawing");
  });
}

test("SIMD dam-break coarsening does not create the late phantom pool", async () => {
  const wasm = await loadFluidWasmForNode(undefined, { artifact: "simd" });
  const definition = findSceneDefinition("water-box-dam-break");
  assert.ok(definition);
  const scene = { id: definition.id, label: definition.name, document: sceneDocument(definition) };
  const world = wasm.FluidWorld.from_scene(JSON.stringify(scene.document), JSON.stringify({
    runEpoch: 1, commandSequence: 0, pressureIterations: 256, pressureRelativeTolerance: 1e-6,
    transportExperiment: "level-set-volume", adaptiveSdf: true, tracerBudget: 0,
    production: { dtS: 1 / 30, timeStep: "paper" },
  }));
  let sawCoarsening = false;
  try {
    const initialVolume = Number(parsePhysicsReceipt(world.receipt()).liquidMeasure);
    for (let frame = 1; frame <= 160; frame++) {
      const receipt = parsePhysicsReceipt(world.advance(frame, 1 / 30));
      assert.equal(receipt.fault, null);
      assert.ok(Math.abs(Number(receipt.liquidMeasure) - initialVolume) < 1e-4);
      const decoded = decodePhysicsPublication({ id: frame, revision: receipt,
        bytes: world.snapshot(0xf).slice(), release() {} });
      try {
        const view = createAdvanceView(decoded, undefined, scene);
        const resolution = decoded.metadata.resolution as { demotedBrickCount: number };
        sawCoarsening ||= resolution.demotedBrickCount > 0;
        if (frame < 80) continue;
        const transport = receipt.levelSetVolume as { phiImpliedLiquidVolume: number };
        const after = Number(view.rdf.receipt!.representedAreaFine);
        // In this settling pool, a whole-width, one-fine-cell displacement is
        // already a generous smoke bound. The old frame-100 transfer adds
        // 54.7 fine-cell² across a domain only 24 cells wide. Compare the same
        // frame before/after remesh, excluding physical transport from the error.
        assert.ok(Math.abs(after - transport.phiImpliedLiquidVolume) <= view.nx,
          `frame ${frame}: remeshing changed area from ${transport.phiImpliedLiquidVolume} to ${after}`);
        assert.ok(view.rdf.vertexPhiFine.every(Number.isFinite));
      } finally { decoded.release(); }
    }
    assert.ok(sawCoarsening, "stability must retain adaptive demotion");
  } finally { world.free(); }
});
