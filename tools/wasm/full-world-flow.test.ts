import assert from "node:assert/strict";
import test from "node:test";
import { findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument } from "../../lib/core/scene-definition";
import { createAdvanceView, type AdvanceGraph, type AdvanceView } from "../../lib/physics-wasm/advance-view";
import { decodePhysicsPublication } from "../../lib/physics-wasm/publication";
import { parsePhysicsReceipt, type PhysicsCommandReceipt, type PhysicsPublication } from "../../lib/physics-wasm/protocol";
import { loadFluidWasmForNode } from "./load-module.mjs";

interface NativeWorld {
  advance(sequence: number, dt: number): string;
  apply_command(command: string): string;
  receipt(): string;
  snapshot(mask: number): Uint8Array;
  free(): void;
}

const authored = (id: string) => {
  const definition = findSceneDefinition(id);
  if (!definition) throw new Error(`Missing scene ${id}`);
  return { id, label: definition.name, document: sceneDocument(definition) };
};

function snapshot(world: NativeWorld, receipt: PhysicsCommandReceipt,
  graph: AdvanceGraph | undefined, scene: ReturnType<typeof authored>): AdvanceView {
  const bytes = world.snapshot(0xf).slice();
  const publication: PhysicsPublication = { id: 0, revision: receipt, bytes, release() {} };
  const decoded = decodePhysicsPublication(publication);
  try { return createAdvanceView(decoded, graph, scene); }
  finally { decoded.release(); }
}

const command = (world: NativeWorld, sequence: number, runEpoch: number,
  body: Record<string, unknown>): PhysicsCommandReceipt => parsePhysicsReceipt(world.apply_command(
    JSON.stringify({ ...body, commandSequence: sequence, runEpoch })));

test("real scalar Wasm world publishes the complete Advance Lab flow", async () => {
  const wasm = await loadFluidWasmForNode();
  const first = authored("water-box-dam-break");
  let world = wasm.FluidWorld.from_scene(JSON.stringify(first.document), JSON.stringify({
    runEpoch: 1, commandSequence: 0, pressureIterations: 12, tracerBudget: 64,
    production: { dtS: 1 / 30, timeStep: "paper" },
  })) as NativeWorld;
  try {
    let receipt = parsePhysicsReceipt(world.receipt());
    let view = snapshot(world, receipt, undefined, first);
    assert.equal(view.revision.frame, 0);
    assert.ok(view.nx > 1 && view.ny > 1 && view.graph.cells.length > 0);
    assert.equal(view.previousLiquidVolumeFine.length, view.nx * view.ny);
    assert.equal(view.faceVelocityXBeforePressure.length, (view.nx + 1) * view.ny);
    assert.equal(view.faceVelocityYBeforePressure.length, view.nx * (view.ny + 1));
    assert.equal(view.previousBrickRung.length, view.bx * view.by);
    assert.equal(view.brickActivity.length, view.bx * view.by);
    assert.equal(view.materialFine.length, view.nx * view.ny);
    assert.equal(view.capacityFine.length, view.nx * view.ny);
    const initialGraph = view.graph;

    receipt = parsePhysicsReceipt(world.advance(1, 1 / 30));
    view = snapshot(world, receipt, initialGraph, first);
    assert.equal(view.revision.commandSequence, 1);
    assert.equal(view.revision.frame, 1);
    assert.ok(view.lattice.cells.some(cell => cell.open && cell.capacity > 0));
    assert.ok(view.previousLiquidVolumeFine.some(Number.isFinite));

    receipt = command(world, 2, 1, { type: "inject-liquid",
      drop: { centreFine: [view.nx * .5, view.ny * .72], radiusFine: 2 } });
    view = snapshot(world, receipt, view.graph, first);
    assert.equal(view.revision.injections, 1);
    assert.equal((view.receipt.lastInjection as { accepted?: boolean } | null)?.accepted, true);

    receipt = command(world, 3, 1, { type: "set-refinement-regions", regions: [{
      minimumFine: [0, 0], maximumFine: [Math.min(8, view.nx), Math.min(8, view.ny)],
      minimumCellWidth: 2, maximumCellWidth: 2,
    }] });
    view = snapshot(world, receipt, view.graph, first);
    assert.equal(view.revision.commandSequence, 3);
    assert.equal(view.graph.dimension, 2);

    receipt = command(world, 4, 1, { type: "set-time-step", dt_s: 1 / 60 });
    receipt = command(world, 5, 1, { type: "set-pressure-budget",
      iterations: 20, relativeTolerance: 1e-6 });
    receipt = command(world, 6, 1, { type: "set-tracers", enabled: true });
    receipt = command(world, 7, 1, { type: "reseed-tracers" });
    receipt = parsePhysicsReceipt(world.advance(8, 1 / 60));
    view = snapshot(world, receipt, view.graph, first);
    assert.equal(view.revision.commandSequence, 8);
    assert.equal(view.metadata.tracersEnabled, true);
    assert.ok(view.markers.length > 0 && view.markers.some(marker => marker.alive),
      "published stable tracer slots include at least one live marker after reseeding");
  } finally { world.free(); }

  const reset = authored("coarse-first-pool-impact-quarter");
  world = wasm.FluidWorld.from_scene(JSON.stringify(reset.document), JSON.stringify({
    runEpoch: 2, commandSequence: 0, pressureIterations: 12,
    production: { dtS: 1 / 30, timeStep: "paper" },
  })) as NativeWorld;
  try {
    const receipt = parsePhysicsReceipt(world.receipt());
    const view = snapshot(world, receipt, undefined, reset);
    assert.equal(view.revision.runEpoch, 2);
    assert.equal(view.revision.frame, 0);
    assert.equal(view.scene.id, reset.id);
    assert.ok(view.rdf.vertexPhiFine.length > 0);
  } finally { world.free(); }
});
