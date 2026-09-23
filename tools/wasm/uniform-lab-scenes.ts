import "../../lib/methods";
/** Real owned-world + worker protocol + UI decoder scene acceptance; no mock physics. */
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { addFluidBall } from "../../lib/core/editor-fluid-volume";
import {
  createSceneQueryLayerCache,
  parseQueryState,
} from "../../lib/core/url-state";
import { visualLayers } from "../../lib/core/visual-layers";
import { uniformLabQuery } from "../../advance-lab/uniform-lab-state";
import { createBodyDescription } from "../../lib/core/rigid-body";
import { findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument } from "../../lib/core/scene-definition";
import {
  UniformLabController,
  uniformLabSeed,
  UNIFORM_LAB_VALUES,
} from "../../lib/physics-wasm/uniform-controller";
import { UNIFORM_GEOMETRIC_DEFAULTS, UNIFORM_GEOMETRIC_NATIVE_PARAMS } from "../../lib/methods/uniform/uniform-geometric-parameters";
import { PhysicsWasmWorkerRuntime } from "../../lib/physics-wasm/worker-runtime";
import type { WorkerPort } from "../../lib/physics-wasm/client";
import type {
  PhysicsWorkerResponse,
  PhysicsWorkerRequest,
} from "../../lib/physics-wasm/protocol";
import type { FluidWasmModule } from "../../lib/physics-wasm/module";
for (const key of ["totalSurfaceVolume","surfaceDeficitBalancing"] as const) {
  assert.equal(uniformLabQuery.read(new URLSearchParams())[key],true,`${key} defaults on, as in 3D`);
  const query=new URLSearchParams();
  uniformLabQuery.write(query,{...uniformLabQuery.read(query),[key]:false});
  assert.equal(query.get(key),"0");
  assert.equal(uniformLabQuery.read(query)[key],false);
}
const summaries: unknown[] = [];
const baseline = new Map<string, unknown>();
assert.deepEqual(
  UNIFORM_LAB_VALUES,
  Object.fromEntries(UNIFORM_GEOMETRIC_NATIVE_PARAMS.map((p) => [p.key, UNIFORM_GEOMETRIC_DEFAULTS[p.key]])),
  "the lab runs the 3D defaults",
);
for (const artifact of ["scalar", "simd"] as const) {
  const root = new URL(
    `../../public/wasm/fluid-wasm/${artifact}/`,
    import.meta.url,
  );
  const wasm = (await import(
    new URL("fluid_wasm.js", root).href
  )) as FluidWasmModule;
  await wasm.default({
    module_or_path: readFileSync(new URL("fluid_wasm_bg.wasm", root)),
  });
  const factory = (): WorkerPort => {
    const listeners = new Set<
      (e: MessageEvent<PhysicsWorkerResponse>) => void
    >();
    const runtime = new PhysicsWasmWorkerRuntime(
      (message, transfer) => {
        const data = structuredClone(message, { transfer: transfer ?? [] });
        queueMicrotask(() =>
          listeners.forEach((listener) =>
            listener({ data } as MessageEvent<PhysicsWorkerResponse>),
          ),
        );
      },
      async () => ({ ...wasm, default: async () => {} }),
    );
    return {
      postMessage(message: PhysicsWorkerRequest, transfer?: Transferable[]) {
        runtime.receive(structuredClone(message, { transfer: transfer ?? [] }));
      },
      addEventListener(type: string, listener: unknown) {
        if (type === "message")
          listeners.add(
            listener as (e: MessageEvent<PhysicsWorkerResponse>) => void,
          );
      },
      terminate() {
        listeners.clear();
      },
    };
  };
  const controller = await UniformLabController.create({
    artifact,
    workerFactory: factory,
  });
  try {
    const sceneryScene = sceneDocument(findSceneDefinition("water-box-dam-break")!);
    sceneryScene.rigidBodies = [];
    sceneryScene.scenery = { palettes: {}, nodes: [{ kind: "terrain-shell", id: "shell", materialModel: "porcelain" }, {
      kind: "box", id: "scenery-wall",
      place: { units: "metres", position: { x: 0, y: sceneryScene.container.height_m / 2, z: 0 } },
      halfSize: { x: sceneryScene.voxelDomain.finestCellSize_m,
        y: sceneryScene.container.height_m / 2, z: sceneryScene.container.depth_m / 2 },
      material: { colorLinear: [1, 1, 1] },
    }] };
    const scenerySeed = uniformLabSeed(sceneryScene, 0);
    let sceneryView = await controller.load(sceneryScene, undefined, 0);
    const wallCells = scenerySeed.capacity.flatMap((capacity, i) => {
      const x = (i % sceneryView.nx + 0.5) * sceneryView.cellSize[0] - sceneryScene.container.width_m / 2;
      return Math.abs(x) < sceneryScene.voxelDomain.finestCellSize_m && capacity === 0 ? [i] : [];
    });
    assert.ok(wallCells.length >= sceneryView.ny, "scenery wall is voxelized");
    for (let step = 0; step < 12; step++) {
      sceneryView = await controller.advance(1 / 30);
      for (const i of wallCells) {
        assert.equal(sceneryView.capacity[i], 0, "scenery capacity survives advance");
        assert.equal(sceneryView.volume[i], 0, "water cannot enter scenery");
      }
      assert.ok(sceneryView.volume.every(Number.isFinite));
    }
    const balancedScene = sceneDocument(findSceneDefinition("coarse-first-pool-impact-half")!);
    let balancedView = await controller.load(balancedScene);
    assert.equal(balancedView.receipt.surfaceDeficitBalancing, true, "on by default, as in 3D");
    balancedView = await controller.advance(1 / 30);
    assert.ok(balancedView.velocity.every(Number.isFinite));
    const baselineView = await controller.load(balancedScene, { totalSurfaceVolume: true, surfaceDeficitBalancing: false });
    assert.equal(baselineView.receipt.surfaceDeficitBalancing, false, "the viewer's opt-out reaches the solver");
    assert.equal(baselineView.revision.frame, 0);
    for (const sceneId of [
      "water-box-dam-break",
      "minimal-power-dam-break-32",
      "ceiling-slab-drop",
    ]) {
      const scene = sceneDocument(findSceneDefinition(sceneId)!);
      const seed = uniformLabSeed(scene);
      const initial = await controller.load(scene);
      assert.equal(initial.revision.frame, 0);
      assert.deepEqual([...initial.volume], [...new Float32Array(seed.volume)]);
      assert.deepEqual([...initial.phi], [...new Float32Array(seed.phi)]);
      assert.equal(initial.receipt.method, "uniform-volume");
      const initialMass = initial.volume.reduce((a, b) => a + b, 0);
      let view = initial,
        dust = 0;
      const costs: number[] = [];
      for (let frame = 1; frame <= 60; frame++) {
        const start = performance.now();
        view = await controller.advance(1 / 30);
        costs.push(performance.now() - start);
        assert.equal(view.revision.frame, frame);
        assert.equal(view.revision.runEpoch, initial.revision.runEpoch);
        for (const field of [
          view.volume,
          view.phi,
          view.velocity,
          view.pressure,
        ])
          assert.ok(
            field.every(Number.isFinite),
            `${sceneId} ${frame}: finite fields`,
          );
        const receipt = view.receipt.uniform as {
          transport: { dustVolume: number };
          sharpeningDust: number;
        };
        dust += receipt.transport.dustVolume + receipt.sharpeningDust;
      }
      const finalMass = view.volume.reduce((a, b) => a + b, 0);
      assert.ok(
        Math.abs(finalMass + dust - initialMass) <
          1e-5 * Math.max(1, initialMass),
        `${sceneId}: conservative V with reported dust`,
      );
      const final = {
        volume: [...view.volume],
        phi: [...view.phi],
        velocity: [...view.velocity],
        pressure: [...view.pressure],
        released: [...view.released],
        tiles: [...view.tiles],
      };
      if (artifact === "scalar") baseline.set(sceneId, final);
      else
        assert.deepEqual(
          final,
          baseline.get(sceneId),
          `${sceneId}: scalar/SIMD owned-world parity`,
        );
      const reset = await controller.load(scene);
      assert.equal(reset.revision.frame, 0);
      assert.ok(reset.revision.runEpoch > initial.revision.runEpoch);
      assert.deepEqual(reset.volume, initial.volume);
      assert.deepEqual(reset.phi, initial.phi);
      const stepped = await controller.advance(1 / 30);
      assert.equal(stepped.revision.frame, 1);
      assert.deepEqual(
        reset.volume,
        initial.volume,
        "publication stays owned after the slot is recycled",
      );
      costs.sort((a, b) => a - b);
      const summary = {
        artifact,
        scene: sceneId,
        frames: 60,
        dimensions: [view.nx, view.ny],
        initialMass,
        finalMass,
        dust,
        medianMs: costs[30],
        p95Ms: costs[57],
      };
      summaries.push(summary);
      console.log(JSON.stringify(summary));
    }
    for (const totalSurfaceVolume of [true, false]) {
      const scene = sceneDocument(findSceneDefinition("water-box-dam-break")!);
      let view = await controller.load(scene, { totalSurfaceVolume, surfaceDeficitBalancing: true });
      assert.equal(view.receipt.totalSurfaceVolume, totalSurfaceVolume);
      for (let frame=0; frame<12; frame++) view=await controller.advance(1/30);
      const receipt=view.receipt.uniform as {contourArea:number; volume:number};
      if (totalSurfaceVolume) assert.ok(Math.abs(receipt.contourArea-receipt.volume)<0.03);
      const fields={volume:[...view.volume], phi:[...view.phi], velocity:[...view.velocity]};
      const key=`tsv-${totalSurfaceVolume}`;
      if (artifact==="scalar") baseline.set(key,fields);
      else assert.deepEqual(fields,baseline.get(key), `${key}: scalar/SIMD parity`);
    }
    const scene = sceneDocument(findSceneDefinition("water-box-dam-break")!);
    let edited = await controller.load(scene);
    const before = edited.volume.reduce((a, b) => a + b, 0);
    edited = await controller.injectLiquid([0.8, 0.6], 0.1);
    assert.equal(
      edited.revision.frame,
      0,
      "dropping queues a source without advancing time",
    );
    assert.equal((edited.receipt.pendingDrops as unknown[]).length, 1);
    edited = await controller.advance(1 / 60);
    assert.ok(
      Number(edited.receipt.injectedVolume) > 0,
      "source adds liquid in the next advance",
    );
    assert.ok(edited.volume.reduce((a, b) => a + b, 0) > before);
    const body = {
      ...createBodyDescription("sphere", 0, scene.container.height_m),
      position_m: { x: 0.35, y: 0.6, z: 0 },
      dimensions_m: { x: 0.07, y: 0.07, z: 0.07 },
    };
    edited = await controller.addRigidBody(body, true);
    assert.equal((edited.receipt.rigidBodies as unknown[]).length, 1);
    edited = await controller.setRigidPose(
      body.id,
      { x: 0.3, y: 0.5, z: 0 },
      { x: 0, y: 0, z: 0 },
      true,
    );
    for (let frame = 0; frame < 5; frame++)
      edited = await controller.advance(1 / 60);
    let pose = (
      edited.receipt.rigidBodies as {
        position_m: { x: number; y: number };
        held: boolean;
      }[]
    )[0]!;
    assert.ok(
      Math.abs(pose.position_m.y - 0.5) < 1e-6,
      "held body stays in the hand",
    );
    edited = await controller.setRigidPose(
      body.id,
      { x: 0.3, y: 0.5, z: 0 },
      { x: 0, y: 0, z: 0 },
      false,
    );
    for (let frame = 0; frame < 30; frame++) {
      edited = await controller.advance(1 / 60);
      for (const field of [
        edited.volume,
        edited.phi,
        edited.velocity,
        edited.pressure,
      ])
        assert.ok(
          field.every(Number.isFinite),
          "live body scene remains finite",
        );
    }
    pose = (
      edited.receipt.rigidBodies as {
        position_m: { x: number; y: number };
        held: boolean;
      }[]
    )[0]!;
    assert.ok(pose.position_m.y < 0.49, "released body moves under gravity");
    assert.ok(pose.position_m.y >= 0, "body respects the floor");
    const live = {
      volume: [...edited.volume],
      phi: [...edited.phi],
      velocity: [...edited.velocity],
      bodies: edited.receipt.rigidBodies,
    };
    if (artifact === "scalar") baseline.set("live-tools", live);
    else
      assert.deepEqual(
        live,
        baseline.get("live-tools"),
        "live tools scalar/SIMD parity",
      );
    summaries.push({
      artifact,
      scene: "live-liquid-and-rigid-tools",
      frames: 36,
      bodyY: pose.position_m.y,
      injectedVolume: edited.receipt.injectedVolume,
      liquidVolume: edited.volume.reduce((a, b) => a + b, 0),
      volumeChange:
        edited.volume.reduce((a, b) => a + b, 0) -
        before -
        Number(edited.receipt.injectedVolume),
    });
    console.log(JSON.stringify(summaries.at(-1)));
    edited = await controller.removeRigidBody(body.id);
    assert.deepEqual(edited.receipt.rigidBodies, []);
    assert.deepEqual(
      [...edited.capacity],
      uniformLabSeed(scene).capacity,
      "removing the last body restores static capacity",
    );
    edited = await controller.load(scene);
    assert.deepEqual(edited.receipt.rigidBodies, []);
    assert.deepEqual(edited.receipt.pendingDrops, []);
    assert.equal(edited.revision.frame, 0);
    assert.equal(edited.receipt.injectedVolume, 0);
    // Author with the production liquid tool, serialize with the studio scene
    // layer, reload the URL, then run the restored document in the owned world.
    const drop = addFluidBall(
      {
        ...scene,
        container: { ...scene.container, depthBoundary: "symmetry" },
      },
      { x: 0.35, y: 0.65, z: 0 },
      0.08,
    );
    const authored = { ...scene, fluid: drop.fluid, rigidBodies: [body] };
    const params = new URLSearchParams(
      createSceneQueryLayerCache()({
        scene: authored,
        presetId: "water-box-dam-break",
      }).map(([k, v]) => [k, v]),
    );
    const ui = {
      totalSurfaceVolume: false,
      surfaceDeficitBalancing: false,
      sliceDepth_m: -0.15,
      sceneId: "water-box-dam-break",
      dt: 1 / 60,
      layers: visualLayers(["pressure", "grid"]),
      sliceView: { x: 0.4, y: 0.6, zoom: 2 },
    };
    uniformLabQuery.write(params, ui);
    assert.deepEqual(uniformLabQuery.read(params), ui);
    const restored = parseQueryState(params.toString()).scene;
    assert.deepEqual(
      restored.fluid.initialLiquidVolumes,
      authored.fluid.initialLiquidVolumes,
    );
    assert.deepEqual(restored.rigidBodies, authored.rigidBodies);
    const authoredView = await controller.load(authored);
    const restoredView = await controller.load(restored);
    assert.deepEqual(restoredView.volume, authoredView.volume);
    assert.deepEqual(restoredView.phi, authoredView.phi);
    assert.deepEqual(
      restoredView.receipt.rigidBodies,
      authoredView.receipt.rigidBodies,
    );
    await controller.advance(ui.dt);
    summaries.push({ artifact, scene: "authored-tool-url-reload", frames: 1 });

    // A submerged light disk must rise in a quiet pool with planar mass.
    const pool = {
      ...scene,
      container: { ...scene.container, fillFraction: 0.8 },
      fluid: { ...scene.fluid, initialCondition: "tank-fill" as const },
    };
    await controller.load(pool);
    const floating = {
      ...body,
      position_m: { x: 0, y: 0.3, z: 0 },
      angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    };
    edited = await controller.addRigidBody(floating);
    for (let frame = 0; frame < 30; frame++)
      edited = await controller.advance(1 / 120);
    const floatingPose = (
      edited.receipt.rigidBodies as { position_m: { y: number } }[]
    )[0]!;
    assert.ok(
      floatingPose.position_m.y > 0.31,
      `submerged light disk rises: ${floatingPose.position_m.y}`,
    );
    summaries.push({
      artifact,
      scene: "planar-rigid-buoyancy",
      frames: 30,
      bodyY: floatingPose.position_m.y,
    });
    edited = await controller.load(scene);
    for (const [index, shape] of (
      ["sphere", "box", "capsule", "cylinder", "cup"] as const
    ).entries()) {
      const description = {
        ...createBodyDescription(shape, index, scene.container.height_m),
        position_m: { x: -0.45 + 0.22 * index, y: 0.65, z: 0 },
      };
      const withShape = { ...scene, rigidBodies: [description] };
      const shapeQuery = new URLSearchParams(
        createSceneQueryLayerCache()({
          scene: withShape,
          presetId: "water-box-dam-break",
        }).map(([k, v]) => [k, v]),
      );
      shapeQuery.set("scene", "water-box-dam-break");
      const restoredShape = parseQueryState(shapeQuery.toString()).scene;
      assert.deepEqual(
        restoredShape.rigidBodies,
        withShape.rigidBodies,
        `${shape} survives shared URL validation`,
      );
      edited = await controller.addRigidBody(restoredShape.rigidBodies[0]!);
    }
    for (let frame = 0; frame < 15; frame++)
      edited = await controller.advance(1 / 120);
    assert.equal((edited.receipt.rigidBodies as unknown[]).length, 5);
    for (const field of [
      edited.volume,
      edited.phi,
      edited.velocity,
      edited.pressure,
    ])
      assert.ok(
        field.every(Number.isFinite),
        "all shared rigid shapes remain finite",
      );
    const shapes = {
      volume: [...edited.volume],
      phi: [...edited.phi],
      bodies: edited.receipt.rigidBodies,
    };
    if (artifact === "scalar") baseline.set("rigid-shapes", shapes);
    else assert.deepEqual(shapes, baseline.get("rigid-shapes"));
    summaries.push({
      artifact,
      scene: "shared-rigid-shape-roster",
      frames: 15,
      shapes: 5,
    });
  } finally {
    await controller.destroy();
  }
}
writeFileSync(
  "docs/research/uniform-geometric-2d-2026-09-20/ui-scene-acceptance.json",
  JSON.stringify({ profile: UNIFORM_LAB_VALUES, scenes: summaries }, null, 2) +
    "\n",
);
