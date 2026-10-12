import "../../lib/methods";
/** Real owned-world + worker protocol + UI decoder scene acceptance; no mock physics. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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
  UNIFORM_LAB_METHOD,
  UNIFORM_LAB_VALUES,
  UNIFORM_PARTICLE_STRIDE,
  type UniformView,
} from "../../lib/physics-wasm/uniform-controller";
import { NARROW_BAND_FINE_ADVECTION, NARROW_BAND_VOLUME_CONTROL } from "../../lib/methods/uniform/uniform-narrow-band-controls";
import { PhysicsWasmWorkerRuntime } from "../../lib/physics-wasm/worker-runtime";
import type { WorkerPort } from "../../lib/physics-wasm/client";
import type {
  PhysicsWorkerResponse,
  PhysicsWorkerRequest,
} from "../../lib/physics-wasm/protocol";
import type { FluidWasmModule } from "../../lib/physics-wasm/module";
{
  const defaults = uniformLabQuery.read(new URLSearchParams());
  assert.equal(defaults.volumeControlSeconds, NARROW_BAND_VOLUME_CONTROL.default, "volume control defaults as in 3D");
  assert.equal(defaults.phiFineAdvection, NARROW_BAND_FINE_ADVECTION.default, "level-set advection defaults as in 3D");
  assert.equal(defaults.dt, 1 / 60, "the lab steps at the 3D method's default");
  const legacy = new URLSearchParams("method=uniform-volume&totalSurfaceVolume=0&phiDrain=1");
  assert.equal(uniformLabQuery.read(legacy).volumeControlSeconds, NARROW_BAND_VOLUME_CONTROL.default, "a Uniform Geometric link opens the method's defaults");
}
assert.equal(UNIFORM_LAB_VALUES.volumeControlSeconds, NARROW_BAND_VOLUME_CONTROL.default);
assert.equal(UNIFORM_LAB_VALUES.phiFineAdvection, NARROW_BAND_FINE_ADVECTION.default);
interface Receipt {
  volume: number; budget: number; controlRate: number; injectedVolume: number; outflowVolume: number;
  particles: number; reseedClipped: number; maxSpeed: number;
  census: { outside: number; unsupported: number; deep: number };
  pressure: { cycles: number; residual: number };
}
const receiptOf = (view: UniformView) => view.receipt.narrowBandFlip as Receipt;
const total = (field: Float32Array) => field.reduce((a, b) => a + b, 0);
function assertFinite(view: UniformView, label: string) {
  for (const field of [view.volume, view.phi, view.velocity, view.pressure, view.particles])
    assert.ok(field.every(Number.isFinite), `${label}: finite fields`);
}
const baseline = new Map<string, unknown>();
/** How far a scene's liquid may stray from its budget at either step: at any
 * frame, and after two seconds. The 24x16 slab splashes to 19% and settles
 * to 5%; the dam breaks stay within 6%. */
const VOLUME_PEAK = 0.2;
const VOLUME_SETTLED = 0.05;
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
      assertFinite(sceneryView, "scenery");
      for (let k = 0; k < sceneryView.particles.length; k += UNIFORM_PARTICLE_STRIDE) {
        const cell = Math.floor(sceneryView.particles[k]!) + sceneryView.nx * Math.floor(sceneryView.particles[k + 1]!);
        assert.ok(sceneryView.capacity[cell]! > 0, "no sample rests inside scenery");
      }
    }
    // The viewer's two controls reach the solver. The samples decide the
    // surface wherever they cover it, so the advection choice first shows
    // once the front thins: at frame 40 here.
    const controls = sceneDocument(findSceneDefinition("water-box-dam-break")!);
    const advected: number[][] = [];
    for (const phiFineAdvection of ["cubic", "trilinear"] as const) {
      let view = await controller.load(controls, { volumeControlSeconds: 0, phiFineAdvection });
      for (let frame = 0; frame < 60; frame++) {
        view = await controller.advance(1 / 60);
        assert.equal(receiptOf(view).controlRate, 0, "0 turns volume control off");
      }
      advected.push([...view.phi]);
    }
    assert.notDeepEqual(advected[0], advected[1], "the level-set advection choice is the solver's");
    for (const sceneId of [
      "water-box-dam-break",
      "minimal-power-dam-break-32",
      "ceiling-slab-drop",
    ]) {
      const scene = sceneDocument(findSceneDefinition(sceneId)!);
      const seed = uniformLabSeed(scene);
      // Volume is read at both steps over the same simulated time.
      for (const [dt, frames] of [[1 / 30, 60], [1 / 60, 120]] as const) {
        const initial = await controller.load(scene);
        assert.equal(initial.revision.frame, 0);
        assert.deepEqual([...initial.capacity], [...new Float32Array(seed.capacity)]);
        assert.equal(initial.receipt.method, UNIFORM_LAB_METHOD);
        assert.equal(initial.particles.length, receiptOf(initial).particles * UNIFORM_PARTICLE_STRIDE);
        assert.ok(receiptOf(initial).particles > 0, `${sceneId}: the band is seeded`);
        const budget = receiptOf(initial).budget;
        assert.ok(Math.abs(total(initial.volume) - budget) < 1e-3 * budget, `${sceneId}: the budget is the liquid loaded`);
        let view = initial, gap = 0, fastest = 0, outflow = 0;
        const costs: number[] = [];
        for (let frame = 1; frame <= frames; frame++) {
          const start = performance.now();
          view = await controller.advance(dt);
          costs.push(performance.now() - start);
          assert.equal(view.revision.frame, frame);
          assert.equal(view.revision.runEpoch, initial.revision.runEpoch);
          assertFinite(view, `${sceneId} ${frame}`);
          const receipt = receiptOf(view);
          assert.ok(receipt.particles > 0, `${sceneId} ${frame}: the band keeps its samples`);
          outflow += receipt.outflowVolume;
          assert.ok(Math.abs(receipt.budget + outflow - budget) < 1e-3 * budget, `${sceneId} ${frame}: the budget moves only by outflow`);
          gap = Math.max(gap, Math.abs(receipt.volume - receipt.budget) / receipt.budget);
          fastest = Math.max(fastest, receipt.maxSpeed);
        }
        const receipt = receiptOf(view);
        assert.ok(fastest > 0.05, `${sceneId}: the liquid moves`);
        assert.ok(gap < VOLUME_PEAK, `${sceneId} at 1/${Math.round(1 / dt)} s: volume strays ${(100 * gap).toFixed(2)}% from its budget`);
        assert.ok(
          Math.abs(receipt.volume - receipt.budget) < VOLUME_SETTLED * receipt.budget,
          `${sceneId} at 1/${Math.round(1 / dt)} s: volume ends ${(100 * (receipt.volume - receipt.budget) / receipt.budget).toFixed(2)}% from its budget`,
        );
        const final = {
          volume: [...view.volume],
          phi: [...view.phi],
          velocity: [...view.velocity],
          pressure: [...view.pressure],
          particles: [...view.particles],
          tiles: [...view.tiles],
        };
        const key = `${sceneId}@${frames}`;
        if (artifact === "scalar") baseline.set(key, final);
        else
          assert.deepEqual(
            final,
            baseline.get(key),
            `${key}: scalar/SIMD owned-world parity`,
          );
        const reset = await controller.load(scene);
        assert.equal(reset.revision.frame, 0);
        assert.ok(reset.revision.runEpoch > initial.revision.runEpoch);
        assert.deepEqual(reset.volume, initial.volume);
        assert.deepEqual(reset.phi, initial.phi);
        assert.deepEqual(reset.particles, initial.particles);
        const stepped = await controller.advance(dt);
        assert.equal(stepped.revision.frame, 1);
        assert.deepEqual(
          reset.volume,
          initial.volume,
          "publication stays owned after the slot is recycled",
        );
        costs.sort((a, b) => a - b);
        console.log(JSON.stringify({
          artifact,
          scene: sceneId,
          dt: `1/${Math.round(1 / dt)}`,
          frames,
          dimensions: [view.nx, view.ny],
          budget: receipt.budget,
          volume: receipt.volume,
          finalGapPercent: 100 * (receipt.volume - receipt.budget) / receipt.budget,
          maxGapPercent: 100 * gap,
          outflow,
          particles: receipt.particles,
          ballistic: receipt.census.unsupported,
          reseedClipped: receipt.reseedClipped,
          fastest,
          medianMs: costs[costs.length >> 1],
          p95Ms: costs[Math.floor(0.95 * costs.length)],
        }));
      }
    }
    const scene = sceneDocument(findSceneDefinition("water-box-dam-break")!);
    let edited = await controller.load(scene);
    const before = total(edited.volume);
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
    assert.ok(total(edited.volume) > before);
    assert.ok(
      Math.abs(receiptOf(edited).budget - before - Number(edited.receipt.injectedVolume)) < 1e-3 * before,
      "dropped liquid joins the budget",
    );
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
      assertFinite(edited, "live body scene");
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
      particles: [...edited.particles],
      bodies: edited.receipt.rigidBodies,
    };
    if (artifact === "scalar") baseline.set("live-tools", live);
    else
      assert.deepEqual(
        live,
        baseline.get("live-tools"),
        "live tools scalar/SIMD parity",
      );
    console.log(JSON.stringify({
      artifact,
      scene: "live-liquid-and-rigid-tools",
      frames: 36,
      bodyY: pose.position_m.y,
      injectedVolume: edited.receipt.injectedVolume,
      liquidVolume: total(edited.volume),
      budget: receiptOf(edited).budget,
    }));
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
      volumeControlSeconds: 0.5,
      phiFineAdvection: "trilinear" as const,
      sliceDepth_m: -0.15,
      sceneId: "water-box-dam-break",
      dt: 1 / 60,
      layers: visualLayers(["pressure", "particles", "grid"]),
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
    assert.deepEqual(restoredView.particles, authoredView.particles);
    assert.deepEqual(
      restoredView.receipt.rigidBodies,
      authoredView.receipt.rigidBodies,
    );
    await controller.advance(ui.dt);

    // A submerged light disk must rise in a quiet pool with planar mass.
    const pool = {
      ...scene,
      container: { ...scene.container, fillFraction: 0.8 },
      fluid: { ...scene.fluid, initialCondition: "tank-fill" as const },
    };
    // The pool rests under the box's lid at either step: its top rows are
    // where a coarse pressure level can lose its free surface.
    for (const [dt, frames] of [[1 / 30, 60], [1 / 60, 120]] as const) {
      let view = await controller.load(pool);
      let fastest = 0;
      for (let frame = 0; frame < frames; frame++) {
        view = await controller.advance(dt);
        fastest = Math.max(fastest, receiptOf(view).maxSpeed);
      }
      const receipt = receiptOf(view);
      const gap = (receipt.volume - receipt.budget) / receipt.budget;
      console.log(JSON.stringify({ artifact, scene: "resting-pool-under-lid", dt: `1/${Math.round(1 / dt)}`, gapPercent: 100 * gap, fastest, residual: receipt.pressure.residual }));
      assert.ok(Math.abs(gap) < 0.015, `resting pool at 1/${Math.round(1 / dt)} s: volume ${(100 * gap).toFixed(2)}% from its budget`);
      assert.ok(fastest < 0.02, `resting pool at 1/${Math.round(1 / dt)} s: moves at ${fastest} m/s`);
    }
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
    console.log(JSON.stringify({ artifact, scene: "planar-rigid-buoyancy", frames: 30, bodyY: floatingPose.position_m.y }));
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
    assertFinite(edited, "all shared rigid shapes");
    const shapes = {
      volume: [...edited.volume],
      phi: [...edited.phi],
      bodies: edited.receipt.rigidBodies,
    };
    if (artifact === "scalar") baseline.set("rigid-shapes", shapes);
    else assert.deepEqual(shapes, baseline.get("rigid-shapes"));
  } finally {
    await controller.destroy();
  }
}
