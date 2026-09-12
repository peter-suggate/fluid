import assert from "node:assert/strict";
import test from "node:test";
import {
  baseInitialLiquidFractionAtCell,
  initialLiquidFractionAtCell,
} from "../../../core/initial-fluid";
import { sceneDocument } from "../../../core/scene-definition";
import { sceneLatticeDimensions } from "../../../core/scene-lattice";
import { SCENE_CATALOG } from "../../../core/scenes";
import { fluidSolidWorldForScene, sampleSolidWorld } from "../../../core/solid-world";
import {
  ADVANCE_PRODUCTION_SCENES,
  advanceProductionSceneDefinition,
  productionSceneSliceSeedById,
} from "./production-scene-slice";
import { sliceDynamicGeometry, type SliceRigidPose } from "./slice-dynamic-geometry";
import { commitSliceSourceLedger, planSliceDynamicRemap } from "./slice-dynamic-remap";
import { advanceSliceRigidAuthority, createSliceRigidAuthority,
  sliceRigidCouplingLoads, sliceRigidPoses } from "./slice-rigid-dynamics";
import { validateSliceSceneSeed } from "./slice-scene-seed";
import { buildSliceLattice, createSliceLattice, latticePlane } from "./slice-lattice";
import { advanceSlice, clipUnitSquare, createAdvanceSlice, resetAdvanceSlice,
  UNIT_SQUARE } from "./slice-solver";

function polygonArea(points: readonly number[]): number {
  let twice = 0;
  for (let i = 0; i < points.length; i += 2) {
    const j = (i + 2) % points.length;
    twice += points[i]! * points[j + 1]! - points[j]! * points[i + 1]!;
  }
  return Math.abs(twice) / 2;
}

test("the selector is the complete production scene catalogue", () => {
  assert.deepEqual(
    ADVANCE_PRODUCTION_SCENES.map((scene) => scene.id),
    SCENE_CATALOG.map((definition) => definition.id),
  );
  assert.equal(new Set(ADVANCE_PRODUCTION_SCENES.map((scene) => scene.id)).size,
    SCENE_CATALOG.length);
});

test("the native centre plane uses production solid and liquid samples", () => {
  const definition = advanceProductionSceneDefinition("water-box-dam-break");
  const scene = sceneDocument(definition);
  const dimensions = sceneLatticeDimensions(scene) as [number, number, number];
  const world = fluidSolidWorldForScene(scene);
  const seed = productionSceneSliceSeedById(definition.id);
  validateSliceSceneSeed(seed);

  assert.deepEqual(seed.dimensions, dimensions.slice(0, 2));
  assert.equal(seed.viewport.centerCellZ, Math.floor(dimensions[2] / 2));
  assert.equal(seed.viewport.centerZ, 0);
  assert.equal(seed.viewport.sourceCellCenterZ,
    -0.5 * scene.container.depth_m
      + (seed.viewport.centerCellZ + 0.5) * seed.viewport.sourceCellSize);
  assert.equal(seed.sourceAtlas?.dimensions.join("x"), dimensions.join("x"));
  assert.equal(seed.id, definition.id);
  assert.equal(seed.production?.scene.sceneId, scene.sceneId);

  const [nx, ny] = seed.dimensions;
  for (let sourceY = 0; sourceY < ny; sourceY += 1) for (let x = 0; x < nx; x += 1) {
    const canvasY = ny - 1 - sourceY;
    const i = canvasY * nx + x;
    const solid = sampleSolidWorld(world, [x, sourceY, seed.viewport.centerCellZ]);
    const open = 1 - solid.solidFraction;
    const base = baseInitialLiquidFractionAtCell(scene, x, sourceY,
      seed.viewport.centerCellZ, dimensions);
    const density = open * initialLiquidFractionAtCell(scene, x, sourceY,
      seed.viewport.centerCellZ, dimensions, base);
    assert.equal(seed.capacity[i], Math.fround(open));
    assert.equal(seed.density[i], Math.fround(density));
    assert.equal(seed.materialId[i], solid.materialId);
  }
});

test("the y-down boundary reflects production velocity and gravity once", () => {
  const seed = productionSceneSliceSeedById("coarse-surface-translation");
  assert.deepEqual(seed.gravity, [0, 0]);
  assert.ok(seed.velocityY.every((velocity) =>
    Math.abs(velocity - 0.4) <= 2 * Number.EPSILON + 1e-7));
  assert.ok(seed.velocityX.every((velocity) => velocity === 0));

  const falling = productionSceneSliceSeedById("water-box-dam-break");
  const sourceGravity = falling.production!.scene.fluid.gravity_m_s2;
  assert.deepEqual(falling.gravity, [sourceGravity.x, -sourceGravity.y]);
});

test("frame-zero curved liquid publishes reconstructed PLIC planes", () => {
  const seed = productionSceneSliceSeedById("coarse-first-pool-impact-half");
  const slice = createAdvanceSlice(seed);
  const lattice = createSliceLattice(slice);
  buildSliceLattice(lattice, slice);
  const partial = lattice.cells.filter(cell => cell.open
    && cell.fill > 1e-3 && cell.fill < 1 - 1e-3);
  assert.equal(partial.length, 68, "production sphere sampling fixture changed");
  for (const cell of partial) {
    const plane = latticePlane(lattice, cell);
    assert.ok(plane, `partial cell ${cell.topologyCell} has no frame-zero plane`);
    const polygon = clipUnitSquare(UNIT_SQUARE, plane.nx, plane.ny, plane.offset);
    assert.ok(polygon.length >= 6);
    assert.ok(Math.abs(polygonArea(polygon) - cell.fill) <= 2e-6,
      `cell ${cell.topologyCell} plane area does not reproduce its sampled volume`);
  }

  const reset = resetAdvanceSlice(slice, seed);
  const resetLattice = createSliceLattice(reset);
  buildSliceLattice(resetLattice, reset);
  assert.equal(resetLattice.cells.filter(cell => cell.open
    && cell.fill > 1e-3 && cell.fill < 1 - 1e-3
    && latticePlane(resetLattice, cell)).length, partial.length);
});

test("adaptive steps publish reconstructed planes for each accepted generation", () => {
  const seed = productionSceneSliceSeedById("coarse-first-pool-impact-half");
  const slice = createAdvanceSlice(seed);
  let changedGeneration = false;
  for (let step = 0; step < 3; step += 1) {
    const before = slice.topology.accepted.generation;
    advanceSlice(slice, { pressureIterations: 4 });
    changedGeneration ||= slice.topology.accepted.generation !== before;
    const partial = slice.numericalTopology.cells.filter(cell => {
      const capacity = slice.fields.capacity[cell.id]!;
      const fill = capacity > 1e-8 ? slice.fields.density[cell.id]! / capacity : 0;
      return fill > 1e-3 && fill < 1 - 1e-3;
    });
    assert.ok(partial.length > 0);
    assert.ok(partial.every(cell => Number.isFinite(slice.fields.interfaceOffset[cell.id]!)
      && (slice.fields.interfaceNormal[2 * cell.id] !== 0
        || slice.fields.interfaceNormal[2 * cell.id + 1] !== 0)),
    `frame ${slice.frame} generation ${slice.topology.accepted.generation} published an unresolved partial cell`);
  }
  assert.equal(changedGeneration, true, "fixture must exercise candidate acceptance");
});

test("coarse-first pool support follows motion without bridging disconnected liquid", () => {
  const slice = createAdvanceSlice(
    productionSceneSliceSeedById("coarse-first-pool-impact-half"));
  const activeBrickCount = () => slice.topology.accepted.bricks
    .filter(brick => brick.active !== false).length;
  const dryActive = () => slice.topology.accepted.bricks.filter(brick =>
    brick.active !== false && slice.topology.accepted.cells
      .filter(cell => cell.brickKey === brick.key)
      .every(cell => slice.fields.density[cell.id] === 0));
  const initialDry = dryActive();
  assert.equal(initialDry.length, 0,
    "a stationary interface must use sparse-air rows instead of active dry pages");
  assert.equal(slice.topology.accepted.cells.length, 796,
    "generation zero must contain material pages only");
  assert.equal(slice.topology.accepted.bricks.some(brick =>
    brick.active !== false && brick.coordinate[1] === 2), false,
  "the empty page row between the ball and pool must remain unrepresented");
  const counts = [activeBrickCount()];
  for (let step = 0; step < 3; step += 1) {
    advanceSlice(slice, { pressureIterations: 64 });
    counts.push(activeBrickCount());
    if (step === 0) assert.equal(dryActive().length, 0,
      "the falling ball's downward sweep must not create unswept side-air pages");
  }
  assert.deepEqual(counts, [28, 28, 28, 28],
    "only the next-step swept receiver corridor may expand the page domain");
});

test("omitted slice growth budget reserves the bounded physical page arena", () => {
  const seed = productionSceneSliceSeedById("sparse-cm12-ladder-symmetric-2d");
  const automatic = createAdvanceSlice(seed);
  assert.equal(automatic.maximumSliceLeaves, 8,
    "capacity covers the 4x2 tank without activating its dry pages");
  assert.equal(automatic.topology.accepted.bricks.filter(brick => brick.active !== false).length, 1);

  const noGrowth = createAdvanceSlice({ ...seed, production: { ...seed.production!,
    options: { ...seed.production!.options, topologyPageBudget: 0 } } });
  assert.equal(noGrowth.maximumSliceLeaves, 4,
    "an explicit zero budget keeps the authored no-growth arena");
});

test("a 2.5-cell projected translation admits its receiver before microstep zero", () => {
  const source = productionSceneSliceSeedById("sparse-cm12-ladder-symmetric-2d");
  // The sparse-air face preparation averages the one-sided fixture row with
  // vacuum. Prescribe five cells so the final projected field is exactly the
  // 2.5-cell whole-frame oracle case.
  const speed = 5 * source.viewport.sourceCellSize / source.dt;
  const seed = { ...source, gravity: [0, 0] as const,
    boundary: { ...source.boundary, xMin: "open" as const, xMax: "open" as const },
    velocityX: new Float32Array(source.velocityX.length).fill(speed),
    velocityY: new Float32Array(source.velocityY.length) };
  const slice = createAdvanceSlice(seed);
  let firstMicrostepPages: readonly string[] = [];
  let projectedTravelFine = 0;
  advanceSlice(slice, { pressureIterations: 0, onStageComplete: (stage, state) => {
    if(stage === "velocity-projection")projectedTravelFine = state.fields.faceVelocity
      .reduce((maximum, velocity) => Math.max(maximum, Math.abs(velocity)), 0) * state.scene.dt;
  }, onTransportMicrostep: (_microstep, state) => {
    if(firstMicrostepPages.length)return;
    firstMicrostepPages = state.topology.accepted.bricks
      .filter(brick => brick.active !== false).map(brick => brick.coordinate.join(","));
  } });
  assert.equal(slice.fault, null);
  assert.ok(Math.abs(projectedTravelFine - 2.5) <= 1e-6,
    `fixture projected travel changed to ${projectedTravelFine}`);
  assert.ok(slice.microsteps >= 5);
  assert.deepEqual(firstMicrostepPages, ["0,0", "1,0"],
    "the exact +X page path must be ready without a transverse halo");
});

test("projected receivers are accepted before transport and the ladder reaches the RHS", () => {
  const slice = createAdvanceSlice(
    productionSceneSliceSeedById("sparse-cm12-ladder-symmetric-2d"));
  const volumeFine2 = () => slice.topology.accepted.cells.reduce((sum, cell) =>
    sum + slice.fields.density[cell.id]! * cell.volumeFineCells, 0);
  const initialVolumeFine2 = volumeFine2();
  assert.equal(initialVolumeFine2, 64);
  let maximumAbsoluteDriftFine2 = 0;
  let firstMicrostepPages: readonly string[] = [];
  for (let step = 0; step < 90; step += 1) {
    advanceSlice(slice, { pressureIterations: 64,
      onTransportMicrostep: step === 0 ? (_microstep, state) => {
        firstMicrostepPages = state.topology.accepted.bricks
          .filter(brick => brick.active !== false)
          .map(brick => brick.coordinate.join(","));
      } : undefined });
    assert.equal(slice.fault, null, `frame ${slice.frame} faulted`);
    maximumAbsoluteDriftFine2 = Math.max(maximumAbsoluteDriftFine2,
      Math.abs(volumeFine2() - initialVolumeFine2));
  }
  assert.deepEqual(firstMicrostepPages, ["0,0", "1,0"],
    "the projected +X receiver, and no transverse halo, must exist before microstep 0");
  assert.ok(maximumAbsoluteDriftFine2 <= 2e-5,
    `closed-tank raw area drifted by ${maximumAbsoluteDriftFine2}`);
  const rhsVolumeFine2 = slice.topology.accepted.cells.reduce((sum, cell) =>
    sum + (cell.maximumFine[0] === slice.nx
      ? slice.fields.density[cell.id]! * cell.volumeFineCells : 0), 0);
  assert.ok(rhsVolumeFine2 > 1,
    `the travelling body did not reach the tank RHS: ${rhsVolumeFine2}`);
  const runtimePages = slice.presentation.accepted.pages
    .filter(page => page.sourceAtlasBrick === -1);
  assert.ok(runtimePages.length > 0, "the run must publish runtime-created pages");
  for(const page of slice.presentation.accepted.pages){
    if(page.sourceAtlasBrick<0)continue;
    assert.equal(slice.scene.sourceAtlas!.bricks[page.sourceAtlasBrick]!.key,page.sourceBrickKey,
      "an authored page must retain exact source-atlas provenance");
  }
});

test("native staggered apertures preserve the production shell and face samples", () => {
  const seed = productionSceneSliceSeedById("water-box-dam-break");
  const [nx, ny] = seed.dimensions;
  assert.equal(seed.apertureX?.length, (nx + 1) * ny);
  assert.equal(seed.apertureY?.length, nx * (ny + 1));
  assert.equal(seed.solidVelocityX?.length, (nx + 1) * ny);
  assert.equal(seed.solidVelocityY?.length, nx * (ny + 1));
  for (let y = 0; y < ny; y += 1) {
    assert.equal(seed.apertureX![y * (nx + 1)], 0);
    assert.equal(seed.apertureX![y * (nx + 1) + nx], 0);
  }
  // Canvas face ny is source y=0: the authored floor is closed.
  for (let x = 0; x < nx; x += 1) assert.equal(seed.apertureY![ny * nx + x], 0);
  // This scene's authored top boundary is closed.
  assert.equal(seed.boundary.yMax, "closed");
  assert.ok(seed.apertureY!.subarray(0, nx).every(value => value === 0));

  const openTop = productionSceneSliceSeedById("water-box-tank-fill");
  assert.equal(openTop.boundary.yMax, "open");
  assert.ok(openTop.apertureY!.subarray(0, openTop.dimensions[0])
    .every(value => value === 1));
});

test("empty and dimensionally reduced production slices stay explicit", () => {
  const dry = productionSceneSliceSeedById("garden-svo-lighting");
  assert.ok(dry.density.every((density) => density === 0));
  assert.ok(dry.limitations?.some((note) => note.includes("disables fluid")));

  const source = productionSceneSliceSeedById("garden-hose");
  assert.equal(source.dynamic?.some((entry) => entry.kind === "source"), true);
  assert.equal(source.dynamic?.find((entry) => entry.kind === "source")?.supported, true);
  assert.ok(source.limitations?.some((note) => note.includes("inflow")));

  const throughPlane = productionSceneSliceSeedById("hero-garden-hose");
  assert.ok(throughPlane.limitations?.some((note) =>
    note.includes("injects no in-plane liquid")));
  const throughPlaneSlice = createAdvanceSlice(throughPlane);
  const reduced = sliceDynamicGeometry({ seed: throughPlane,
    topology: throughPlaneSlice.numericalTopology, time_s: 1, dt_s: throughPlane.dt });
  assert.equal(reduced.requestedSourceAreaFine, 0,
    "a z-dominant production source must not become an invented in-plane jet");
});

test("dynamic rigid and inflow reductions stay aligned to accepted topology", () => {
  const rigidSeed = productionSceneSliceSeedById("rigid-float");
  const rigidSlice = createAdvanceSlice(rigidSeed);
  const body = rigidSeed.production!.scene.rigidBodies[0]!;
  const previous: SliceRigidPose = {
    description: body, position_m: body.position_m, orientation: body.orientation,
    linearVelocity_m_s: body.linearVelocity_m_s,
    angularVelocity_rad_s: body.angularVelocity_rad_s,
  };
  const current: SliceRigidPose = {
    ...previous, position_m: { ...body.position_m, x: body.position_m.x + 0.1 },
  };
  const moved = sliceDynamicGeometry({ seed: rigidSeed,
    topology: rigidSlice.numericalTopology, time_s: 1, dt_s: rigidSeed.dt,
    bodies: [current], previousBodies: [previous] });
  assert.equal(moved.capacity.length, rigidSlice.fields.capacity.length);
  assert.equal(moved.openFraction.length, rigidSlice.numericalTopology.rows.length);
  assert.ok(moved.capacityRate.some(rate => rate !== 0));

  const sourceSeed = productionSceneSliceSeedById("garden-hose");
  const sourceSlice = createAdvanceSlice(sourceSeed);
  const source = sliceDynamicGeometry({ seed: sourceSeed,
    topology: sourceSlice.numericalTopology, time_s: 1, dt_s: sourceSeed.dt,
    density: sourceSlice.fields.density });
  assert.equal(source.sourceRate.length, sourceSlice.fields.density.length);
  assert.equal(source.inflowCoverage.length, sourceSlice.numericalTopology.rows.length);
  assert.ok(source.requestedSourceAreaFine > 0);
  assert.ok(source.sourceRateAreaFine > 0);
  assert.ok(source.inflowCoverage.some(coverage => coverage > 0));

  const plan = planSliceDynamicRemap({ topology: sourceSlice.numericalTopology,
    density: sourceSlice.fields.density,
    capacityBefore: sourceSlice.fields.capacity, capacityAfter: source.capacity,
    capacityRate: source.capacityRate, sourceRate: source.sourceRate,
    dt: sourceSeed.dt, pendingSourceAreaFine: 0,
    requestedSourceAreaRateFine: source.requestedSourceAreaFine });
  assert.deepEqual(plan.density, sourceSlice.fields.density,
    "the GCL stage must preserve rho, including production excess states");
  const commit = commitSliceSourceLedger(plan.ledger, plan.sourceRate, sourceSeed.dt);
  assert.equal(commit.fault, null);
  assert.ok(commit.emittedArea > 0);
  assert.ok(Math.abs(commit.ledger.pending
    - (plan.ledger.pending - commit.emittedArea)) < 1e-3);
});

test("production rigid descriptors advance and feed old/new slice geometry", () => {
  const seed = productionSceneSliceSeedById("rigid-float");
  const slice = createAdvanceSlice(seed);
  const initial = createSliceRigidAuthority(seed);
  assert.equal(initial.current.length, seed.production!.scene.rigidBodies.length);
  assert.deepEqual(initial.current.map(body => body.description.shape),
    seed.production!.scene.rigidBodies.map(body => body.shape));
  const advanced = advanceSliceRigidAuthority(initial, seed, seed.dt);
  assert.notDeepEqual(advanced.current.map(body => body.position_m),
    advanced.previous.map(body => body.position_m));
  const geometry = sliceDynamicGeometry({ seed, topology: slice.numericalTopology,
    time_s: seed.dt, dt_s: seed.dt, bodies: sliceRigidPoses(advanced.current),
    previousBodies: sliceRigidPoses(advanced.previous), density: slice.fields.density });
  assert.deepEqual(geometry.openFraction,
    Float32Array.from(geometry.openFractionAfter,
      (value, index) => Math.fround(0.5 * Math.fround(
        geometry.openFractionBefore[index]! + value))));
  const coupled = sliceRigidCouplingLoads(seed, slice.numericalTopology,
    slice.fields, advanced.current);
  assert.equal(coupled.loads.size, advanced.current.length);
  assert.equal(coupled.receipts.length, advanced.current.length);
  assert.ok(coupled.receipts.every(receipt => receipt.displacedVolume_m3 >= 0));
});
