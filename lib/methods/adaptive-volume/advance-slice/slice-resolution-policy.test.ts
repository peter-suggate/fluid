import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import { SPARSE_CM12_ACTIVITY_POLICY,
  type SparseCM12ActivityPolicy } from "../features/adaptivity/policy";
import { compileSliceTopology, type SliceTopologyBrick } from "./slice-topology";
import { initializeSliceResolutionPolicy, planSliceProjectedTransportSupport, planSliceResolution,
  SliceActivityReason, SLICE_RESOLUTION_POLICY_SOURCE,
  type SliceResolutionPolicyState } from "./slice-resolution-policy";

const closed = { negativeX: "closed", positiveX: "closed",
  negativeY: "closed", positiveY: "closed" } as const;

function setup(bricks: readonly SliceTopologyBrick[], dimensions: readonly [number, number]) {
  const topology = compileSliceTopology(bricks, dimensions, 1, 0.5, closed);
  return { topology, fields: { density: Float32Array.from(topology.cells, cell => cell.density),
    capacity: new Float32Array(topology.cells.length).fill(1),
    cellVelocity: new Float32Array(2 * topology.cells.length) } };
}

function policy(overrides: Partial<SparseCM12ActivityPolicy> = {}): SparseCM12ActivityPolicy {
  return { ...SPARSE_CM12_ACTIVITY_POLICY, topologyCadenceSteps: 1,
    prepareBricksPerFrame: 64, ...overrides };
}

test("policy source fingerprints pin the resident activity ABI", () => {
  const resident = readFileSync(new URL("../webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url));
  const policySource = readFileSync(new URL("../features/adaptivity/policy.ts", import.meta.url));
  const residentText = resident.toString();
  const activityHash = createHash("sha256");
  for (const name of SLICE_RESOLUTION_POLICY_SOURCE.residentFunctions) {
    const begin = residentText.indexOf(`fn ${name}`);
    const end = residentText.indexOf("\nfn ", begin + 3);
    assert.ok(begin >= 0);
    activityHash.update(residentText.slice(begin, end < 0 ? residentText.length : end));
  }
  activityHash.update(readFileSync(new URL(
    "../features/adaptivity/coarse-first.wgsl.ts", import.meta.url)));
  assert.equal(activityHash.digest("hex"),
    SLICE_RESOLUTION_POLICY_SOURCE.activitySourcesSha256);
  assert.equal(createHash("sha256").update(policySource).digest("hex"),
    SLICE_RESOLUTION_POLICY_SOURCE.policySha256);
  for (const name of SLICE_RESOLUTION_POLICY_SOURCE.residentFunctions) {
    assert.match(residentText, new RegExp(`fn ${name}\\b`));
  }
});

test("represented planar interface preserves surface and density reason bits", () => {
  const density = new Float32Array(64);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 4; x++) density[x + 8 * y] = 1;
  const { topology, fields } = setup([{ id: 0, key: 0, coordinate: [0, 0],
    resolution: 8, density }], [8, 8]);
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy() } });
  const record = decision.receipt.bricks[0]!;
  assert.ok(record.reasons & SliceActivityReason.Surface);
  assert.ok(record.reasons & SliceActivityReason.DensitySurface);
  // No accepted output-space proof exists, so production keeps the surface rung.
  assert.equal(record.scheduledResolution, 8);
});

test("remote impact prediction cannot refine a calm coarse receiver", () => {
  const sourceDensity = new Float32Array(64);
  for (let y = 0; y < 4; y++) for (let x = 0; x < 8; x++) {
    sourceDensity[x + 8 * y] = 1;
  }
  const { topology, fields } = setup([
    { id: 0, key: 0, coordinate: [0, 0], resolution: 1,
      density: new Float32Array([0.5]) },
    { id: 1, key: 1, coordinate: [0, 2], resolution: 8, density: sourceDensity },
  ], [8, 24]);
  for (const cell of topology.cells) if (cell.brickKey === 1) {
    fields.cellVelocity[2 * cell.id + 1] = -20;
  }
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 30, cellSize: 0.05,
    options: { policy: policy(), allocateMissingPages: false } });
  const receiver = decision.receipt.bricks.find(brick => brick.brickKey === 0)!;
  assert.equal(receiver.reasons & SliceActivityReason.Surface,
    SliceActivityReason.Surface, "the receiver must be a represented surface");
  assert.equal(receiver.scheduledResolution, 1,
    "approaching remote liquid may retain established detail but cannot create it");
});

test("swept material support activates an allocated dry frontier leaf", () => {
  const wet = new Float32Array(64);
  for (let y = 0; y < 8; y++) wet[7 + 8 * y] = 1;
  const { topology, fields } = setup([
    { id: 0, key: 0, coordinate: [0, 0], resolution: 8, density: wet },
    { id: 1, key: 1, coordinate: [1, 0], resolution: 8, active: false },
  ], [16, 8]);
  for (const cell of topology.cells) if (cell.brickKey === 0) {
    fields.cellVelocity[2 * cell.id] = 1;
  }
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy() } });
  const receiver = decision.receipt.bricks.find(brick => brick.brickKey === 1)!;
  assert.equal(receiver.acceptedActive, false);
  assert.equal(receiver.candidateActive, true);
  assert.equal(receiver.scheduledResolution, 8);
});

test("projected transport support changes only its receiver and required grading", () => {
  const wet = new Float32Array(64);
  for (let y = 0; y < 8; y++) wet[7 + 8 * y] = 1;
  const {topology,fields}=setup([
    {id:0,key:0,coordinate:[0,0],resolution:8,density:wet},
    {id:1,key:1,coordinate:[1,0],resolution:4,active:false},
    {id:3,key:3,coordinate:[3,0],resolution:1,density:new Float32Array([.5])},
  ],[32,8]);
  for(const cell of topology.cells)if(cell.brickKey===0)
    fields.cellVelocity[2*cell.id]=1;
  const decision=planSliceProjectedTransportSupport({topology,fields,
    dt:1/60,cellSize:.05,policy:policy(),maximumLeaves:4,maximumCells:256});
  assert.equal(decision.faultBits,0);
  assert.deepEqual(decision.candidateBricks.map(brick=>
    [brick.key,brick.resolution,brick.active!==false]),[
    [0,8,true],[1,4,true],[3,1,true],
  ]);
  assert.deepEqual([...decision.demandedBrickKeys],[1]);
});

test("swept material support does not grow an unswept face halo", () => {
  const wet = new Float32Array(64);
  for (let y = 0; y < 8; y++) wet[8 * y] = 1;
  const { topology, fields } = setup([
    { id: 0, key: 0, coordinate: [0, 0], resolution: 8, active: false },
    { id: 1, key: 1, coordinate: [1, 0], resolution: 8, active: false },
    { id: 2, key: 2, coordinate: [2, 0], resolution: 8, active: false },
    { id: 3, key: 3, coordinate: [3, 0], resolution: 8, density: wet },
  ], [32, 8]);
  for (const cell of topology.cells) if (cell.brickKey === 3) {
    fields.cellVelocity[2 * cell.id] = -1;
  }
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy() } });
  const far = decision.receipt.bricks.find(brick => brick.brickKey === 0)!;
  const receiver = decision.receipt.bricks.find(brick => brick.brickKey === 2)!;
  const backing = decision.receipt.bricks.find(brick => brick.brickKey === 1)!;
  assert.equal(receiver.candidateActive, true);
  assert.equal(receiver.scheduledResolution, 8);
  assert.equal(backing.candidateActive, false,
    "the upstream donor already supplies the receiver's characteristic support");
  assert.equal(far.candidateActive, false,
    "an unswept page must remain outside the support domain");
});

test("accepted diagonal material support does not retire and reactivate every frame", () => {
  const wet = new Float32Array(64); wet[7 + 8 * 7] = 1;
  const { topology, fields } = setup([
    { id: 0, key: 0, coordinate: [0, 0], resolution: 8, density: wet },
    { id: 1, key: 1, coordinate: [1, 1], resolution: 1,
      density: new Float32Array(1) },
  ], [16, 16]);
  for (const cell of topology.cells) if (cell.brickKey === 0) {
    fields.cellVelocity[2 * cell.id] = 1;
    fields.cellVelocity[2 * cell.id + 1] = 1;
  }
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy(), allocateMissingPages: false } });
  const receiver = decision.receipt.bricks.find(brick => brick.brickKey === 1)!;
  assert.equal(receiver.candidateActive, true,
    "a page selected by an adjacent swept-support bit must remain accepted");
  assert.equal(receiver.scheduledResolution, 1,
    "pure diagonal support must remain at the coarsest 2:1-compatible rung");
});

test("swept support allocates a missing in-domain WDR page deterministically", () => {
  const wet = new Float32Array(64);
  for (let y = 0; y < 8; y++) wet[7 + 8 * y] = 1;
  const { topology, fields } = setup([
    { id: 4, key: 9, coordinate: [0, 0], resolution: 8, density: wet },
  ], [16, 8]);
  for (const cell of topology.cells) fields.cellVelocity[2 * cell.id] = 1;
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy() } });
  assert.equal(decision.receipt.allocatedBrickCount, 1);
  assert.deepEqual(decision.candidateBricks.map(brick => [brick.id, brick.key,
    brick.coordinate, brick.resolution, brick.active]), [
    [4, 9, [0, 0], 8, true], [5, 10, [1, 0], 8, true],
  ]);
});

test("missing support claims the production WDR free stack before high-water growth", () => {
  const wet = new Float32Array(64);
  for (let y = 0; y < 8; y++) wet[7 + 8 * y] = 1;
  const { topology, fields } = setup([
    { id: 4, key: 9, coordinate: [0, 0], resolution: 8, density: wet },
    { id: 2, key: 10, coordinate: [2, 0], resolution: 8, active: false },
    { id: 3, key: 11, coordinate: [3, 0], resolution: 8, active: false },
  ], [32, 8]);
  for (const cell of topology.cells) if (cell.brickKey === 9) {
    fields.cellVelocity[2 * cell.id] = 1;
  }
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy(), freeLeafIds: [2, 3] } });
  assert.deepEqual(Array.from(decision.receipt.claimedLeafIds), [3]);
  assert.deepEqual(decision.candidateBricks.map(brick => [brick.id, brick.coordinate]), [
    [4, [0, 0]], [3, [1, 0]],
  ]);
});

test("stationary boundary liquid does not activate a dry page", () => {
  const wet = new Float32Array(64);
  for (let y = 0; y < 8; y++) wet[7 + 8 * y] = 1;
  const { topology, fields } = setup([
    { id: 0, key: 0, coordinate: [0, 0], resolution: 8, density: wet },
    { id: 1, key: 1, coordinate: [1, 0], resolution: 4, active: false },
  ], [16, 8]);
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy(), allocateMissingPages: false } });
  const receiver = decision.receipt.bricks.find(brick => brick.brickKey === 1)!;
  assert.equal(receiver.candidateActive, false,
    "static interface evidence may refine a represented page but must not expand residency");
});

test("sub-roundoff boundary flux does not allocate a page", () => {
  const wet = new Float32Array(64); wet[7] = 1;
  const { topology, fields } = setup([
    { id: 0, key: 0, coordinate: [0, 0], resolution: 8, density: wet },
    { id: 1, key: 1, coordinate: [1, 0], resolution: 8, active: false },
  ], [16, 8]);
  const donor = topology.cells.find(cell => fields.density[cell.id] !== 0)!;
  fields.cellVelocity[2 * donor.id] = 0.25 * 9.5367431640625e-7 / (1 / 60);
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy(), allocateMissingPages: false } });
  assert.equal(decision.receipt.bricks.find(brick => brick.brickKey === 1)!.candidateActive,
    false, "transport accepts this displacement inside its f32 volume margin");
});

test("boundary flux above transport roundoff still activates its receiver", () => {
  const wet = new Float32Array(64); wet[7] = 1;
  const { topology, fields } = setup([
    { id: 0, key: 0, coordinate: [0, 0], resolution: 8, density: wet },
    { id: 1, key: 1, coordinate: [1, 0], resolution: 8, active: false },
  ], [16, 8]);
  const donor = topology.cells.find(cell => fields.density[cell.id] !== 0)!;
  fields.cellVelocity[2 * donor.id] = 2 * 9.5367431640625e-7 / (1 / 60);
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy(), allocateMissingPages: false } });
  assert.equal(decision.receipt.bricks.find(brick => brick.brickKey === 1)!.candidateActive,
    true, "a physically represented slow flux must reserve its receiver");
});

test("a body travelling multiple cells in one step reserves the crossed page", () => {
  const { topology, fields } = setup([
    { id: 0, key: 0, coordinate: [0, 0], resolution: 8,
      density: new Float32Array(64).fill(1) },
    { id: 1, key: 1, coordinate: [1, 0], resolution: 8, active: false },
  ], [16, 8]);
  // 150 finest cells/s for 1/60 s translates the block by 2.5 cells.
  for (const cell of topology.cells) if (cell.brickKey === 0) {
    fields.cellVelocity[2 * cell.id] = 150;
  }
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy(), allocateMissingPages: false } });
  assert.deepEqual(decision.candidateBricks.filter(brick => brick.active !== false)
    .map(brick => brick.coordinate), [[0, 0], [1, 0]]);
});

test("diagonal motion activates only its swept page corridor", () => {
  const bricks: SliceTopologyBrick[] = [];
  for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) {
    const source = x === 1 && y === 1;
    const density = new Float32Array(64);
    if (source) density[7 + 8 * 7] = 1;
    bricks.push({ id: x + 3 * y, key: x + 3 * y, coordinate: [x, y],
      resolution: 8, density, active: source });
  }
  const { topology, fields } = setup(bricks, [24, 24]);
  for (const cell of topology.cells) if (fields.density[cell.id] !== 0) {
    fields.cellVelocity[2 * cell.id] = 1;
    fields.cellVelocity[2 * cell.id + 1] = 1;
  }
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy(), allocateMissingPages: false } });
  const activeCoordinates = decision.candidateBricks.filter(brick => brick.active !== false)
    .map(brick => brick.coordinate.join(",")).sort();
  assert.deepEqual(activeCoordinates, ["1,1", "1,2", "2,1", "2,2"],
    "only pages intersected by the Cartesian swept corridor become resident");
});

test("retirement requires exact zero while a sub-residency amount remains resident", () => {
  const run = (value: number) => {
    const density = new Float32Array(64); density[0] = value;
    const { topology, fields } = setup([{ id: 0, key: 0, coordinate: [0, 0],
      resolution: 8, density }], [8, 8]);
    return planSliceResolution({ topology, fields,
      previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
      options: { policy: policy() } }).receipt.bricks[0]!;
  };
  assert.equal(run(0).candidateActive, false);
  assert.equal(run(Math.fround(1e-30)).candidateActive, true);
});

test("refine-only closure raises the coarse side to physical 2:1", () => {
  const { topology, fields } = setup([
    { id: 0, key: 0, coordinate: [0, 0], resolution: 1,
      density: new Float32Array([1]) },
    { id: 1, key: 1, coordinate: [1, 0], resolution: 2,
      density: new Float32Array(4).fill(1) },
  ], [16, 8]);
  for (const cell of topology.cells) if (cell.brickKey === 1) {
    fields.cellVelocity[2 * cell.id] = 100;
    fields.cellVelocity[2 * cell.id + 1] = 100;
  }
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1, cellSize: 1,
    options: { policy: policy({ energyThreshold: 0.01 }) } });
  const coarse = decision.receipt.bricks.find(brick => brick.brickKey === 0)!;
  assert.equal(coarse.scheduledResolution, 4);
  assert.equal(decision.receipt.faultBits, 0);
});

test("ordinary demotion admission skips an over-budget pair and keeps rotating", () => {
  const bricks = [0, 1, 2].map(key => ({ id: key, key, coordinate: [key, 0] as const,
    resolution: 8 as const, density: new Float32Array(64).fill(1) }));
  const { topology, fields } = setup(bricks, [24, 8]);
  let state: SliceResolutionPolicyState = initializeSliceResolutionPolicy(topology);
  // Seed quiet persistence without letting the first transaction alter authority.
  state = { ...state, history: new Map([...state.history].map(([key, entry]) => [key,
    { ...entry, quietEpochs: 8 }])) };
  const decision = planSliceResolution({ topology, fields, previous: state,
    dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy({ coarseFirst: false, prepareBricksPerFrame: 1,
      demoteEpochs: 1 }) } });
  assert.equal(decision.receipt.demotedBrickCount, 1);
  assert.equal(decision.receipt.deferredDemotionCount, 2);
  assert.equal(decision.receipt.bricks.find(brick => brick.brickKey === 1)!.scheduledResolution, 4);
});

test("a symmetric demotion request spends its budget on a complete reflection pair", () => {
  const bricks = [0, 1, 2, 3].map(key => ({ id: key, key,
    coordinate: [key, 0] as const, resolution: 8 as const,
    density: new Float32Array(64).fill(1) }));
  const { topology, fields } = setup(bricks, [32, 8]);
  let state: SliceResolutionPolicyState = initializeSliceResolutionPolicy(topology);
  state = { ...state, history: new Map([...state.history].map(([key, entry]) => [key,
    { ...entry, quietEpochs: 8 }])) };
  const decision = planSliceResolution({ topology, fields, previous: state,
    dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy({ coarseFirst: false, prepareBricksPerFrame: 2,
      demoteEpochs: 1 }) } });
  assert.deepEqual(decision.receipt.bricks
    .filter(brick => brick.scheduledResolution === 4).map(brick => brick.brickKey), [0, 3]);
});

test("an over-budget reflection pair accumulates credit and is eventually admitted", () => {
  const bricks = [0, 1].map(key => ({ id:key,key,coordinate:[key,0] as const,
    resolution:8 as const,density:new Float32Array(64).fill(1) }));
  const {topology,fields}=setup(bricks,[16,8]);
  let state:SliceResolutionPolicyState=initializeSliceResolutionPolicy(topology);
  state={...state,history:new Map([...state.history].map(([key,entry])=>[key,
    {...entry,quietEpochs:8}]))};
  const input={topology,fields,dt:1/60,cellSize:0.05,
    options:{policy:policy({coarseFirst:false,prepareBricksPerFrame:1,demoteEpochs:1})}};
  const first=planSliceResolution({...input,previous:state});
  assert.equal(first.receipt.demotedBrickCount,0);
  assert.equal(first.state.schedulingCredits,1);
  const second=planSliceResolution({...input,previous:first.state});
  assert.equal(second.receipt.demotedBrickCount,2);
  assert.equal(second.state.schedulingCredits,0);
});

test("hard minimum-cell region coarsens its outward grading halo", () => {
  const bricks = [0, 1].map(key => ({ id: key, key, coordinate: [key, 0] as const,
    resolution: 8 as const, density: new Float32Array(64).fill(1) }));
  const { topology, fields } = setup(bricks, [16, 8]);
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy(), refinementRegions: [{ minimumFine: [0, 0],
      maximumFine: [8, 8], minimumCellWidth: 8 }] } });
  assert.deepEqual(decision.receipt.bricks.map(brick => brick.scheduledResolution), [1, 2]);
  assert.equal(decision.receipt.faultBits, 0);
});

test("admission capacity fault leaves accepted topology authoritative", () => {
  const wet = new Float32Array(64); wet[7] = 1;
  const { topology, fields } = setup([
    { id: 0, key: 0, coordinate: [0, 0], resolution: 8, density: wet },
  ], [16, 8]);
  for (const cell of topology.cells) fields.cellVelocity[2 * cell.id] = 1;
  const decision = planSliceResolution({ topology, fields,
    previous: initializeSliceResolutionPolicy(topology), dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy(), maximumLeaves: 1 } });
  assert.ok(decision.receipt.faultBits !== 0);
  assert.equal(decision.candidateBricks.length, 1);
  assert.equal(decision.receipt.candidateGeneration, topology.generation);
});

test("an allocated retiring neighbour still constrains planned-rung closure", () => {
  const left = new Float32Array(16); left[0] = Math.fround(1e-30);
  const { topology, fields } = setup([
    { id: 0, key: 0, coordinate: [0, 0], resolution: 4, density: left },
    { id: 1, key: 1, coordinate: [1, 0], resolution: 8,
      density: new Float32Array(64) },
  ], [16, 8]);
  let state = initializeSliceResolutionPolicy(topology);
  state = { ...state, history: new Map([...state.history].map(([key, entry]) => [key,
    { ...entry, quietEpochs: 8 }])) };
  const decision = planSliceResolution({ topology, fields, previous: state,
    dt: 1 / 60, cellSize: 0.05,
    options: { policy: policy({ coarseFirst: false, demoteEpochs: 1 }),
      frozenBrickKeys: new Set([1]) } });
  const retained = decision.receipt.bricks.find(brick => brick.brickKey === 0)!;
  const retiring = decision.receipt.bricks.find(brick => brick.brickKey === 1)!;
  assert.equal(retiring.candidateActive, false);
  assert.equal(retained.scheduledResolution, 4);
  assert.equal(decision.receipt.faultBits, 0);
});
