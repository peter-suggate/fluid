import assert from "node:assert/strict";
import test from "node:test";
import { proceduralSolidField, validateProceduralSolid } from "../lib/core/procedural-solid";
import "../lib/methods";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { parseScene, serializeScene, validateScene, type SceneDescription } from "../lib/core/model";
import { NBFLIP_LETTERS_GRID, NBFLIP_METHOD_PROFILE } from "../lib/core/nbflip-paper-scenes";
import {
  NBFLIP_SLIDE_END_RAD, NBFLIP_SLIDE_GRID, NBFLIP_SLIDE_LONG_ARCS, NBFLIP_SLIDE_LONG_BENDS, NBFLIP_SLIDE_LONG_GRID, NBFLIP_SLIDE_LONG_LENGTH,
  NBFLIP_SLIDE_LONG_SCENE_ID, NBFLIP_SLIDE_SCENE_ID, nbflipSlideCentreline, nbflipSlideLongCentreline, nbflipSlideSolid, nbflipSlideLongSolid,
} from "../lib/core/nbflip-water-slide-scene";

/** A slide's own solids, and the top of the lowest of them in each column of cells (the hood's lid hangs over it). */
function slideColumns(scene: SceneDescription, grid: readonly [number, number, number]) {
  const [nx, ny, nz] = grid;
  const patches = scene.solidVoxels.filter(patch => patch.minimum.every((cell, axis) => cell >= 0 && patch.maximumExclusive[axis]! <= grid[axis]!));
  if (scene.solidBoundary) {
    const field = scene.sceneId === NBFLIP_SLIDE_LONG_SCENE_ID ? proceduralSolidField(scene) : undefined;
    // Boundary bricks omit wholly solid and wholly empty regions. Preserve
    // that distinction when a vertical scan crosses an eight-cell boundary.
    const interiors = new Set(patches.filter(p => !p.kind || p.kind === "box").map(p => p.minimum.join()));
    const sample = (x: number, y: number, z: number) => field
      ? field.sample([x,y,z])?.[3] ?? (interiors.has([x,y,z].map(v => Math.floor(v / 8) * 8).join()) ? -4 : 4)
      : nbflipSlideSolid([x,y,z]);
    const floorAt=(x:number,z:number)=>{
      let inside=false;
      for(let y=.125;y<ny;y+=.25){
        const value = sample(x,y,z);
        const solid=value<0;
        if(inside&&!solid){let low=y-.25,high=y;for(let k=0;k<24;k++){const m=(low+high)/2;if(sample(x,m,z)<0)low=m;else high=m;}return Math.round((low+high)/2*1e6)/1e6;}
        inside=solid;
      }
      return 0;
    };
    const top=Float64Array.from({length:nx*nz},(_,i)=>floorAt(i%nx+.5,Math.floor(i/nx)+.5));
    return {patches,top,topAt:floorAt};
  }
  const base = new Int32Array(nx * nz).fill(ny), top = new Float64Array(nx * nz);
  for (const { minimum, maximumExclusive } of patches) {
    for (let z = minimum[2]; z < maximumExclusive[2]; z++) for (let x = minimum[0]; x < maximumExclusive[0]; x++) {
      if (minimum[1] < base[x + nx * z]!) { base[x + nx * z] = minimum[1]; top[x + nx * z] = maximumExclusive[1]; }
    }
  }
  return { patches, top, topAt: (x: number, z: number) => top[Math.floor(x) + nx * Math.floor(z)]! };
}
/** What a scene's source emits, in cells. */
function emittedCells(scene: SceneDescription) {
  const inflow = scene.fluid.inflow!;
  return Math.PI * inflow.radius_m ** 2 * Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z)
    * (inflow.end_s - inflow.start_s) / scene.voxelDomain.finestCellSize_m ** 3;
}
function assertEmptyClosedNbflipBox(id: string, grid: readonly [number, number, number]) {
  const definition = getSceneDefinition(id), scene = sceneDocument(definition), h = scene.voxelDomain.finestCellSize_m;
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(parseScene(serializeScene(scene)).fluid, scene.fluid);
  assert.deepEqual(parseScene(serializeScene(scene)).solidVoxels, scene.solidVoxels);
  assert.deepEqual(definition.methodProfile, NBFLIP_METHOD_PROFILE);
  assert.deepEqual([scene.container.width_m, scene.container.height_m, scene.container.depth_m].map(length => Math.round(length / h)), grid);
  assert.ok(grid.every(cells => cells % 32 === 0));
  assert.ok(grid[0] * grid[1] * grid[2] <= NBFLIP_LETTERS_GRID[0] * NBFLIP_LETTERS_GRID[1] * NBFLIP_LETTERS_GRID[2]);
  assert.equal(scene.container.top, "closed");
  assert.equal(scene.container.fluidWallMode, "free-slip");
  assert.equal(scene.container.fillFraction, 0);
  assert.equal(scene.fluid.initialLiquidVolumes, undefined);
  assert.ok(scene.fluid.inflow!.end_s < scene.duration_s);
}

const scene = sceneDocument(getSceneDefinition(NBFLIP_SLIDE_SCENE_ID));
const [nx, ny] = NBFLIP_SLIDE_GRID, h = scene.voxelDomain.finestCellSize_m;
const { patches: drum, top, topAt } = slideColumns(scene, NBFLIP_SLIDE_GRID);
/** The centreline every quarter cell or finer, from the source's outlet to the channel's end. */
function* centreline() {
  const inflow = scene.fluid.inflow!, x = inflow.center_m.x / h + nx / 2;
  let angle = 0;
  while (nbflipSlideCentreline(angle)[0] < x) angle += 1e-3;
  for (; angle <= NBFLIP_SLIDE_END_RAD; angle += 2e-3) yield [angle, ...nbflipSlideCentreline(angle)] as const;
}

test("the water slide is a closed NB-FLIP box of whole 32-cell pages that starts empty", () => {
  assertEmptyClosedNbflipBox(NBFLIP_SLIDE_SCENE_ID, NBFLIP_SLIDE_GRID);
});

test("the water slide stands clear of the box it is solved in", () => {
  // Block payload bounds include empty padding; test occupied geometry.
  for(let z=.5;z<NBFLIP_SLIDE_GRID[2];z++)for(let x=.5;x<nx;x++){
    if(x<4||x>nx-4||z<4||z>NBFLIP_SLIDE_GRID[2]-4) assert.equal(topAt(x,z),0,`${x},${z}`);
  }
  assert.ok(Math.max(...top)<ny-4);
});

test("the water slide's channel only falls, between walls, to a hood that opens on the pool", () => {
  const samples = [...centreline()], open: number[] = [];
  let floor = Infinity;
  for (const [angle, x, z] of samples) {
    assert.ok(topAt(x, z) <= floor, `the floor rises at ${angle}`);
    floor = topAt(x, z);
    // Eight cells either side of the centreline is the middle of a wall.
    const outer = topAt(x + 8 * Math.sin(angle), z - 8 * Math.cos(angle)), inner = topAt(x - 8 * Math.sin(angle), z + 8 * Math.cos(angle));
    assert.ok(outer >= floor + 14, `outer wall ${outer - floor} at ${angle}`);
    if (inner < floor + 6) open.push(angle);
  }
  assert.ok(topAt(samples[0]![1], samples[0]![2]) - floor >= 24, "falls at least 24 cells");
  // The inner wall is missing only where the hood spills: one stretch, at the end.
  assert.ok(open.length > 0 && open[open.length - 1]! > NBFLIP_SLIDE_END_RAD - 4e-3);
  assert.ok(open[open.length - 1]! - open[0]! < (open.length + 1) * 2e-3 && open[0]! > NBFLIP_SLIDE_END_RAD - 1);
});

test("the water slide's pool holds everything the source emits below the hood's sill", () => {
  const [x, z] = nbflipSlideCentreline(NBFLIP_SLIDE_END_RAD);
  const pit = top.reduce((lowest, cells) => cells > 0 && cells < lowest ? cells : lowest, ny);
  const capacity = top.filter(cells => cells === pit).length * (topAt(x - 0.5, z) - pit);
  assert.ok(capacity > 1.2 * emittedCells(scene), `${capacity} cells for ${emittedCells(scene)}`);
});

const long = sceneDocument(getSceneDefinition(NBFLIP_SLIDE_LONG_SCENE_ID));
const { patches: flume, top: longTop, topAt: longTopAt } = slideColumns(long, NBFLIP_SLIDE_LONG_GRID);
/** The long slide's centreline every quarter cell: distance along it, x, z, heading. */
function* longCentreline() {
  for (let distance = 1; distance <= NBFLIP_SLIDE_LONG_LENGTH; distance += 0.25) yield [distance, ...nbflipSlideLongCentreline(distance)] as const;
}
/** The top of the column `across` cells off the long centreline, on the side an arc of that turn curves toward. */
const longBeside = (x: number, z: number, heading: number, across: number) => longTopAt(x - across * Math.sin(heading), z + across * Math.cos(heading));

test("the long water slide is a closed NB-FLIP box of Letters' cell count that starts empty", () => {
  assertEmptyClosedNbflipBox(NBFLIP_SLIDE_LONG_SCENE_ID, NBFLIP_SLIDE_LONG_GRID);
  assert.equal(NBFLIP_SLIDE_LONG_GRID[0] * NBFLIP_SLIDE_LONG_GRID[1] * NBFLIP_SLIDE_LONG_GRID[2], NBFLIP_LETTERS_GRID[0] * NBFLIP_LETTERS_GRID[1] * NBFLIP_LETTERS_GRID[2]);
});

test("the long water slide stands clear of the box it is solved in", () => {
  // Baked block bounds include empty padding; inspect the occupied surface.
  const [nx, ny, nz] = NBFLIP_SLIDE_LONG_GRID;
  assert.ok(flume.length > 0);
  for (let z = .5; z < nz; z++) for (let x = .5; x < nx; x++) {
    if (x < 4 || x > nx - 4 || z < 4 || z > nz - 4) assert.equal(longTopAt(x,z), 0, `${x},${z}`);
  }
  assert.ok(Math.max(...longTop) < ny - 4);
});

test("the long water slide snakes the box's length through bends that alternate and use its width", () => {
  const bends = NBFLIP_SLIDE_LONG_ARCS.slice(0, NBFLIP_SLIDE_LONG_BENDS), [nx, , nz] = NBFLIP_SLIDE_LONG_GRID;
  assert.ok(bends.length >= 4);
  bends.forEach(([from, to, turn], bend) => {
    if (bend > 0) assert.equal(turn, -bends[bend - 1]![2], `bend ${bend} turns the way of the one before`);
    // Each is a half turn, and comes back across the box's width.
    assert.ok(Math.abs(Math.abs(nbflipSlideLongCentreline(to)[2] - nbflipSlideLongCentreline(from)[2]) - Math.PI) < 1e-9);
    assert.ok(nbflipSlideLongCentreline(to)[0] > nbflipSlideLongCentreline(from)[0]);
  });
  const xs = [...longCentreline()].map(([, x]) => x), zs = [...longCentreline()].map(([, , z]) => z);
  assert.ok(Math.max(...xs) - Math.min(...xs) > 0.9 * nx, "runs the length");
  // The centreline, that is: the channel and its walls reach eleven cells further each way.
  assert.ok(Math.max(...zs) - Math.min(...zs) > 0.7 * nz, "swings the width");
});

test("the long water slide's channel only falls, between walls high enough on every bend, to a hood that opens on the pool", () => {
  const samples = [...longCentreline()], open: number[] = [];
  const pitSide = NBFLIP_SLIDE_LONG_ARCS[NBFLIP_SLIDE_LONG_ARCS.length - 1]![2];
  let floor = Infinity;
  for (const [distance, x, z, heading] of samples) {
    // Trilinear samples of circular bends have sub-millimetre interpolation error.
    assert.ok(longTopAt(x, z) <= floor + .025, `the floor rises at ${distance}`);
    floor = longTopAt(x, z);
    // Nine cells either side of the centreline is the middle of a wall.
    assert.ok(longBeside(x, z, heading, -9 * pitSide) >= floor + 10, `wall ${longBeside(x, z, heading, -9 * pitSide) - floor} away from the pool at ${distance}`);
    if (longBeside(x, z, heading, 9 * pitSide) < floor + 10) open.push(distance);
    // The stream leans on the outside of a bend: two stream depths of wall there.
    const bend = NBFLIP_SLIDE_LONG_ARCS.find(([from, to]) => distance >= from && distance <= to);
    if (bend) assert.ok(longBeside(x, z, heading, -9 * bend[2]) >= floor + 22, `outer wall ${longBeside(x, z, heading, -9 * bend[2]) - floor} at ${distance}`);
  }
  assert.ok(longTopAt(samples[0]![1], samples[0]![2]) - floor >= 20, "falls at least 20 cells");
  // The wall on the pool's side is missing only where the hood spills: one stretch, at the end.
  assert.ok(open.length > 0 && open[open.length - 1]! > NBFLIP_SLIDE_LONG_LENGTH - 2);
  assert.ok(open[open.length - 1]! - open[0]! < (open.length + 1) * 0.25 && open[0]! > NBFLIP_SLIDE_LONG_LENGTH - 24);
});

test("the long water slide's pool ends between 60% and 80% full below the hood's sill", () => {
  const [x, z, heading] = nbflipSlideLongCentreline(NBFLIP_SLIDE_LONG_LENGTH - 1), sill = longTopAt(x, z);
  const pit = longTop.reduce((lowest, cells) => cells > 0 && cells < lowest ? cells : lowest, NBFLIP_SLIDE_LONG_GRID[1]);
  // Integrate the sampled pool bed; curved boundary cells need not have the
  // exact same height as the flat interior. Channel floors lie at/above sill.
  const capacity = longTop.reduce((sum, cells) => sum + (cells > 0 && cells < sill ? sill - cells : 0), 0);
  const emitted = emittedCells(long);
  assert.equal(longTopAt(x+39*Math.sin(heading),z-39*Math.cos(heading)),4,"flat pool interior stays four cells high");
  // Trilinear cells where the curved wall meets the pool may lower a perimeter
  // sample by up to two millimetres; capacity integrates those cells above.
  assert.ok(pit >= 3.9 && pit <= 4, `pool perimeter floor ${pit}`);
  // With or without the tenth more that the solver has been seen to make of a stream.
  assert.ok(emitted >= 0.6 * capacity && 1.1 * emitted <= 0.8 * capacity, `${capacity} cells for ${emitted}`);
  // The pool is the bottom layer of pages, and the channel's floor never dips into it.
  assert.equal(sill % 32, 0);
});


test("the long slide publishes a continuous boundary with smooth circular wall normals", () => {
  assert.ok(long.solidBoundary);
  validateProceduralSolid(long.solidBoundary);
  const restored = parseScene(serializeScene(long));
  assert.ok(JSON.stringify(restored.solidBoundary) === JSON.stringify(long.solidBoundary), "boundary survives serialization");
  const field = proceduralSolidField(restored)!;
  assert.ok(field, "the registered scene preserves boundary provenance");
  for (const [from, to] of NBFLIP_SLIDE_LONG_ARCS) for (const t of [.13, .35, .65, .8]) {
    const distance = from + t * (to - from);
    const [x,z,heading] = nbflipSlideLongCentreline(distance), floor = longTopAt(x,z);
    for (const side of [-1,1]) {
      const outward = [-side*Math.sin(heading), 0, side*Math.cos(heading)];
      const p = [x+7*outward[0]!, floor+5, z+7*outward[2]!] as const;
      assert.ok(Math.abs(nbflipSlideLongSolid(p)) < 1e-6, "inner wall is on the circle");
      const gradient = field.sample(p)!;
      assert.ok(gradient, "wall carries boundary nodes");
      const alignment = -(gradient[0]*outward[0]!+gradient[2]*outward[2]!)/Math.hypot(...gradient.slice(0,3));
      assert.ok(alignment > .998, `wall normal follows the curve: ${alignment} at ${distance}`);
      assert.ok(Math.abs(gradient[3]) < .025, "sampled wall is within half a millimetre of the circle");
    }
  }
});

test("continuous long-slide joins, slab, hood and spill retain their separate surfaces", () => {
  const field = proceduralSolidField(long)!;
  for (const [from,to] of NBFLIP_SLIDE_LONG_ARCS) for (const distance of [from-.01,from,from+.01,to-.01,to,to+.01]) {
    if (distance >= NBFLIP_SLIDE_LONG_LENGTH) continue;
    const [x,z] = nbflipSlideLongCentreline(distance);
    const floor = 56-24*Math.min(1,distance/(NBFLIP_SLIDE_LONG_LENGTH-30));
    assert.ok(nbflipSlideLongSolid([x,floor-.1,z]) < 0, `connected bed at ${distance}`);
    assert.ok(nbflipSlideLongSolid([x,floor+.1,z]) > 0, `open channel at ${distance}`);
    if (distance < NBFLIP_SLIDE_LONG_LENGTH - 1) {
      assert.ok(Math.abs(field.sample([x,floor,z])![3]) < .025, `baked join stays on the bed at ${distance}`);
    }
  }
  const [x,z] = nbflipSlideLongCentreline(16), floor = longTopAt(x,z);
  assert.ok(nbflipSlideLongSolid([x,floor-7,z]) < 0, "slab remains eight cells thick");
  assert.ok(nbflipSlideLongSolid([x,floor-9,z]) > 0, "air remains below the suspended slab");
  const [hx,hz,heading] = nbflipSlideLongCentreline(NBFLIP_SLIDE_LONG_LENGTH-9);
  assert.ok(nbflipSlideLongSolid([hx,45,hz]) > 0, "clear space under the hood");
  assert.ok(nbflipSlideLongSolid([hx,47,hz]) < 0, "four-cell roof");
  assert.ok(nbflipSlideLongSolid([hx,51,hz]) > 0, "air above the roof");
  assert.ok(nbflipSlideLongSolid([hx+9*Math.sin(heading),37,hz-9*Math.cos(heading)]) > 0, "inner spill gap stays open");
});


test("the long slide rounds exposed rims without moving the wall or flat top", () => {
  const distance=16, [x,z,heading]=nbflipSlideLongCentreline(distance);
  const floor=56-24*distance/(NBFLIP_SLIDE_LONG_LENGTH-30), top=floor+24;
  const sample=(across:number,y:number)=>nbflipSlideLongSolid([x-across*Math.sin(heading),y,z+across*Math.cos(heading)]);
  assert.ok(sample(7.4,top-.4)>0,"the square corner is rounded away");
  assert.ok(sample(7.75,top-.75)<0,"the rounded rim retains wall thickness");
  assert.ok(Math.abs(sample(7,top-3))<1e-6,"vertical channel wall stays in place");
  assert.ok(Math.abs(sample(9,top))<1e-6,"the flat rim top stays at its authored height");
});


test("wall height changes have no internal cap sheet below the step", () => {
  const step=NBFLIP_SLIDE_LONG_ARCS[0]![1]+32+Math.PI/2*32;
  for (const distance of [step-.01,step,step+.01]) {
    const [x,z,heading]=nbflipSlideLongCentreline(distance),floor=56-24*distance/(NBFLIP_SLIDE_LONG_LENGTH-30);
    const p=[x-9*Math.sin(heading),floor+6,z+9*Math.cos(heading)] as const;
    assert.ok(nbflipSlideLongSolid(p)<-1,"the shared lower wall stays solid through the height step");
  }
});
