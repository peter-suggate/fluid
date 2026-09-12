import assert from "node:assert/strict";
import test from "node:test";

import {
  extendSliceVelocity,
  reconstructSliceInterfaces,
  projectSlicePressureVelocity,
  sliceInterfaceCertificateSample,
  transportSliceVolumeMicrostep,
  type SliceNumericalFields,
  type SliceNumericalTopology,
} from "./slice-stage-numerics";
import { productionSceneSliceSeedById } from "./production-scene-slice";
import { advanceSlice, createAdvanceSlice } from "./slice-solver";

function fields(cells: number, rows: number, subfaces: number): SliceNumericalFields {
  return { density: new Float32Array(cells), gamma: new Float32Array(cells).fill(1),
    capacity: new Float32Array(cells).fill(1), cellVelocity: new Float32Array(2 * cells),
    faceVelocity: new Float32Array(rows), pressure: new Float32Array(cells),
    pressureRhs: new Float32Array(cells), pressureDiagonal: new Float32Array(cells),
    pressureMember: new Uint8Array(cells), extensionDepth: new Uint8Array(cells),
    interfaceNormal: new Float32Array(2 * cells), interfaceOffset: new Float32Array(cells),
    lowFlux: new Float32Array(subfaces), highFlux: new Float32Array(subfaces),
    limitedFlux: new Float32Array(subfaces), fault: null };
}

test("interface certificates clamp only the transport-admissible volume margin",()=>{
  const margin=9.5367431640625e-7;
  assert.equal(sliceInterfaceCertificateSample(-margin),0);
  assert.equal(sliceInterfaceCertificateSample(1+margin),1);
  assert.equal(sliceInterfaceCertificateSample(-2*margin),null);
  assert.equal(sliceInterfaceCertificateSample(1+2*margin),null);
  assert.equal(sliceInterfaceCertificateSample(Number.NaN),null);
});

test("zero-gradient ELVIRA axes do not invent a positive reflection orientation", () => {
  const cells = Array.from({ length: 9 }, (_, id) => { const x = id % 3, y = Math.floor(id / 3);
    return { id, minimum: [x, y] as const, maximum: [x + 1, y + 1] as const,
      center: [x + .5, y + .5] as const, widths: [1, 1] as const, area: 1 }; });
  const rows: SliceNumericalTopology["rows"][number][] = [], incidences = Array.from({ length: 9 }, () => [] as number[]);
  const connect = (a: number, b: number, axis: 0 | 1, center: readonly [number, number]) => {
    const id = rows.length; rows.push({ id, kind: "intra-brick", axis, center, area: 1,
      distance: 1, dualWeight: 1, terms: [{ cellId: a, coefficient: -1 }, { cellId: b, coefficient: 1 }] });
    incidences[a]!.push(id); incidences[b]!.push(id);
  };
  for (let y = 0; y < 3; y += 1) for (let x = 0; x < 2; x += 1)
    connect(x + 3 * y, x + 1 + 3 * y, 0, [x + 1, y + .5]);
  for (let y = 0; y < 2; y += 1) for (let x = 0; x < 3; x += 1)
    connect(x + 3 * y, x + 3 * (y + 1), 1, [x + .5, y + 1]);
  const state = fields(9, rows.length, 0);
  state.density.set([1, 1, 1, 1, .9999998807907104, 1, 1, .9999999403953552, 1]);
  reconstructSliceInterfaces({ dimensions: [3, 3], cells, rows, incidences, subfaces: [] }, state);
  assert.equal(Math.abs(state.interfaceNormal[2 * 4]!), 0,
    "an X-invariant stencil must not publish a +X-oriented plane");
  assert.equal(Math.abs(state.interfaceNormal[2 * 4 + 1]!), 1);
});

test("VEX uses physical mixed-port pieces and the production axis reduction tree", () => {
  const topology: SliceNumericalTopology = { cells: [
    { id: 0, minimum: [0, 0], maximum: [2, 2], center: [1, 1], widths: [2, 2], area: 4 },
    { id: 1, minimum: [2, 0], maximum: [3, 1], center: [2.5, .5], widths: [1, 1], area: 1 },
    { id: 2, minimum: [2, 1], maximum: [3, 2], center: [2.5, 1.5], widths: [1, 1], area: 1 },
    { id: 3, minimum: [0, 2], maximum: [2, 4], center: [1, 3], widths: [2, 2], area: 4 },
  ], rows: [
    { id: 0, kind: "mixed-seam", axis: 0, center: [2, 1], area: 2, distance: 1.5,
      dualWeight: 3, terms: [{ cellId: 0, coefficient: -2 / 3 },
        { cellId: 1, coefficient: 1 / 3 }, { cellId: 2, coefficient: 1 / 3 }] },
    { id: 1, kind: "brick-face", axis: 1, center: [1, 2], area: 2, distance: 2,
      dualWeight: 4, terms: [{ cellId: 0, coefficient: -0.5 },
        { cellId: 3, coefficient: 0.5 }] },
  ], incidences: [[0, 1], [0], [0], [1]], subfaces: [] };
  const state = fields(4, 2, 0);
  state.density.set([0, 1, 1, 1]);
  state.cellVelocity.set([0, 0, 1, 0, 3, 0, 10, 0]);
  extendSliceVelocity(topology, state, 1);

  // X contributes two physical area-one pieces at distance 1.5; Y contributes
  // area two at distance two. The WGSL tree evaluates (x-+x+)+(y-+y+).
  const f = Math.fround;
  const mixedWeight = f(f(f(2) * f(Math.abs(-2 / 3) * 1.5))
    * f(Math.abs(1 / 3) * 1.5) / f(1.5));
  const yWeight = f(f(f(2) * f(Math.abs(-.5) * 2)) * f(Math.abs(.5) * 2) / f(2));
  const expected = f(f(f(mixedWeight * 1) + f(mixedWeight * 3) + f(yWeight * 10))
    / f(f(mixedWeight + mixedWeight) + yWeight));
  assert.equal(state.cellVelocity[0], expected);
  assert.equal(state.extensionDepth[0], 1);
});

test("mixed-seam pressure gradients are equivariant when reflection swaps the fine side",()=>{
  const cells=Array.from({length:6},(_,id)=>({id,minimum:[id,0] as [number,number],
    maximum:[id+1,1] as [number,number],center:[id+.5,.5] as [number,number],
    widths:[1,1] as [number,number],area:1}));
  const topology:SliceNumericalTopology={cells,rows:[
    {id:0,kind:"mixed-seam",axis:0,center:[1,.5],area:1,distance:1,dualWeight:1,
      terms:[{cellId:0,coefficient:-2/3},{cellId:1,coefficient:1/3},{cellId:2,coefficient:1/3}]},
    {id:1,kind:"mixed-seam",axis:0,center:[5,.5],area:1,distance:1,dualWeight:1,
      terms:[{cellId:3,coefficient:-1/3},{cellId:4,coefficient:-1/3},{cellId:5,coefficient:2/3}]},
  ],incidences:Array.from({length:6},()=>[]),subfaces:[]};
  const state=fields(6,2,0);state.pressureMember.fill(1);
  state.pressure.set([100000,100001,99999,99999,100001,100000]);
  state.faceVelocity.set([.25,-.25]);
  projectSlicePressureVelocity(topology,state,{active:new Uint8Array([1,1]),
    theta:new Float32Array([1,1])});
  assert.equal(state.faceVelocity[0],-state.faceVelocity[1]);
});

test("moving low-flux authority evacuates an exactly closed cell", () => {
  const topology: SliceNumericalTopology = { dimensions: [2, 1], cells: [
    { id: 0, minimum: [0, 0], maximum: [1, 1], center: [.5, .5], widths: [1, 1], area: 1 },
    { id: 1, minimum: [1, 0], maximum: [2, 1], center: [1.5, .5], widths: [1, 1], area: 1 },
  ], rows: [{ id: 0, kind: "intra-brick", axis: 0, center: [1, .5], area: 1,
    distance: 1, dualWeight: 1, terms: [{ cellId: 0, coefficient: -1 },
      { cellId: 1, coefficient: 1 }], openFraction: 1 }],
  subfaces: [{ id: 0, rowId: 0, axis: 0, center: [1, .5], area: 1,
    negativeCell: 0, positiveCell: 1 }], incidences: [[0], [0]],
  subfaceIncidences: [[{ subfaceId: 0, negative: true }],
    [{ subfaceId: 0, negative: false }]] };
  const state = fields(2, 1, 1);
  state.density.set([.1, .2]); state.faceVelocity[0] = .1;
  state.solidMotionActive = true;
  state.capacityBefore = new Float32Array([1, 1]);
  state.capacityAfter = new Float32Array([0, 1]);
  state.capacity.set(state.capacityAfter);

  const receipt = transportSliceVolumeMicrostep(topology, state, 1);
  assert.equal(state.fault, null);
  assert.equal(receipt.initialLowFlux[0], Math.fround(Math.fround(.1) * Math.fround(.1)));
  assert.equal(receipt.lowFlux[0], Math.fround(.1));
  assert.equal(receipt.nextVolume[0], 0);
  assert.equal(state.density[0], 0);
  assert.equal(state.density[1], Math.fround(.3));
});

test("transport-roundoff samples keep reflected ELVIRA certificates through impact", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-first-pool-impact-half"));
  for (let frame = 0; frame < 5; frame += 1) advanceSlice(slice, { pressureIterations: 64 });
  assert.equal(slice.fault, null);
  assert.equal(slice.frame, 5, "the symmetry check must cover five completed steps");
  const topology = slice.topology.accepted, width = topology.dimensions[0];
  const byBounds = new Map(topology.cells.map(cell => [
    `${cell.minimumFine[0]}:${cell.minimumFine[1]}:${cell.maximumFine[0]}:${cell.maximumFine[1]}`,
    cell,
  ]));
  let maximum = 0;
  for (const cell of topology.cells) {
    const mirror = byBounds.get(`${width - cell.maximumFine[0]}:${cell.minimumFine[1]}:${
      width - cell.minimumFine[0]}:${cell.maximumFine[1]}`);
    assert.ok(mirror, `missing reflected owner for cell ${cell.id}`);
    maximum = Math.max(maximum,
      Math.abs(slice.fields.density[cell.id]! - slice.fields.density[mirror.id]!));
  }
  assert.ok(maximum <= 4 * 9.5367431640625e-7,
    `roundoff-valid reflected impact diverged by ${maximum}`);
});
