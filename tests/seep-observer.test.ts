import assert from "node:assert/strict";
import test from "node:test";
import type { SeepObservationSource } from "../lib/methods/seep/seep-observation";
import { levelSetFill, normaliseProfile, restoreLostVolume, seepObservation, seepZoneOracle, type SeepGeometry } from "../tools/seep-observables";
import { createSeepZoneObserver } from "../tools/seep-observe-seep";
import { withUniformDevice } from "./helpers/uniform-geometric";

// The seep observer's zone sums (tools/seep-observables.ts) on a narrow
// synthetic seep: the real wall and throat x, a short right chamber (64
// cells, four bins), 16 cells tall and 4 deep. The CPU oracle is checked
// against closed forms; the WGSL observer (GPU, WEBGPU_NODE_MODULE only)
// against the oracle on the same synthetic SeepObservationSource.
const g: SeepGeometry = { nx: 320, ny: 16, nz: 4, cell_m: 0.005, wall: { low: 240, high: 256 }, throat: [248, 4, 2], throatSide: 2, gravity_m_s2: 9.80665 };
const throatCell = (x: number, y: number, z: number) => x >= 240 && x < 256 && y >= 4 && y < 6 && z >= 2 && z < 4;
const open = (x: number, y: number, z: number) => x >= 0 && y >= 0 && z >= 0 && x < g.nx && y < g.ny && z < g.nz && (x < 240 || x >= 256 || throatCell(x, y, z));
const LEVEL = 10.3, FILM = 1.7;
/** A planar reservoir at y = 10.3, a film at y = 1.7 rising 1/32 per cell right of the wall, a wet throat. */
function phi(x: number, y: number, z: number): number {
  if (x <= 240) return y - LEVEL;
  if (x < 256) return y >= 4 && y <= 6 && z >= 2 && z <= 4 ? -0.5 : 1;
  return y - FILM - (x - 256) / 32 + 0.25 * Math.sin(0.7 * z + 0.3 * x);
}
/** The same scene plus a descending jet: a tube of radius 1.5 from y = 12 at the wall to y = 8 at x = 320, at z = 2. */
function phiJet(x: number, y: number, z: number): number {
  const base = phi(x, y, z);
  if (x < 256) return base;
  return Math.min(base, Math.hypot(y - (12 - (x - 256) / 16), z - 2) - 1.5);
}
const faceU = (x: number, y: number, z: number, axis: number) => axis === 0 ? 0.01 * x + y + 0.1 * z : -1;

test("level-set fill is exact for planes", () => {
  // phi = y - 0.3: three tenths of the cell is liquid.
  const corners = Array.from({ length: 8 }, (_, k) => ((k >> 1) & 1) - 0.3);
  assert.ok(Math.abs(levelSetFill(corners) - 0.3) < 1e-12);
  const tilted = Array.from({ length: 8 }, (_, k) => 0.2 * (k & 1) + 0.5 * ((k >> 1) & 1) + 0.3 * (k >> 2) - 0.5);
  // Liquid where 0.2x + 0.5y + 0.3z < 0.5: the plane through the cell centre halves it.
  assert.ok(Math.abs(levelSetFill(tilted) - 0.5) < 1e-12);
});

test("seep zone oracle matches closed forms", () => {
  const s = seepZoneOracle(g, phi, open, faceU, 7);
  assert.ok(Math.abs(s.left - 240 * LEVEL * g.nz) < 1e-6, `left ${s.left}`);
  // Right: the floor band holds everything below y = 8 (the film stays under it).
  assert.equal(s.floorBand, s.right); assert.equal(s.jet, 0);
  assert.ok(s.right > 0 && s.bins.length === 16 && s.bins.slice(4).every(b => b === 0));
  assert.ok(Math.abs(s.bins.reduce((a, b) => a + b, 0) - s.right) < 1e-9);
  // The film deepens rightward, so its centroid lies right of the chamber's middle.
  assert.ok(s.moment / s.right > 32);
  assert.equal(s.wetFloor, 64 * g.nz);
  assert.ok(s.wall > 0 && s.wall <= 16 * 4, `wall ${s.wall}`);
  assert.equal(s.throatU, faceU(248, 4, 2, 0));
  // Every open wall cell is a throat cell; nothing is above the floor band, so no jet slice holds liquid.
  assert.equal(s.wallOpen, s.wall);
  assert.ok(s.slices.length === 24 && s.slices.every(sl => sl.liquid === 0));
  const row = seepObservation(g, 60, 1 / 60, s, undefined, undefined);
  assert.ok(Math.abs(row.level_m - LEVEL * g.cell_m) < 1e-12);
  assert.ok(Math.abs(row.head_m - (LEVEL - 5) * g.cell_m) < 1e-12);
  assert.ok(Math.abs(row.liquid_L.left - 240 * LEVEL * g.nz * 1.25e-4) < 1e-9);
  assert.equal(row.qLeft_Lps, null); assert.equal(row.particles, 7); assert.equal(row.drift, 0);
  const next = seepObservation(g, 120, 1 / 60, { ...s, total: s.total - 400, left: s.left - 800, right: s.right + 400 }, row, row.liquid_L.total);
  assert.ok(Math.abs(next.qLeft_Lps! - 0.1) < 1e-9 && Math.abs(next.qRight_Lps! - 0.05) < 1e-9);
  assert.ok(Math.abs(next.drift - (-400 * 1.25e-4) / row.liquid_L.total) < 1e-12);
});

test("seep zone oracle jet slices follow a descending jet", () => {
  const s = seepZoneOracle(g, phiJet, open, faceU, 0);
  const row = seepObservation(g, 60, 1 / 60, s, undefined, undefined);
  const slices = row.jetSlices!;
  assert.equal(slices[0]!.x0, 256); assert.equal(slices[8]!.jet_L, 0);
  // Slices 0-6 hold the jet's ~pi 1.5^2 x 8 cells (less where the tube meets z = 0, y = 8 or another liquid region).
  for (let i = 0; i < 7; i++) {
    const sl = slices[i]!, centre = 12 - (8 * i + 4) / 16;
    assert.ok(sl.jet_L > 0 && sl.meanY !== null && Math.abs(sl.meanY - centre) < 0.75, `slice ${i}: mean y ${sl.meanY} vs ${centre}`);
    // u = 0.01 x + y + 0.1 z by cell index on x faces, averaged over a cell's two: near 0.01 (x_c - 1/2) + (y_c - 1/2) + 0.1 (z_c - 1/2).
    assert.ok(Math.abs(sl.meanU! - (0.01 * (256 + 8 * i + 3.5) + sl.meanY! - 0.5 + 0.1 * 1.5)) < 0.1, `slice ${i}: mean u ${sl.meanU}`);
  }
});

/** The synthetic fields as a SeepObservationSource. */
function syntheticSource(device: GPUDevice, phi: (x: number, y: number, z: number) => number): SeepObservationSource {
  const X = g.nx + 1, Y = g.ny + 1, Z = g.nz + 1, cells = g.nx * g.ny * g.nz;
  const phiData = new Float32Array(X * Y * Z), openData = new Uint32Array(cells), uData = new Float32Array(3 * cells);
  for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) phiData[x + X * (y + Y * z)] = phi(x, y, z);
  for (let z = 0; z < g.nz; z++) for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) {
    const i = x + g.nx * (y + g.ny * z); openData[i] = open(x, y, z) ? 1 : 0; for (let a = 0; a < 3; a++) uData[3 * i + a] = faceU(x, y, z, a);
  }
  const upload = (label: string, data: ArrayBufferView & { byteLength: number }) => {
    const buffer = device.createBuffer({ label, size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength); return buffer;
  };
  const buffers = [
    { name: "obsPhi", wgslType: "array<f32>", buffer: upload("phi", phiData) },
    { name: "obsOpen", wgslType: "array<u32>", buffer: upload("open", openData) },
    { name: "obsU", wgslType: "array<f32>", buffer: upload("u", uData) },
    { name: "obsCount", wgslType: "array<u32>", buffer: upload("count", new Uint32Array([7])) },
  ];
  return {
    dimensions: [g.nx, g.ny, g.nz], cellSize_m: g.cell_m, buffers,
    declarations: group => buffers.map((b, i) => `@group(${group}) @binding(${i}) var<storage,read> ${b.name}:${b.wgslType};`).join("\n"),
    wgsl: /* wgsl */`
const OBS_N:vec3i=vec3i(${g.nx},${g.ny},${g.nz});
fn obsCell(c:vec3i)->u32{return u32(c.x+OBS_N.x*(c.y+OBS_N.y*c.z));}
fn seepVertexPhi(v:vec3i)->f32{let p=clamp(v,vec3i(0),OBS_N);return obsPhi[p.x+(OBS_N.x+1)*(p.y+(OBS_N.y+1)*p.z)];}
fn seepCellOpen(c:vec3i)->bool{if(any(c<vec3i(0))||any(c>=OBS_N)){return false;}return obsOpen[obsCell(c)]!=0u;}
fn seepFaceU(c:vec3i,axis:u32)->f32{return obsU[3u*obsCell(c)+axis];}
fn seepParticleCount()->u32{return obsCount[0];}`,
  };
}

for (const [label, field] of [["film", phi], ["film and jet", phiJet]] as const)
test(`seep observer WGSL matches the CPU oracle (${label})`, { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("seep observer", async device => {
    const observer = await createSeepZoneObserver(device, syntheticSource(device, field), g);
    const gpu = await observer.observe(); observer.destroy();
    const cpu = seepZoneOracle(g, field, open, faceU, 7);
    // Fixed point rounds each cell to 1/4096 (moment 1/64 cell-cell): bound by the cell count.
    const cells = g.nx * g.ny * g.nz, near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: gpu ${a} cpu ${b}`);
    for (const k of ["total", "left", "wall", "right", "floorBand", "jet", "wallAll"] as const) near(gpu[k]!, cpu[k]!, cells / 8192 + 1e-3 * Math.abs(cpu[k]!) * 1e-3, k);
    near(gpu.moment, cpu.moment, cells / 128 + 1e-6 * cpu.moment, "moment");
    gpu.bins.forEach((b, i) => near(b, cpu.bins[i]!, cells / 8192, `bin ${i}`));
    assert.equal(gpu.wetFloor, cpu.wetFloor); assert.equal(gpu.particles, 7);
    near(gpu.throatU!, cpu.throatU!, 1e-6, "throat u");
    near(gpu.wallOpen, cpu.wallOpen, cells / 8192, "open wall");
    gpu.slices.forEach((sl, i) => {
      const c = cpu.slices[i]!;
      near(sl.liquid, c.liquid, cells / 8192, `slice ${i} liquid`);
      near(sl.yMoment, c.yMoment, cells / 128 + 1e-6 * c.yMoment, `slice ${i} y moment`);
      near(sl.uMoment, c.uMoment, cells / 2048 + 1e-5 * Math.abs(c.uMoment), `slice ${i} u moment`);
    });
  });
});

test("G6 restoration adds the lost volume uniformly and moves the centroid toward the chamber centre", () => {
  // 2 L piled in the first bin (x 256-272, centroid 264) with 1.6 L lost.
  const bins = Array.from({ length: 16 }, (_, b) => b === 0 ? 2 : 0);
  const r = restoreLostVolume(256, bins, 2, 264, 1.6);
  assert.equal(r.bins_L.length, 16);
  assert.ok(Math.abs(r.bins_L[0]! - 2.1) < 1e-12 && r.bins_L.slice(1).every(v => Math.abs(v - 0.1) < 1e-12));
  assert.ok(Math.abs(r.right_L - 3.6) < 1e-12);
  // The added volume centres at 256 + 128 = 384: (264*2 + 384*1.6) / 3.6.
  assert.ok(Math.abs(r.centroid_x - (264 * 2 + 384 * 1.6) / 3.6) < 1e-12);
  // The centroid formula agrees with the restored bins' own centroid (bin centres 264, 280, ...).
  const fromBins = r.bins_L.reduce((a, v, b) => a + v * (256 + 16 * b + 8), 0) / r.right_L;
  assert.ok(Math.abs(fromBins - r.centroid_x) < 1e-9);
  const p = normaliseProfile(r.bins_L);
  assert.ok(Math.abs(p.reduce((a, v) => a + v, 0) - 1) < 1e-12 && Math.abs(p[0]! - 2.1 / 3.6) < 1e-12);
  // Nothing lost: unchanged.
  const same = restoreLostVolume(256, bins, 2, 264, 0);
  assert.deepEqual(same.bins_L, bins);
  assert.equal(same.centroid_x, 264);
  assert.deepEqual(normaliseProfile(new Array(16).fill(0)), new Array(16).fill(0));
});
