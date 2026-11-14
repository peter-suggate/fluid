import assert from "node:assert/strict";
import test from "node:test";
import type { SeepObservationSource } from "../lib/methods/seep/seep-observation";
import { levelSetFill, seepObservation, seepZoneOracle, type SeepGeometry } from "../tools/seep-observables";
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
  const row = seepObservation(g, 60, 1 / 60, s, undefined, undefined);
  assert.ok(Math.abs(row.level_m - LEVEL * g.cell_m) < 1e-12);
  assert.ok(Math.abs(row.head_m - (LEVEL - 5) * g.cell_m) < 1e-12);
  assert.ok(Math.abs(row.liquid_L.left - 240 * LEVEL * g.nz * 1.25e-4) < 1e-9);
  assert.equal(row.qLeft_Lps, null); assert.equal(row.particles, 7); assert.equal(row.drift, 0);
  const next = seepObservation(g, 120, 1 / 60, { ...s, total: s.total - 400, left: s.left - 800, right: s.right + 400 }, row, row.liquid_L.total);
  assert.ok(Math.abs(next.qLeft_Lps! - 0.1) < 1e-9 && Math.abs(next.qRight_Lps! - 0.05) < 1e-9);
  assert.ok(Math.abs(next.drift - (-400 * 1.25e-4) / row.liquid_L.total) < 1e-12);
});

/** The synthetic fields as a SeepObservationSource. */
function syntheticSource(device: GPUDevice): SeepObservationSource {
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

test("seep observer WGSL matches the CPU oracle", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("seep observer", async device => {
    const observer = await createSeepZoneObserver(device, syntheticSource(device), g);
    const gpu = await observer.observe(); observer.destroy();
    const cpu = seepZoneOracle(g, phi, open, faceU, 7);
    // Fixed point rounds each cell to 1/4096 (moment 1/64 cell-cell): bound by the cell count.
    const cells = g.nx * g.ny * g.nz, near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: gpu ${a} cpu ${b}`);
    for (const k of ["total", "left", "wall", "right", "floorBand", "jet", "wallAll"] as const) near(gpu[k]!, cpu[k]!, cells / 8192 + 1e-3 * Math.abs(cpu[k]!) * 1e-3, k);
    near(gpu.moment, cpu.moment, cells / 128 + 1e-6 * cpu.moment, "moment");
    gpu.bins.forEach((b, i) => near(b, cpu.bins[i]!, cells / 8192, `bin ${i}`));
    assert.equal(gpu.wetFloor, cpu.wetFloor); assert.equal(gpu.particles, 7);
    near(gpu.throatU!, cpu.throatU!, 1e-6, "throat u");
  });
});
