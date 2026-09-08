import type { UniformVexSnapshotSource } from "../../tools/implicit-density/uniform-vex-map-gpu";

/** Independently authored native snapshot fixture, including padded IDs,
 * reversed accepted order, nonzero frame/topology generations and parity1. */
export function uniformVexFixture(fineVelocity: readonly number[] = [15, -2.5, 4]) {
  const dimensions = [4, 3, 2] as const, capacity = 32, templateCells = 28, cellBase = 32;
  const topologyBase = cellBase + 8 * templateCells, frameBase = topologyBase + 32 + 2 * capacity;
  const topology = new Uint32Array(frameBase + 64), topologyFloats = new Float32Array(topology.buffer);
  topology.set([0x53434d54, 1, templateCells, 0, 0, 0, cellBase]);
  const accepted = Uint32Array.from({ length: 24 }, (_, i) => 26 - i);
  topology.set([9, 9, 1, 2, accepted.length, 0, capacity, 0], topologyBase);
  topology.set([32, 32 + capacity], topologyBase + 14);
  topology.set(accepted, topologyBase + 32 + capacity);
  topology.set([0x46434131, 1, 64, 106, 3, 8, 8, 3, 14, 64], frameBase);
  topology[frameBase + 13] = 3; topology[frameBase + 14] = 17; topology[frameBase + 15] = 18;
  topology[frameBase + 16] = 1; topology[frameBase + 17] = 1;
  topology[frameBase + 18] = 18; topology[frameBase + 22] = 18;
  topology.set([13, 13, 48], frameBase + 27); topology[frameBase + 32] = 18;
  const activity = new Uint32Array(64), vexBase = 8, depthBase = 32;
  activity.set([0x56455832, 2, 16, capacity, 1, 17, 9], vexBase);
  activity.fill(0xffffffff, depthBase);
  const velocity = new Float32Array(4 * capacity);
  for (let physical = 0; physical < 24; physical++) {
    const id = physical + 3;
    topologyFloats.set([physical % 4 + .5, Math.floor(physical / 4) % 3 + .5,
      Math.floor(physical / 12) + .5, 1, 1, 1, 1], cellBase + 8 * id);
    activity[depthBase + id] = physical % 9;
    velocity.set([...fineVelocity, 1], 4 * id);
  }
  // Invalid VEX cells are present in native ownership, but absent from the
  // recognized map domain. The compiler must not replace their zero values.
  activity[depthBase + 3] = 0xffffffff; velocity.fill(0, 12, 16);
  const parameters = new Uint32Array(64), parameterFloats = new Float32Array(parameters.buffer);
  parameters.set(dimensions, 4); parameterFloats.set([1 / 30, .05], 40);
  const state = new Float32Array(1 + 2 * capacity); state.fill(1);
  return { dimensions, capacity, templateCells, cellBase, topologyBase, frameBase,
    vexBase, depthBase, topology, topologyFloats, activity, velocity, parameters, parameterFloats, state };
}
export type UniformVexFixture = ReturnType<typeof uniformVexFixture>;

/** Independent CPU oracle for fixture semantics. No candidate shader, layout
 * builder, decoder, regression tolerance or inferred scene velocity is used. */
export function inspectUniformVexFixture(q: UniformVexFixture) {
  const { topology: t, topologyFloats: f, topologyBase: b, frameBase: c, activity: a, vexBase: v } = q;
  if (t[0] !== 0x53434d54 || t[1] !== 1 || t[2] !== q.templateCells || t[6] !== q.cellBase
    || t[c] !== 0x46434131 || t[c + 1] !== 1 || t[c + 2] !== 64 || t[c + 3] !== 106 || (t[c + 4]! & 3) !== 3
    || (t[c + 4]! & ~15) !== 0 || t[c + 7] !== 3 || t[c + 8] !== 14 || t[c + 9] !== 64 || t[c + 30] !== 0
    || t[c + 13] !== 3 || t[c + 14]! >= 0x7ffffffe || t[c + 15] !== t[c + 14]! + 1 || t[c + 32] !== t[c + 15]
    || t[c + 18] !== t[c + 15] || t[c + 22] !== t[c + 15] || t[c + 28] !== 13 || t[c + 29] !== 48 || (t[c + 27]! & 13) !== 13
    || t[c + 16]! > 1 || t[c + 17]! > 1
    || a[v] !== 0x56455832 || a[v + 1] !== 2 || a[v + 2] !== 16 || a[v + 3] !== q.capacity
    || a[v + 4] !== 1 || a[v + 11] !== 0 || t[b + 4]! > q.capacity || t[b + 6] !== q.capacity
    || t[b + 2]! > 1 || t[b + 14] !== 32 || t[b + 15] !== 32 + q.capacity
    || ![0, 2].includes(t[b + 3]!)) throw new Error("header");
  if (a[v + 5] !== t[c + 14] || a[v + 6] !== t[b] || t[b] !== t[b + 1]) throw new Error("stale");
  const dt = q.parameterFloats[40]!, h = q.parameterFloats[41]!;
  if (!(Number.isFinite(dt) && dt > 0 && Number.isFinite(h) && h > 0)
    || q.dimensions.some((value, axis) => q.parameters[4 + axis] !== value)) throw new Error("parameters");
  const ids = new Set<number>(), positions = new Set<number>(), coverage = new Uint32Array(24), valid: number[] = [];
  const list = b + t[b + 14 + t[b + 2]!]!;
  for (let ordinal = 0; ordinal < t[b + 4]!; ordinal++) {
    const id = t[list + ordinal]!; if (id >= q.templateCells) throw new Error("geometry");
    if (ids.has(id)) throw new Error("ownership"); ids.add(id);
    const at = q.cellBase + 8 * id, p = [f[at]!, f[at + 1]!, f[at + 2]!];
    if (f[at + 3] !== 1 || [4, 5, 6].some(k => f[at + k] !== 1)
      || p.some((value, axis) => !Number.isFinite(value) || value % 1 !== .5 || value < 0 || value >= q.dimensions[axis]!)) throw new Error("geometry");
    const index = Math.floor(p[0]!) + 4 * (Math.floor(p[1]!) + 3 * Math.floor(p[2]!));
    if (positions.has(index)) throw new Error("ownership"); positions.add(index);
    const depth = a[q.depthBase + id]!;
    if (depth === 0xffffffff) { if (q.velocity[4 * id + 3] !== 0) throw new Error("velocity"); continue; }
    if (depth > 8 || q.velocity[4 * id + 3] !== 1 || !Array.from(q.velocity.subarray(4 * id, 4 * id + 3)).every(Number.isFinite)) throw new Error("velocity");
    if (q.state[1 + id] !== 1 || q.state[1 + q.capacity + id] !== 1) throw new Error("geometry");
    valid.push(id); coverage[index] = id + 1;
  }
  if (valid.length === 0) throw new Error("empty");
  const anchor = Math.min(...valid), bits = new Uint32Array(q.velocity.buffer);
  if (valid.some(id => [0, 1, 2].some(axis => bits[4 * id + axis] !== bits[4 * anchor + axis]))) throw new Error("nonuniform");
  const fineVelocity = Array.from(q.velocity.subarray(4 * anchor, 4 * anchor + 3));
  return { anchor, coverage, valid: valid.length, accepted: ids.size, dt, h,
    physicalVelocity: fineVelocity.map(value => value * h),
    // Independent continuous uniform characteristic; GPU substeps introduce
    // bounded f32 rounding and are compared separately to this physical map.
    translation: fineVelocity.map(value => -value * h * dt) };
}

export const uniformVexCorruptions: readonly [string, string, (fixture: UniformVexFixture) => void][] = [
  ["one-ulp nonuniform velocity", "nonuniform", q => { new Uint32Array(q.velocity.buffer)[4 * 26]! += 1; }],
  ["divergence-free affine shear is unsupported", "nonuniform", q => {
    for (let id = 4; id <= 26; id++) q.velocity[4 * id] = q.topologyFloats[q.cellBase + 8 * id + 1]!;
  }],
  ["rigid rotation velocity is unsupported", "nonuniform", q => {
    for (let id = 4; id <= 26; id++) { q.velocity[4 * id] = -q.topologyFloats[q.cellBase + 8 * id + 1]!;
      q.velocity[4 * id + 1] = q.topologyFloats[q.cellBase + 8 * id]!; }
  }],
  ["nonfinite corner velocity", "velocity", q => { q.velocity[4 * 26] = NaN; }],
  ["invalid VEX cell cannot masquerade as selected", "velocity", q => { q.velocity[4 * 3 + 3] = 1; }],
  ["stale frame receipt", "stale", q => { q.activity[q.vexBase + 5]! -= 1; }],
  ["stale topology receipt", "stale", q => { q.activity[q.vexBase + 6]! -= 1; }],
  ["native frame already faulted", "header", q => { q.topology[q.frameBase + 30] = 9; }],
  ["unsealed frame authority", "header", q => { q.topology[q.frameBase + 13] = 2; }],
  ["stale sealed candidate generation", "header", q => { q.topology[q.frameBase + 32] = 17; }],
  ["incomplete sealed authority coverage", "header", q => { q.topology[q.frameBase + 27] = 1; }],
  ["coarse native support", "geometry", q => { q.topologyFloats[q.cellBase + 8 * 26 + 4] = 2; }],
  ["duplicate accepted native ID", "ownership", q => { q.topology[q.topologyBase + 32 + q.capacity] = 25; }],
  ["duplicate physical ownership", "ownership", q => {
    q.topologyFloats.copyWithin(q.cellBase + 8 * 26, q.cellBase + 8 * 25, q.cellBase + 8 * 25 + 3);
  }],
  ["partially blocked support", "geometry", q => { q.state[1 + 26] = .5; }],
  ["no selected VEX support", "empty", q => { q.activity.fill(0xffffffff, q.depthBase); q.velocity.fill(0); }],
  ["nonfinite actual frame timestep", "parameters", q => { q.parameterFloats[40] = NaN; }],
];

export function uploadUniformVexFixture(device: GPUDevice, q: UniformVexFixture) {
  const buffers: GPUBuffer[] = [];
  const upload = (label: string, data: Uint32Array | Float32Array, uniform = false) => {
    // Match the real resident: parameters are UNIFORM|COPY_DST, not COPY_SRC.
    const buffer = device.createBuffer({ label, size: data.byteLength, usage: GPUBufferUsage.COPY_DST
      | (uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) });
    device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer); buffers.push(buffer); return buffer;
  };
  const source: UniformVexSnapshotSource = {
    topologyArena: upload("QA SCMT and native authority", q.topology), activity: upload("QA VEX authority", q.activity),
    state: upload("QA native apertures", q.state), parameters: upload("QA actual frame uniforms", q.parameters, true),
    effectiveTransportVelocity: upload("QA current native VEX", q.velocity), cellCapacity: q.capacity,
    templateCellCount: q.templateCells, templateCellBaseWords: q.cellBase,
    topologyWorklistBaseWords: q.topologyBase, frameControlBaseWords: q.frameBase,
    velocityExtension: { headerBaseWords: q.vexBase, acceptedDepthBaseWords: q.depthBase, packetCapacity: 1 },
    solidCellOpenBaseWords: 1, solidVoxelCellOpenBaseWords: 1 + q.capacity,
  };
  return { source, destroy: () => buffers.forEach(buffer => buffer.destroy()) };
}
