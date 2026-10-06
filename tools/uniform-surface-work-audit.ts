import assert from "node:assert/strict";

/** Diagnostic only. Append counters to the existing surface claims allocation
 * and instrument its shader without changing production bindings or sources.
 * Atomics distort timing: never use this mode as a performance comparison. */
export class UniformSurfaceWorkAudit {
  readonly device: GPUDevice;
  private claims?: { buffer: GPUBuffer; offset: number };
  private read?: GPUBuffer;
  private bytes = 20;
  private dimensions?: number[];
  private modules = 0;
  constructor(private readonly raw: GPUDevice) {
    this.device = new Proxy(raw, { get: (target, key) => {
      if (key === "createBuffer") return (d: GPUBufferDescriptor) => {
        if (d.label !== "Uniform mixed surface job claims") return target.createBuffer(d);
        // Original storage includes two words per 4h vertex. A fine vertex
        // lattice has fewer than 64 times as many vertices, so 32 times the
        // original byte size safely holds one diagnostic word per fine vertex.
        this.bytes = 20 + 32 * d.size;
        this.read?.destroy();
        this.read = target.createBuffer({ size: this.bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const buffer = target.createBuffer({ ...d, size: d.size + this.bytes });
        this.claims = { buffer, offset: d.size };
        return buffer;
      };
      if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => {
        if (d.label !== "Uniform mixed surface") return target.createShaderModule(d);
        assert.ok(this.claims, "Surface claims must exist before shader compilation");
        const firstWord = this.claims.offset / 4;
        let code = d.code;
        const dimensions = /const UM_D=vec3u\((\d+)u,(\d+)u,(\d+)u\)/.exec(code);
        assert.ok(dimensions, "Surface diagnostic requires literal lattice dimensions");
        this.dimensions = dimensions.slice(1).map(Number);
        assert.ok(20 + 4 * this.dimensions.reduce((a, n) => a * (n + 1), 1) <= this.bytes);
        code += "\nvar<private> auditIterations:u32;\n";
        const count = (i: number) => `atomicAdd(&umClaims[${firstWord + i}u],1u);`;
        const replaceOnce = (from: string, to: string) => {
          assert.equal(code.split(from).length, 2, `Expected one surface audit site: ${from}`);
          code = code.replace(from, to);
        };
        replaceOnce("if(fine){deferred.data[umPreparedIndex(p)]", `atomicAdd(&umClaims[${firstWord}u+select(1u,0u,fine)],1u);if(fine){deferred.data[umPreparedIndex(p)]`);
        replaceOnce("let initial=umLoadVertex(vertex);var value=initial;", `${count(2)}let initial=umLoadVertex(vertex);var value=initial;`);
        replaceOnce("value=umPreparedSearch(vec3f(vertex),initial,1u);", `${count(3)}auditIterations=0u;value=umPreparedSearch(vec3f(vertex),initial,1u);auditRecord=2u+auditIterations;`);
        replaceOnce("let g=umPreparedGradient(q);", `${count(4)}auditIterations++;let g=umPreparedGradient(q);`);
        replaceOnce("let initial=umLoadVertex(vertex);var value=initial;", "let initial=umLoadVertex(vertex);var value=initial;var auditRecord=1u;");
        replaceOnce("auditRecord=2u+auditIterations;}", `auditRecord=2u+auditIterations;}atomicStore(&umClaims[${firstWord + 5}u+vertex.x+(UM_D.x+1u)*(vertex.y+(UM_D.y+1u)*vertex.z)],auditRecord|select(0u,65536u,umRegularFine));`);
        this.modules++;
        return target.createShaderModule({ ...d, code });
      };
      const value = Reflect.get(target, key, target);
      return typeof value === "function" ? value.bind(target) : value;
    } });
  }
  reset(): void {
    assert.ok(this.claims);
    const encoder = this.raw.createCommandEncoder();
    encoder.clearBuffer(this.claims.buffer, this.claims.offset, this.bytes);
    this.raw.queue.submit([encoder.finish()]);
  }
  async sample() {
    assert.ok(this.claims && this.read && this.dimensions && this.modules > 0, "Surface audit was not installed");
    const encoder = this.raw.createCommandEncoder();
    encoder.copyBufferToBuffer(this.claims.buffer, this.claims.offset, this.read, 0, this.bytes);
    this.raw.queue.submit([encoder.finish()]);
    await this.read.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(this.read.getMappedRange()).slice();
    const [preparedFine, preparedCoarse, ownedFine, admittedFine, newtonIterations] = data;
    this.read.unmap();
    const distribution = summarizeSurfaceDistribution(data.subarray(5), this.dimensions);
    assert.equal(distribution.owned, ownedFine, "Every owned fine vertex recorded once");
    assert.equal(distribution.admitted, admittedFine, "Every admitted search recorded once");
    assert.equal(distribution.iterations, newtonIterations, "Every Newton iteration recorded once");
    return { preparedFine, preparedCoarse, ownedFine, admittedFine, newtonIterations, distribution };
  }
  destroy(): void { this.read?.destroy(); }
}

/** Logical 32-lane cost model, not measured hardware occupancy. Records retain
 * the original tile/round/lane mapping, including negative domain closures.
 * Compact models pack only within a spatial block and sampling specialization.
 * Actual Newton lengths are used to account for divergent loop tails. */
export function summarizeSurfaceDistribution(data: Uint32Array, dimensions: readonly number[]) {
  const [nx, ny, nz] = dimensions as [number, number, number];
  const groups = new Map<number, number[]>();
  const blocks = [1, 2, 4].map(() => new Map<string, number[]>());
  let owned = 0, admitted = 0, iterations = 0;
  const iterationHistogram: number[] = [];
  for (let z = 0; z <= nz; z++) for (let y = 0; y <= ny; y++) for (let x = 0; x <= nx; x++) {
    const record = data[x + (nx + 1) * (y + (ny + 1) * z)]!;
    const state = record & 65535;
    if (!state) continue;
    owned++;
    const p = [x, y, z];
    const t = p.map(v => Math.floor(Math.max(0, v - 1) / 4));
    const local = p.map((v, a) => v - 4 * t[a]!);
    const boundary = local.some(v => v === 0);
    const j = boundary ? local[0]! + 5 * (local[1]! + 5 * local[2]!) : local[0]! - 1 + 4 * (local[1]! - 1 + 4 * (local[2]! - 1));
    const round = boundary ? 1 + Math.floor(j / 64) : 0;
    const lane = j % 64;
    const tile = t[0]! + (nx / 4) * (t[1]! + (ny / 4) * t[2]!);
    const key = tile * 6 + round * 2 + Math.floor(lane / 32);
    let group = groups.get(key);
    if (!group) { group = new Array<number>(32).fill(0); groups.set(key, group); }
    if (state < 2) continue;
    const length = state - 2;
    admitted++; iterations += length;
    iterationHistogram[length] = (iterationHistogram[length] ?? 0) + 1;
    group[lane % 32] = length;
    [1, 2, 4].forEach((size, i) => {
      const key = `${record >>> 16}:${t.map(v => Math.floor(v / size)).join(",")}`;
      let list = blocks[i]!.get(key);
      if (!list) { list = []; blocks[i]!.set(key, list); }
      list.push(length);
    });
  }
  const cost = (lists: Iterable<number[]>) => {
    let groups = 0, activeGroups = 0, laneIterations = 0;
    const activeLanes: number[] = new Array(33).fill(0);
    for (const list of lists) for (let i = 0; i < list.length; i += 32) {
      const batch = list.slice(i, i + 32); const active = batch.filter(n => n > 0).length;
      groups++; activeLanes[active]!++; if (active) activeGroups++;
      laneIterations += 32 * Math.max(0, ...batch);
    }
    return { groups, activeGroups, laneIterations, usefulIterationFraction: iterations / Math.max(1, laneIterations), activeLanes };
  };
  return { owned, admitted, iterations, iterationHistogram, current: cost(groups.values()),
    spatialCompact: Object.fromEntries([1, 2, 4].map((n, i) => [n, cost(blocks[i]!.values())])) };
}
