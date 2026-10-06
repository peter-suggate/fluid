import assert from "node:assert/strict";

/** Diagnostic only. Append counters to the existing surface claims allocation
 * and instrument its shader without changing production bindings or sources.
 * Atomics distort timing: never use this mode as a performance comparison. */
export class UniformSurfaceWorkAudit {
  readonly device: GPUDevice;
  private claims?: { buffer: GPUBuffer; offset: number };
  private readonly read: GPUBuffer;
  private modules = 0;
  constructor(private readonly raw: GPUDevice) {
    this.read = raw.createBuffer({ size: 20, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.device = new Proxy(raw, { get: (target, key) => {
      if (key === "createBuffer") return (d: GPUBufferDescriptor) => {
        if (d.label !== "Uniform mixed surface job claims") return target.createBuffer(d);
        const buffer = target.createBuffer({ ...d, size: d.size + 20 });
        this.claims = { buffer, offset: d.size };
        return buffer;
      };
      if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => {
        if (d.label !== "Uniform mixed surface") return target.createShaderModule(d);
        assert.ok(this.claims, "Surface claims must exist before shader compilation");
        const firstWord = this.claims.offset / 4;
        let code = d.code;
        const count = (i: number) => `atomicAdd(&umClaims[${firstWord + i}u],1u);`;
        const replaceOnce = (from: string, to: string) => {
          assert.equal(code.split(from).length, 2, `Expected one surface audit site: ${from}`);
          code = code.replace(from, to);
        };
        replaceOnce("if(fine){deferred.data[umPreparedIndex(p)]", `atomicAdd(&umClaims[${firstWord}u+select(1u,0u,fine)],1u);if(fine){deferred.data[umPreparedIndex(p)]`);
        replaceOnce("let initial=umLoadVertex(vertex);var value=initial;", `${count(2)}let initial=umLoadVertex(vertex);var value=initial;`);
        replaceOnce("value=umPreparedSearch(vec3f(vertex),initial,1u);", `${count(3)}value=umPreparedSearch(vec3f(vertex),initial,1u);`);
        replaceOnce("let g=umPreparedGradient(q);", `${count(4)}let g=umPreparedGradient(q);`);
        this.modules++;
        return target.createShaderModule({ ...d, code });
      };
      const value = Reflect.get(target, key, target);
      return typeof value === "function" ? value.bind(target) : value;
    } });
  }
  reset(): void {
    assert.ok(this.claims);
    this.raw.queue.writeBuffer(this.claims.buffer, this.claims.offset, new Uint32Array(5));
  }
  async sample() {
    assert.ok(this.claims && this.modules > 0, "Surface audit was not installed");
    const encoder = this.raw.createCommandEncoder();
    encoder.copyBufferToBuffer(this.claims.buffer, this.claims.offset, this.read, 0, 20);
    this.raw.queue.submit([encoder.finish()]);
    await this.read.mapAsync(GPUMapMode.READ);
    const [preparedFine, preparedCoarse, ownedFine, admittedFine, newtonIterations] = new Uint32Array(this.read.getMappedRange()).slice();
    this.read.unmap();
    return { preparedFine, preparedCoarse, ownedFine, admittedFine, newtonIterations };
  }
  destroy(): void { this.read.destroy(); }
}
