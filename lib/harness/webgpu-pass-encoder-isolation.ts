/** Diagnostic encoder isolation for hardware-counter attribution. PassBroker
 * label isolation is also required: this cannot split dispatches already
 * recorded in one WebGPU pass.
 *
 * A tiny clear is not sufficient on current Dawn/Metal: it can become a
 * compute clear and join the neighboring compute encoder. An explicit copy
 * between the two scratch buffers forces a blit encoder. The
 * extra encoder switches are profiling overhead, never shipping frame cost.
 */
export interface PassEncoderIsolationScratch {readonly source:GPUBuffer;readonly destination:GPUBuffer;destroy():void}

export function isolateComputePassEncoders(
  encoder: GPUCommandEncoder,
  scratch: PassEncoderIsolationScratch,
): GPUCommandEncoder {
  return new Proxy(encoder, {
    get(target, property) {
      if (property === "beginComputePass") {
        return (descriptor?: GPUComputePassDescriptor) => {
          // Legal at every call site: WebGPU forbids encoder commands while a
          // pass is open, so reaching `beginComputePass` already proves the
          // previous pass ended.
          target.copyBufferToBuffer(scratch.source, 0, scratch.destination, 0, 4);
          return target.beginComputePass(descriptor);
        };
      }
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  }) as GPUCommandEncoder;
}

/** One word in each buffer; WebGPU requires different buffer objects. */
export const PASS_ENCODER_ISOLATION_SCRATCH_BYTES = 8;

export function createPassEncoderIsolationScratch(device: GPUDevice): PassEncoderIsolationScratch {
 const source=device.createBuffer({label:"Pass encoder isolation source",size:4,usage:GPUBufferUsage.COPY_SRC});
 const destination=device.createBuffer({label:"Pass encoder isolation destination",size:4,usage:GPUBufferUsage.COPY_DST});
 return {source,destination,destroy(){source.destroy();destination.destroy();}};
}
