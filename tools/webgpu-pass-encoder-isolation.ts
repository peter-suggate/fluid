/**
 * Break every WebGPU compute pass into its own Metal command encoder.
 *
 * READ THIS FIRST: **this is superseded, and it never fixed anything.** Use
 * `--isolate-pass-labels` (`FLUID_GPU_ISOLATE_PASS_LABELS=1`, implemented in
 * `lib/webgpu-pass-broker.ts`) for per-pass attribution. This module is kept
 * only as the measured record of a wrong diagnosis.
 *
 * THE PROBLEM IT WAS AIMED AT. `FLUID_GPU_PASS_TIMESTAMPS=1` looks like
 * per-pass GPU attribution but was not. Measured 2026-07-28 on the mini dam
 * lane: splitting the structured-boundary workset publication into five fenced
 * passes put 3.670 ms on one of them and 0.066 ms on the rest, and the number
 * moved with the slot rather than with the kernel.
 *
 * WHAT THIS DOES. A buffer clear is a blit, not a compute command, so Dawn has
 * to close the open compute encoder to record it. Emitting a 4-byte clear
 * before every `beginComputePass` therefore stops passes sharing an encoder.
 *
 * WHY THAT DIDN'T HELP -- and the diagnosis it rests on is wrong. Encoder
 * merging was never the cause. `PassBroker.compute({label})` returns the
 * ALREADY-OPEN pass and drops the new label, so a five-dispatch group is a
 * single pass whose bracket runs on across every downstream stage until the
 * next fence. Splitting Metal encoders cannot separate dispatches that WebGPU
 * recorded into one pass, which is why isolation changed the attribution by
 * nothing. Fencing on a label change does separate them: the same workset group
 * then reports 0.19 ms in total, `Workset scatter families` above
 * `Workset scatter rows` as their work implies, and the numbers follow the
 * kernels when the two dispatches are swapped.
 *
 * The second half of the old diagnosis was also wrong. Dawn's default
 * `timestamp_quantization` toggle rounds every result to 65536 ns, which is why
 * the old readings were all integer multiples of 0.066 ms and why anything
 * genuinely cheap read as exactly zero. The smoke runner now disables that
 * toggle whenever timestamps are requested.
 *
 * WHAT THIS IS STILL GOOD FOR. The +0.9 ms it costs on a 168-pass frame is a
 * direct measurement of per-encoder switch overhead -- about 5 us each, ~2% of
 * the frame. That is independent evidence that this frame is not bound by
 * command-submission structure.
 *
 * The clear is deliberately NOT routed through the command audit: that audit
 * describes the solver's own commands, and keeping the instrument out of it is
 * what lets an isolated run and an ordinary run be compared dispatch-for-dispatch.
 */

/**
 * Wrap `encoder` so every compute pass begins in a fresh command encoder.
 *
 * Apply this *under* any auditing proxy, so the isolation clears stay invisible
 * to the audit while the audit still observes every pass the solver begins.
 */
export function isolateComputePassEncoders(
  encoder: GPUCommandEncoder,
  scratch: GPUBuffer,
): GPUCommandEncoder {
  return new Proxy(encoder, {
    get(target, property) {
      if (property === "beginComputePass") {
        return (descriptor?: GPUComputePassDescriptor) => {
          // Legal at every call site: WebGPU forbids encoder commands while a
          // pass is open, so reaching `beginComputePass` already proves the
          // previous pass ended.
          target.clearBuffer(scratch, 0, PASS_ENCODER_ISOLATION_SCRATCH_BYTES);
          return target.beginComputePass(descriptor);
        };
      }
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  }) as GPUCommandEncoder;
}

/** Smallest clear Dawn will still record as a blit. */
export const PASS_ENCODER_ISOLATION_SCRATCH_BYTES = 4;

export function createPassEncoderIsolationScratch(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: "Pass encoder isolation scratch",
    size: PASS_ENCODER_ISOLATION_SCRATCH_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });
}
