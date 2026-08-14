#!/usr/bin/env node
/**
 * Minimal proof-of-concept for an in-dispatch WebGPU "software clock".
 *
 * One workgroup repeatedly increments a storage atomic while independent
 * payload workgroups record the counter before and after deterministic ALU
 * work. Hardware timestamp queries sandwich the complete compute pass. The
 * result tells us whether this adapter schedules the ticker concurrently
 * enough to support an explicitly experimental workgroup reconstruction.
 *
 * This is deliberately standalone: it does not touch solver shaders or state.
 *
 * Run:
 *   node --import tsx tools/experiment-webgpu-atomic-clock.ts
 *
 * Optional environment:
 *   FLUID_ATOMIC_CLOCK_WORKGROUPS=2048
 *   FLUID_ATOMIC_CLOCK_WORKGROUP_SIZE=1
 *   FLUID_ATOMIC_CLOCK_ITERATIONS=16384
 *   FLUID_ATOMIC_CLOCK_MAX_TICKS=5000000
 *   FLUID_ATOMIC_CLOCK_TRIALS=10
 *   WEBGPU_BACKEND=metal
 */
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

const unsignedInteger = (name: string, fallback: number, maximum: number): number => {
  const raw = process.env[name];
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < 1 || value > maximum) {
    throw new RangeError(`${name} must be an integer in [1, ${maximum}]`);
  }
  return value;
};

const payloadWorkgroups = unsignedInteger("FLUID_ATOMIC_CLOCK_WORKGROUPS", 2_048, 65_000);
const payloadWorkgroupSize = unsignedInteger("FLUID_ATOMIC_CLOCK_WORKGROUP_SIZE", 1, 256);
const payloadIterations = unsignedInteger("FLUID_ATOMIC_CLOCK_ITERATIONS", 16_384, 1_000_000);
const maximumClockTicks = unsignedInteger("FLUID_ATOMIC_CLOCK_MAX_TICKS", 5_000_000, 100_000_000);
const trials = unsignedInteger("FLUID_ATOMIC_CLOCK_TRIALS", 10, 20);
const maximumTickRateVariation = 0.15;
const maximumInstrumentationOverheadPercent = 25;
const backend = process.env.WEBGPU_BACKEND ?? "metal";
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));

const HEADER_WORDS = 8;
const RECORD_WORDS = 8;
const CLOCK = 0;
const COMPLETED = 1;
const SEQUENCE = 2;
const CLOCK_STATE = 3;
const RECORD_START_TICK = 0;
const RECORD_END_TICK = 1;
const RECORD_ENTRY_TICKET = 2;
const RECORD_EXIT_TICKET = 3;
const RECORD_CHECKSUM = 4;
const RECORD_START_STATE = 5;
const RECORD_END_STATE = 6;

interface NumericSummary {
  minimum: number;
  median: number;
  mean: number;
  maximum: number;
  coefficientOfVariation: number;
}

interface RunResult {
  pass_ms: number;
  words: Uint32Array;
}

interface ClockTrial {
  pass_ms: number;
  finalClock: number;
  clockState: number;
  completedWorkgroups: number;
  sequenceTickets: number;
  startVisibleFraction: number;
  advancingFraction: number;
  clockRunningAtStartFraction: number;
  clockRunningAtEndFraction: number;
  medianPayloadTicks: number;
  maximumConcurrentIntervals: number;
  tickRatePer_ms: number;
  checksumSamplesCorrect: boolean;
}

const summarize = (values: readonly number[]): NumericSummary => {
  if (values.length === 0) {
    return { minimum: 0, median: 0, mean: 0, maximum: 0, coefficientOfVariation: 0 };
  }
  const ordered = [...values].sort((left, right) => left - right);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return {
    minimum: ordered[0],
    median: ordered[Math.floor(ordered.length / 2)],
    mean,
    maximum: ordered.at(-1) ?? ordered[0],
    coefficientOfVariation: mean > 0 ? Math.sqrt(variance) / mean : 0,
  };
};

const expectedChecksum = (logicalWorkgroup: number): number => {
  let x = (Math.imul((logicalWorkgroup + 1) >>> 0, 747_796_405) + 2_891_336_453) >>> 0;
  let accumulator = (x ^ 0x9e37_79b9) >>> 0;
  for (let iteration = 0; iteration < payloadIterations; iteration += 1) {
    x = (x ^ (x >>> 16)) >>> 0;
    x = (Math.imul(x, 2_246_822_519) + 3_266_489_917) >>> 0;
    x = (x ^ (x << 13)) >>> 0;
    x = (x + Math.imul(iteration, 2_654_435_761)) >>> 0;
    accumulator = (accumulator ^ ((x + (accumulator << 5) + (accumulator >>> 2)) >>> 0)) >>> 0;
  }
  return accumulator;
};

const maximumConcurrentIntervals = (words: Uint32Array): number => {
  const events: Array<readonly [number, -1 | 1]> = [];
  for (let workgroup = 0; workgroup < payloadWorkgroups; workgroup += 1) {
    const base = HEADER_WORDS + workgroup * RECORD_WORDS;
    const start = words[base + RECORD_START_TICK];
    const end = words[base + RECORD_END_TICK];
    if (end <= start) continue;
    events.push([start, 1], [end, -1]);
  }
  events.sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  let active = 0;
  let maximum = 0;
  for (const [, delta] of events) {
    active += delta;
    maximum = Math.max(maximum, active);
  }
  return maximum;
};

const decodeClockTrial = (result: RunResult): ClockTrial => {
  let startVisible = 0;
  let advancing = 0;
  let runningAtStart = 0;
  let runningAtEnd = 0;
  const durations: number[] = [];
  const checksumSampleIds = [0, Math.floor(payloadWorkgroups / 3), Math.floor(2 * payloadWorkgroups / 3), payloadWorkgroups - 1];
  let checksumSamplesCorrect = true;
  for (let workgroup = 0; workgroup < payloadWorkgroups; workgroup += 1) {
    const base = HEADER_WORDS + workgroup * RECORD_WORDS;
    const start = result.words[base + RECORD_START_TICK];
    const end = result.words[base + RECORD_END_TICK];
    if (start > 0) startVisible += 1;
    if (end > start) {
      advancing += 1;
      durations.push(end - start);
    }
    if (result.words[base + RECORD_START_STATE] === 1) runningAtStart += 1;
    if (result.words[base + RECORD_END_STATE] === 1) runningAtEnd += 1;
    if (checksumSampleIds.includes(workgroup)) {
      checksumSamplesCorrect &&= result.words[base + RECORD_CHECKSUM] === expectedChecksum(workgroup);
    }
  }
  const finalClock = result.words[CLOCK];
  return {
    pass_ms: result.pass_ms,
    finalClock,
    clockState: result.words[CLOCK_STATE],
    completedWorkgroups: result.words[COMPLETED],
    sequenceTickets: result.words[SEQUENCE],
    startVisibleFraction: startVisible / payloadWorkgroups,
    advancingFraction: advancing / payloadWorkgroups,
    clockRunningAtStartFraction: runningAtStart / payloadWorkgroups,
    clockRunningAtEndFraction: runningAtEnd / payloadWorkgroups,
    medianPayloadTicks: summarize(durations).median,
    maximumConcurrentIntervals: maximumConcurrentIntervals(result.words),
    tickRatePer_ms: result.pass_ms > 0 ? finalClock / result.pass_ms : 0,
    checksumSamplesCorrect,
  };
};

const shaderSource = /* wgsl */ `
const PAYLOAD_WORKGROUPS = ${payloadWorkgroups}u;
const PAYLOAD_ITERATIONS = ${payloadIterations}u;
const MAXIMUM_CLOCK_TICKS = ${maximumClockTicks}u;
const HEADER_WORDS = ${HEADER_WORDS}u;
const RECORD_WORDS = ${RECORD_WORDS}u;

@group(0) @binding(0) var<storage, read_write> telemetry: array<atomic<u32>>;

fn recordBase(logicalWorkgroup: u32) -> u32 {
  return HEADER_WORDS + logicalWorkgroup * RECORD_WORDS;
}

fn payload(logicalWorkgroup: u32) -> u32 {
  var x = (logicalWorkgroup + 1u) * 747796405u + 2891336453u;
  var accumulator = x ^ 0x9e3779b9u;
  for (var iteration = 0u; iteration < PAYLOAD_ITERATIONS; iteration += 1u) {
    x = x ^ (x >> 16u);
    x = x * 2246822519u + 3266489917u;
    x = x ^ (x << 13u);
    x = x + iteration * 2654435761u;
    accumulator = accumulator ^ (x + (accumulator << 5u) + (accumulator >> 2u));
  }
  return accumulator;
}

@compute @workgroup_size(${payloadWorkgroupSize})
fn baseline(
  @builtin(workgroup_id) workgroup: vec3u,
  @builtin(local_invocation_index) localInvocation: u32,
) {
  if (localInvocation != 0u) { return; }
  let logical = workgroup.x;
  let base = recordBase(logical);
  atomicStore(&telemetry[base + ${RECORD_CHECKSUM}u], payload(logical));
}

@compute @workgroup_size(${payloadWorkgroupSize})
fn clocked(
  @builtin(workgroup_id) workgroup: vec3u,
  @builtin(local_invocation_index) localInvocation: u32,
) {
  if (workgroup.x == 0u) {
    if (localInvocation != 0u) { return; }
    atomicStore(&telemetry[${CLOCK_STATE}u], 1u);
    var localTicks = 0u;
    loop {
      atomicAdd(&telemetry[${CLOCK}u], 1u);
      localTicks += 1u;
      if ((localTicks & 255u) == 0u && atomicLoad(&telemetry[${COMPLETED}u]) >= PAYLOAD_WORKGROUPS) {
        atomicStore(&telemetry[${CLOCK_STATE}u], 2u);
        return;
      }
      if (localTicks >= MAXIMUM_CLOCK_TICKS) {
        atomicStore(&telemetry[${CLOCK_STATE}u], 3u);
        return;
      }
    }
  }
  if (localInvocation != 0u) { return; }

  let logical = workgroup.x - 1u;
  let base = recordBase(logical);
  atomicStore(&telemetry[base + ${RECORD_START_STATE}u], atomicLoad(&telemetry[${CLOCK_STATE}u]));
  atomicStore(&telemetry[base + ${RECORD_START_TICK}u], atomicLoad(&telemetry[${CLOCK}u]));
  atomicStore(&telemetry[base + ${RECORD_ENTRY_TICKET}u], atomicAdd(&telemetry[${SEQUENCE}u], 1u));
  atomicStore(&telemetry[base + ${RECORD_CHECKSUM}u], payload(logical));
  atomicStore(&telemetry[base + ${RECORD_EXIT_TICKET}u], atomicAdd(&telemetry[${SEQUENCE}u], 1u));
  atomicStore(&telemetry[base + ${RECORD_END_TICK}u], atomicLoad(&telemetry[${CLOCK}u]));
  atomicStore(&telemetry[base + ${RECORD_END_STATE}u], atomicLoad(&telemetry[${CLOCK_STATE}u]));
  atomicAdd(&telemetry[${COMPLETED}u], 1u);
}
`;

await acquireWebGPUExclusiveLock("atomic-clock-experiment", "tools/experiment-webgpu-atomic-clock.ts");
try {
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${backend}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error(`No WebGPU adapter for backend ${backend}`);
  if (!adapter.features.has("timestamp-query")) throw new Error("Atomic clock experiment requires timestamp-query calibration");
  const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });

  const telemetryBytes = (HEADER_WORDS + payloadWorkgroups * RECORD_WORDS) * Uint32Array.BYTES_PER_ELEMENT;
  const telemetry = device.createBuffer({
    label: "Atomic clock telemetry",
    size: telemetryBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const telemetryRead = device.createBuffer({
    label: "Atomic clock telemetry readback",
    size: telemetryBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const timestampQueries = device.createQuerySet({ type: "timestamp", count: 2 });
  const timestampResolve = device.createBuffer({
    label: "Atomic clock timestamp resolve",
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const timestampRead = device.createBuffer({
    label: "Atomic clock timestamp readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const module = device.createShaderModule({ label: "Experimental atomic shader clock", code: shaderSource });
  const compilation = await module.getCompilationInfo();
  const shaderErrors = compilation.messages.filter((message) => message.type === "error");
  if (shaderErrors.length > 0) {
    throw new Error(`Atomic clock shader failed: ${shaderErrors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`).join(" | ")}`);
  }
  const baselinePipeline = await device.createComputePipelineAsync({
    label: "Atomic clock baseline payload",
    layout: "auto",
    compute: { module, entryPoint: "baseline" },
  });
  const clockedPipeline = await device.createComputePipelineAsync({
    label: "Atomic clock instrumented payload",
    layout: "auto",
    compute: { module, entryPoint: "clocked" },
  });
  const bindGroupFor = (pipeline: GPUComputePipeline, label: string) => device.createBindGroup({
    label,
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: telemetry } }],
  });
  const baselineBindGroup = bindGroupFor(baselinePipeline, "Atomic clock baseline telemetry binding");
  const clockedBindGroup = bindGroupFor(clockedPipeline, "Atomic clock instrumented telemetry binding");

  const run = async (
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    dispatchCount: number,
  ): Promise<RunResult> => {
    const encoder = device.createCommandEncoder({ label: "Atomic clock experiment" });
    encoder.clearBuffer(telemetry);
    const pass = encoder.beginComputePass({
      label: "Atomic clock measured pass",
      timestampWrites: {
        querySet: timestampQueries,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      },
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchCount);
    pass.end();
    encoder.resolveQuerySet(timestampQueries, 0, 2, timestampResolve, 0);
    encoder.copyBufferToBuffer(timestampResolve, 0, timestampRead, 0, 16);
    encoder.copyBufferToBuffer(telemetry, 0, telemetryRead, 0, telemetryBytes);
    device.queue.submit([encoder.finish()]);
    await Promise.all([
      timestampRead.mapAsync(GPUMapMode.READ),
      telemetryRead.mapAsync(GPUMapMode.READ),
    ]);
    const timestamps = new BigUint64Array(timestampRead.getMappedRange().slice(0));
    const words = new Uint32Array(telemetryRead.getMappedRange().slice(0));
    timestampRead.unmap();
    telemetryRead.unmap();
    const begin = timestamps[0] ?? 0n;
    const end = timestamps[1] ?? 0n;
    if (begin === 0n || end < begin) throw new Error(`Invalid hardware timestamps: ${begin}, ${end}`);
    return { pass_ms: Number(end - begin) / 1e6, words };
  };

  // Warm both pipelines before collecting paired samples.
  await run(baselinePipeline, baselineBindGroup, payloadWorkgroups);
  await run(clockedPipeline, clockedBindGroup, payloadWorkgroups + 1);

  const baselineTimes: number[] = [];
  const clockTrials: ClockTrial[] = [];
  for (let trial = 0; trial < trials; trial += 1) {
    // Alternate order to reduce directional thermal/clock bias.
    if ((trial & 1) === 0) {
      baselineTimes.push((await run(baselinePipeline, baselineBindGroup, payloadWorkgroups)).pass_ms);
      clockTrials.push(decodeClockTrial(await run(clockedPipeline, clockedBindGroup, payloadWorkgroups + 1)));
    } else {
      clockTrials.push(decodeClockTrial(await run(clockedPipeline, clockedBindGroup, payloadWorkgroups + 1)));
      baselineTimes.push((await run(baselinePipeline, baselineBindGroup, payloadWorkgroups)).pass_ms);
    }
  }

  const baseline = summarize(baselineTimes);
  const instrumented = summarize(clockTrials.map((trial) => trial.pass_ms));
  const tickRate = summarize(clockTrials.map((trial) => trial.tickRatePer_ms));
  const advancing = summarize(clockTrials.map((trial) => trial.advancingFraction));
  const startVisible = summarize(clockTrials.map((trial) => trial.startVisibleFraction));
  const runningAtStart = summarize(clockTrials.map((trial) => trial.clockRunningAtStartFraction));
  const allComplete = clockTrials.every((trial) => trial.clockState === 2
    && trial.completedWorkgroups === payloadWorkgroups
    && trial.sequenceTickets === payloadWorkgroups * 2
    && trial.checksumSamplesCorrect);
  const overheadPercent = baseline.median > 0 ? (instrumented.median / baseline.median - 1) * 100 : 0;
  const usable = allComplete
    && advancing.minimum >= 0.90
    && runningAtStart.minimum >= 0.90
    && tickRate.coefficientOfVariation <= maximumTickRateVariation
    && overheadPercent <= maximumInstrumentationOverheadPercent;
  const verdictReasons = [
    ...(!allComplete ? ["one or more trials timed out, lost workgroups/tickets, or failed checksum validation"] : []),
    ...(advancing.minimum < 0.90 ? [`minimum advancing coverage ${(advancing.minimum * 100).toFixed(1)}% < 90%`] : []),
    ...(runningAtStart.minimum < 0.90 ? [`minimum clock-at-entry coverage ${(runningAtStart.minimum * 100).toFixed(1)}% < 90%`] : []),
    ...(tickRate.coefficientOfVariation > maximumTickRateVariation
      ? [`tick-rate CV ${(tickRate.coefficientOfVariation * 100).toFixed(1)}% > ${(maximumTickRateVariation * 100).toFixed(0)}%`] : []),
    ...(overheadPercent > maximumInstrumentationOverheadPercent
      ? [`median instrumentation overhead ${overheadPercent.toFixed(1)}% > ${maximumInstrumentationOverheadPercent}%`] : []),
  ];

  console.log(JSON.stringify({
    experiment: "webgpu-atomic-shader-clock",
    adapter: Object.fromEntries(Object.entries(adapter.info)),
    backend,
    configuration: {
      payloadWorkgroups,
      payloadWorkgroupSize,
      payloadIterations,
      maximumClockTicks,
      trials,
      acceptance: {
        minimumAdvancingFraction: 0.90,
        minimumClockRunningAtStartFraction: 0.90,
        maximumTickRateVariation,
        maximumInstrumentationOverheadPercent,
      },
    },
    baselinePass_ms: baseline,
    instrumentedPass_ms: instrumented,
    instrumentationOverhead_percent: overheadPercent,
    clock: {
      tickRatePer_ms: tickRate,
      startVisibleFraction: startVisible,
      advancingFraction: advancing,
      clockRunningAtStartFraction: runningAtStart,
      trialDetails: clockTrials,
    },
    verdict: {
      usableForExperimentalReconstruction: usable,
      reasons: usable ? ["bounded atomic ticker overlapped the payload consistently on this adapter"] : verdictReasons,
      caveat: "Logical workgroup timing only; no physical core identity or portable scheduler guarantee.",
    },
  }, null, 2));

  timestampQueries.destroy();
  timestampResolve.destroy();
  timestampRead.destroy();
  telemetry.destroy();
  telemetryRead.destroy();
  device.destroy();
} finally {
  await releaseWebGPUExclusiveLock();
}
