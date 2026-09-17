import assert from "node:assert/strict";

/**
 * Native-only wall timers published by the geometric-remap receipt.
 *
 * Keep this list explicit: a newly optimized arm must not silently omit a
 * stage and then appear faster because the harness treated it as zero.
 */
export const cellwiseStageTimingKeys = [
  "baseFieldBuild",
  "receiverBandClosure",
  "streamfunctionExtension",
  "harmonicFill",
  "diagnostics",
  "trace",
  "geometryRefinement",
  "gather",
  "commit",
  "other",
  "totalRemap",
] as const;

export const worldStageTimingKeys = [
  "totalAdvance",
  "fieldBuild",
  "primaryPressure",
  "supportPlanningTransfer",
  "postSupportPressure",
  "transport",
  "postTransport",
  "resolutionPublication",
  "other",
] as const;

export type CellwiseStageTimingKey = typeof cellwiseStageTimingKeys[number];
export type CellwiseStageTimings = { available: boolean } & Record<
  CellwiseStageTimingKey,
  number
>;
export type WorldStageTimingKey = typeof worldStageTimingKeys[number];
export type WorldStageTimings = { available: boolean } & Record<
  WorldStageTimingKey,
  number
>;

function requireTimingObject<Key extends string>(
  timings: Record<string, unknown> | undefined,
  keys: readonly Key[],
  frame: number,
  scope: string,
): { available: boolean } & Record<Key, number> {
  assert.ok(timings && typeof timings === "object",
    `frame ${frame}: missing ${scope} stage timings`);
  assert.equal(timings.available, true,
    `frame ${frame}: ${scope} stage timings are unavailable in the native arm`);
  for (const key of keys) {
    const value: unknown = timings[key];
    assert.equal(typeof value, "number", `frame ${frame}: missing ${scope}.${key} timing`);
    assert.ok(Number.isSafeInteger(value as number) && (value as number) >= 0,
      `frame ${frame}: invalid ${scope}.${key} timing ${String(value)}`);
  }
  return timings as { available: boolean } & Record<Key, number>;
}

export function requireNativeWorldStageTimings(
  receipt: Record<string, any>,
  frame: number,
): WorldStageTimings {
  const timings = requireTimingObject(receipt.stageTimings, worldStageTimingKeys, frame, "world");
  const accounted = worldStageTimingKeys
    .filter(key => key !== "totalAdvance")
    .reduce((sum, key) => sum + timings[key], 0);
  assert.equal(accounted, timings.totalAdvance,
    `frame ${frame}: World timing split ${accounted} != totalAdvance ${timings.totalAdvance}`);
  return timings;
}

export function requireNativeCellwiseStageTimings(
  remap: Record<string, any>,
  frame: number,
): CellwiseStageTimings {
  const timings = requireTimingObject(
    remap.timings,
    cellwiseStageTimingKeys,
    frame,
    "cellwiseRemap",
  );
  const accounted = cellwiseStageTimingKeys
    .filter(key => key !== "totalRemap")
    .reduce((sum, key) => sum + timings[key], 0);
  assert.equal(accounted, timings.totalRemap,
    `frame ${frame}: cellwise timing split ${accounted} != totalRemap ${timings.totalRemap}`);
  return timings;
}

function summarizeTimingFrames<Key extends string>(
  frames: Array<Record<Key, number>>,
  keys: readonly Key[],
): Record<string, unknown> {
  assert.ok(frames.length > 0, "no measured cellwise frames");
  return Object.fromEntries(keys.map(key => {
    const samples = frames.map(frame => frame[key]);
    const ordered = [...samples].sort((a, b) => a - b);
    const totalNanoseconds = samples.reduce((sum, value) => sum + value, 0);
    return [key, {
      totalNanoseconds,
      meanNanoseconds: totalNanoseconds / samples.length,
      p95Nanoseconds: ordered[Math.ceil(0.95 * ordered.length) - 1],
      maximumNanoseconds: Math.max(...samples),
    }];
  }));
}

export function summarizeNativeStageTimings(
  worldFrames: WorldStageTimings[],
  cellwiseFrames?: CellwiseStageTimings[],
): Record<string, unknown> {
  assert.ok(worldFrames.length > 0, "no measured world frames");
  if (cellwiseFrames) {
    assert.equal(cellwiseFrames.length, worldFrames.length,
      "world and cellwise timing frame counts differ");
    for (let index = 0; index < worldFrames.length; index += 1) {
      assert.ok(cellwiseFrames[index]!.totalRemap <= worldFrames[index]!.transport,
        `frame ${index + 1}: remap total exceeds World transport stage`);
    }
  }
  return {
    clock: "native-monotonic-wall",
    unit: "nanoseconds",
    frames: worldFrames.length,
    world: summarizeTimingFrames(worldFrames, worldStageTimingKeys),
    cellwiseRemap: cellwiseFrames
      ? summarizeTimingFrames(cellwiseFrames, cellwiseStageTimingKeys)
      : null,
    transportOutsideRemap: cellwiseFrames
      ? summarizeTimingFrames(
        worldFrames.map((frame, index) => ({
          nanoseconds: frame.transport - cellwiseFrames[index]!.totalRemap,
        })),
        ["nanoseconds"],
      ).nanoseconds
      : null,
  };
}
