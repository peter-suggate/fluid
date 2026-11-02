import assert from "node:assert/strict";
import test from "node:test";

/**
 * A capture is one phase at a time. These are the regressions the
 * recorder-or-output checks had: a second START during a capture opened a
 * second one over it, and a second STOP stopped the same capture twice.
 */
let stops = 0;
class FakeMediaRecorder {
  static isTypeSupported() { return true; }
  state: "inactive" | "recording" | "paused" = "inactive";
  mimeType = "video/webm";
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onerror: (() => void) | null = null;
  onstop: (() => void) | null = null;
  start() { this.state = "recording"; }
  pause() { this.state = "paused"; }
  resume() { this.state = "recording"; }
  stop() {
    stops += 1;
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["frame"]) });
    queueMicrotask(() => this.onstop?.());
  }
}
const canvas = { width: 64, height: 64, captureStream: () => ({ getTracks: () => [] }) };
Object.assign(globalThis, {
  window: globalThis,
  document: { querySelector: () => canvas },
  MediaRecorder: FakeMediaRecorder,
});
URL.createObjectURL = () => "blob:capture";
URL.revokeObjectURL = () => {};

const { simulationRecording } = await import("../lib/core/simulation/recording");

test("a second start during a capture is refused, and a second stop does nothing", async () => {
  assert.equal(await simulationRecording.start(0), true);
  assert.equal(simulationRecording.capturing, "wall-clock");
  assert.equal(await simulationRecording.start(0), false);
  simulationRecording.stop(1);
  simulationRecording.stop(1);
  assert.equal(stops, 1);
  await new Promise(resolve => setTimeout(resolve));
  assert.equal(simulationRecording.capturing, "idle");
  assert.equal(await simulationRecording.start(1), true);
  simulationRecording.stop(2);
  await new Promise(resolve => setTimeout(resolve));
  assert.equal(stops, 2);
});
