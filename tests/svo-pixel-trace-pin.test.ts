import assert from "node:assert/strict";
import test from "node:test";
import {
  resolveSvoPixelTracePin, settleSvoPixelTracePin, SVO_PIXEL_TRACE_PIN_PATIENCE, svoPixelTracePinClick,
  svoPixelTracePinnedAim, type SvoPixelTracePin,
} from "../lib/svo/features/diagnostics/svo-pixel-trace";

const click = (normalizedX: number, revision = 10, cameraKey = "view-a"): SvoPixelTracePin =>
  ({ phase: "requested", request: svoPixelTracePinClick({ normalizedX, normalizedY: 0.5, cameraKey, revision }).request });

test("a click freezes its own pixel once that pixel comes back, not before", () => {
  let pin = click(0.25);
  const ask = (answered: boolean, revision: number) =>
    pin?.phase === "requested" ? resolveSvoPixelTracePin(pin.request, { answered, cameraKey: "view-a", probeCanAnswer: true, revision }) : undefined;

  pin = settleSvoPixelTracePin(pin, { resolution: ask(false, 11), pinned: false });
  assert.equal(pin?.phase, "requested", "the readback present at the click is a neighbouring ray");
  assert.equal(svoPixelTracePinnedAim(pin, true), null, "nothing is frozen while the request waits");

  pin = settleSvoPixelTracePin(pin, { resolution: ask(true, 12), pinned: false });
  assert.deepEqual(pin, { phase: "pinned", aim: { normalizedX: 0.25, normalizedY: 0.5, cameraKey: "view-a" } });
  assert.deepEqual(svoPixelTracePinnedAim(pin, true), { normalizedX: 0.25, normalizedY: 0.5, cameraKey: "view-a" });
});

test("a request the probe will never answer is dropped, and the probe follows the pointer again", () => {
  const moved = click(0.25);
  assert.equal(moved?.phase, "requested");
  if (moved?.phase !== "requested") return;
  const orbited = resolveSvoPixelTracePin(moved.request, { answered: true, cameraKey: "view-b", probeCanAnswer: true, revision: 11 });
  assert.equal(settleSvoPixelTracePin(moved, { resolution: orbited, pinned: false }), null);
  const exhausted = resolveSvoPixelTracePin(moved.request,
    { answered: false, cameraKey: "view-a", probeCanAnswer: true, revision: 10 + SVO_PIXEL_TRACE_PIN_PATIENCE });
  assert.equal(settleSvoPixelTracePin(moved, { resolution: exhausted, pinned: false }), null);
});

test("a frozen aim lives exactly as long as the pin switch", () => {
  const pinned: SvoPixelTracePin = { phase: "pinned", aim: { normalizedX: 0.1, normalizedY: 0.2, cameraKey: "view-a" } };
  assert.equal(settleSvoPixelTracePin(pinned, { pinned: true }), pinned);
  assert.equal(svoPixelTracePinnedAim(pinned, false), null, "an aim is never traced with the switch off");
  assert.equal(settleSvoPixelTracePin(pinned, { pinned: false }), null, "unpinned, the aim is gone for good");
  // The switch coming back on does not revive an aim that was dropped.
  assert.equal(svoPixelTracePinnedAim(settleSvoPixelTracePin(pinned, { pinned: false }), true), null);
});

test("a click while pinned is a new request, not a re-use of the old aim", () => {
  const replaced = click(0.75, 20);
  assert.equal(replaced?.phase, "requested");
  // The switch is released before the new aim is recorded; the request survives that.
  assert.equal(settleSvoPixelTracePin(replaced, { pinned: false }), replaced);
});
