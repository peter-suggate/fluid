import assert from "node:assert/strict";
import test from "node:test";
import {
  FLYOUT_EDGE_MARGIN_PX,
  FLYOUT_TRANSPORT_KEEPOUT_PX,
  resolveFlyoutPlacement,
} from "../components/anchored-flyout";

/** The field picker's shape: the authored 12px gap, anchored 60% down its own height. */
const fieldFlyout = (leftFraction: number, topFraction: number, overrides: Partial<{
  panelWidth: number; panelHeight: number; containerWidth: number; containerHeight: number;
}> = {}) => resolveFlyoutPlacement({
  leftFraction, topFraction, gap: 12, originY: 0.6, offsetY: 0,
  panelWidth: 172, panelHeight: 400, containerWidth: 1200, containerHeight: 700, ...overrides,
});

const margin = FLYOUT_EDGE_MARGIN_PX;
const keepout = FLYOUT_TRANSPORT_KEEPOUT_PX;

test("an anchor with room keeps the authored offset", () => {
  const placement = fieldFlyout(0.25, 0.5);
  assert.equal(placement.left, 0.25 * 1200 + 12);
  assert.equal(placement.top, 0.5 * 700 - 0.6 * 400);
});

test("an anchor near the right edge flips the panel to the anchor's other side", () => {
  const placement = fieldFlyout(0.98, 0.5);
  // Right would end at 1360 in a 1200-wide shell that clips.
  assert.equal(placement.left, 0.98 * 1200 - 12 - 172);
  assert.ok(placement.left + 172 <= 1200 - margin);
});

test("a merely tight fit does not flip", () => {
  // The panel's right edge lands exactly on the margin: still the authored side,
  // so a camera creeping toward the edge slides rather than jumping across.
  const leftFraction = (1200 - margin - 172 - 12) / 1200;
  assert.equal(fieldFlyout(leftFraction, 0.5).left, leftFraction * 1200 + 12);
});

test("neither side fitting clamps inside the shell rather than flipping off it", () => {
  const placement = fieldFlyout(0.5, 0.5, { containerWidth: 200 });
  assert.equal(placement.left, 200 - margin - 172);
  assert.ok(placement.left >= margin);
});

test("a panel wider than its shell still starts on screen", () => {
  assert.equal(fieldFlyout(0.5, 0.5, { containerWidth: 100 }).left, margin);
});

test("the vertical origin is clamped at both edges", () => {
  assert.equal(fieldFlyout(0.25, 0.01).top, margin);
  // The bottom edge a panel stops at is the transport's band, not the shell's.
  assert.equal(fieldFlyout(0.25, 0.99).top, 700 - keepout - 400);
});

test("a panel anchored at the bottom edge stops above the transport", () => {
  const placement = fieldFlyout(0.25, 0.99, { panelHeight: 200 });
  assert.equal(placement.top + 200, 700 - keepout);
});

test("the height cap clears both the top margin and the transport, and ignores the panel's own size", () => {
  const short = fieldFlyout(0.25, 0.5, { panelHeight: 120 });
  const tall = fieldFlyout(0.25, 0.5, { panelHeight: 900 });
  assert.equal(short.maxHeight, 700 - keepout - margin);
  assert.equal(tall.maxHeight, 700 - keepout - margin);
  // A panel taller than the cap scrolls; its top is still inside the shell.
  assert.equal(tall.top, margin);
});

test("the selection chip's own offsets survive placement", () => {
  // gap 16, anchored at its top edge, nudged 14px up — the authored chip offset.
  const chipFlyout = (panelHeight: number) => resolveFlyoutPlacement({
    leftFraction: 0.5, topFraction: 0.5, gap: 16, originY: 0, offsetY: -14,
    panelWidth: 178, panelHeight, containerWidth: 1200, containerHeight: 700,
  });
  const placement = chipFlyout(200);
  assert.equal(placement.left, 616);
  assert.equal(placement.top, 336);
  // Until the panel is long enough to reach the transport, which outranks the
  // authored nudge: an expanded tank is a column of groups, not a chip.
  assert.equal(chipFlyout(300).top, 700 - keepout - 300);
});
