import assert from "node:assert/strict";
import test from "node:test";
import { octreeLosassoAdaptivePhiTransportCorrectionEnabled }
  from "../lib/webgpu-octree-losasso-adaptive-phi";
import {
  octreeLosassoAdaptivePhiCorrectionWGSL,
  octreeLosassoAdaptivePhiReverseBacktraceWGSL,
} from "../lib/webgpu-octree-losasso-adaptive-phi.wgsl";

const f32 = Math.fround;
const quantize = (value: number): number => f32(Math.round(f32(value * 65536)) / 65536);
const bits = (value: number): number => new Uint32Array(new Float32Array([value]).buffer)[0]!;

test("MacCormack diagnostic arm defaults on and explicitly selects simple SL", () => {
  assert.equal(octreeLosassoAdaptivePhiTransportCorrectionEnabled({}), true);
  assert.equal(octreeLosassoAdaptivePhiTransportCorrectionEnabled({
    FLUID_ADAPTIVE_PHI_MACCORMACK: "1",
  }), true);
  assert.equal(octreeLosassoAdaptivePhiTransportCorrectionEnabled({
    FLUID_ADAPTIVE_PHI_MACCORMACK: "0",
  }), false);
});

const boundedCorrection = (prior: number, predictor: number, reversed: number,
  donorMinimum: number, donorMaximum: number, inflow = false): number => {
  if (inflow) return quantize(predictor);
  return quantize(Math.min(donorMaximum, Math.max(donorMinimum,
    f32(predictor + f32(.5 * f32(prior - reversed))))));
};

test("compact MacCormack is exact for a translated linear scalar", () => {
  const field = (x: number) => f32(.3125 * x - .1875);
  const x = 3.25;
  const displacement = .375;
  const predictor = field(x - displacement);
  const reversed = field(x);
  const donorMinimum = Math.min(field(2), field(3));
  const donorMaximum = Math.max(field(2), field(3));
  assert.equal(boundedCorrection(field(x), predictor, reversed,
    donorMinimum, donorMaximum), quantize(field(x - displacement)));
});

test("compact MacCormack clamp cannot overshoot its backward donor leaf", () => {
  assert.equal(boundedCorrection(.9, .8, -.8, -.25, .75), quantize(.75));
  assert.equal(boundedCorrection(-.9, -.8, .8, -.75, .25), quantize(-.75));
});

test("compact MacCormack is D4-equivariant and zero-dt identical", () => {
  const old = quantize(-.123456);
  assert.equal(bits(boundedCorrection(old, old, old, -.5, .5)), bits(old));

  const affine = ([x, y, z]: readonly number[]) => f32(.25 * x - .125 * y + .375 * z);
  const q = [1.25, .5, 2.75] as const;
  const swapXZ = ([x, y, z]: readonly number[]) => [z, y, x] as const;
  const reflectedField = ([x, y, z]: readonly number[]) => affine([z, y, x]);
  const original = boundedCorrection(affine(q), affine(q), affine(q), -2, 2);
  const transformed = boundedCorrection(reflectedField(swapXZ(q)),
    reflectedField(swapXZ(q)), reflectedField(swapXZ(q)), -2, 2);
  assert.equal(bits(original), bits(transformed));
});

test("compact MacCormack keeps inflow authoritative and fails closed at invalid leaves", () => {
  assert.equal(boundedCorrection(.7, -.2, .4, -.1, .1, true), quantize(-.2));
  const correction = octreeLosassoAdaptivePhiCorrectionWGSL.replace(/\s+/g, "");
  assert.match(correction,
    /fndonorBounds\(leaf:u32,b:u32\)->vec3f\{if\(leaf>=leafCount\(\)\)\{returnvec3f\(0\);\}/);
  assert.match(correction,
    /if\(reversed\.valid==0u\|\|bounds\.z!=1\.\)\{atomicAdd\(&control\[19\],1u\);atomicOr\(&control\[12\],ERR_SAMPLE\);return;\}/);
  assert.match(correction,
    /if\(p\.inflowVelocityStrength\.w>0\.&&finite\(source\)\)\{corrected=predictor;\}/);
  const reverse = octreeLosassoAdaptivePhiReverseBacktraceWGSL.replace(/\s+/g, "");
  assert.match(reverse, /letboundedMid=clamp\(mid,vec3f\(0\),vec3f\(p\.dims\.xyz\)\)/,
    "reverse RK2 boundary sampling must remain inside the compact domain");
});
