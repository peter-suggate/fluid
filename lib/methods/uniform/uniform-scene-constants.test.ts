import assert from "node:assert/strict";
import test from "node:test";
import { uniformSceneConstantsWGSL } from "./uniform-scene-constants";

const shader = (x: number, h: number) => `
@group(0) @binding(1) var<uniform> umCounts:vec4u;
const UM_D=vec3u(${x}u,32u,16u);const UM_T=UM_D/4u;
const UM_TILES:u32=${x * 32 * 16 / 64}u;
const UM_PD:vec3u=(UM_T+vec3u(3u))/4u;
const UM_PAGES:u32=UM_PD.x*UM_PD.y*UM_PD.z;
const UM_DETAIL:u32=4u*UM_TILES;
const UM_H=vec3f(${h},${h},${h});
fn sample()->f32{return f32(UM_D.x+UM_DETAIL+UM_PAGES+umCounts.x)*UM_H.x;}
`;

test("integer dimensions specialize addressing while physical spacing stays reusable", () => {
  const small = uniformSceneConstantsWGSL(shader(32, 0.05));
  assert.equal(small, uniformSceneConstantsWGSL(shader(32, 0.025)));
  const large = uniformSceneConstantsWGSL(shader(64, 0.025));
  assert.notEqual(small, large);
  assert.match(small, /fn UM_D\(\)->vec3u\{return vec3u\(32u,32u,16u\);\}/);
  assert.match(large, /fn UM_D\(\)->vec3u\{return vec3u\(64u,32u,16u\);\}/);
  assert.match(small, /fn UM_TILES\(\)->u32\{return 256u;\}/);
  assert.match(large, /fn UM_TILES\(\)->u32\{return 512u;\}/);
  assert.match(small, /return umCounts.spacing.xyz/);
  assert.match(small, /fn UM_DETAIL\(\)->u32\{return 4u\*UM_TILES\(\);\}/);
  assert.match(small, /UM_PAGES\(\)\+umCounts.work.x/);
});

test("compile-time array extents stay unchanged until their layouts are padded", () => {
  const code = shader(32, 0.05) + "var<workgroup> values:array<f32,UM_TILES>;";
  assert.equal(uniformSceneConstantsWGSL(code), code);
});

test("namespaced ownerships retain separate specialized integer geometry", () => {
  const code = shader(32, 0.05).replace(/\b(umCounts|UM_\w+)\b/g, "old$1");
  const rewritten = uniformSceneConstantsWGSL(code);
  assert.match(rewritten, /fn oldUM_D\(\)->vec3u\{return vec3u\(32u,32u,16u\);\}/);
  assert.match(rewritten, /return oldumCounts.spacing.xyz/);
  assert.match(rewritten, /fn oldUM_T\(\)->vec3u\{return oldUM_D\(\)\/4u;\}/);
});

test("the geometry struct follows leading directives", () => {
  const rewritten = uniformSceneConstantsWGSL("enable subgroups;\n" + shader(32, 0.05));
  assert.match(rewritten, /^enable subgroups;\s*struct UMSceneCounts/);
});
