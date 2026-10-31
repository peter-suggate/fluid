import assert from "node:assert/strict";
import test from "node:test";
import { UNIFORM_DETAIL_TWIN_BINDING, uniformDetailDenseShader } from "../lib/methods/uniform/uniform-detail-fields";

// A t=0 kernel of the dense rewrite's shape: one stored and one loaded field.
const kernel = /* wgsl */ `
@group(0) @binding(4) var volumeIn:texture_3d<f32>;
@group(0) @binding(5) var volumeOut:texture_storage_3d<r32float,write>;
@compute @workgroup_size(4,4,4) fn main(@builtin(global_invocation_id) g:vec3u){
  let id=vec3i(g);
  textureStore(volumeOut,id,textureLoad(volumeIn,id,0));
}`;
const classes = { volumeIn: "cell", volumeOut: "cell" } as const;

test("the dense t=0 rewrite stores each canonical texel through the stored field's twin, before the slotted h store", () => {
  const code = uniformDetailDenseShader(kernel, classes);
  assert.match(code, new RegExp(`@group\\(0\\) @binding\\(${5 + UNIFORM_DETAIL_TWIN_BINDING}\\) var udb_volumeOut:texture_storage_3d<r32float,write>;`));
  // Dense: a loaded field has no twin; its loads read the bound texture.
  assert.doesNotMatch(code, /udb_volumeIn/);
  const store = /fn udStore_volumeOut\(q:vec3i,value:vec4f\)\{([\s\S]*?)\n\}/.exec(code)?.[1];
  assert.ok(store, "the store accessor");
  // The base store comes first: a slotless block's h store returns before anything after it.
  const twin = store.indexOf("textureStore(udb_volumeOut"), h = store.indexOf("textureStore(volumeOut");
  assert.ok(twin >= 0 && h > twin, "the twin store precedes the h store");
});

test("under a sparse domain each loaded field binds its base twin, which a slotless block's load reads", () => {
  const code = uniformDetailDenseShader(kernel, classes, true);
  assert.match(code, new RegExp(`@group\\(0\\) @binding\\(${4 + UNIFORM_DETAIL_TWIN_BINDING}\\) var udb_volumeIn:texture_3d<f32>;`));
  const load = /fn udLoad_volumeIn\(q:vec3i\)->vec4f\{([\s\S]*?)\n\}/.exec(code)?.[1];
  assert.ok(load, "the load accessor");
  // One resolution serves the slotless test and the h texel.
  assert.match(load, /let s=umCellSlotOf\(q\);if\(umBlockStored\(s\.x\)&&s\.y==0u\)\{return textureLoad\(udb_volumeIn,/);
  assert.match(load, /textureLoad\(volumeIn,umCellTexelAt\(q,s\.y\),0\)/);
});
