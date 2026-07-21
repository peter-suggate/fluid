import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { SVO_CONTACT_VISIBILITY_CONTRACT } from "../lib/svo-contact-visibility";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

const drySceneSource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");

function shaderFunction(name: string, nextName: string): string {
  const start = svoDrySceneShader.indexOf(`fn ${name}(`);
  const end = svoDrySceneShader.indexOf(`fn ${nextName}(`, start);
  assert.ok(start >= 0 && end > start, `${name} must precede ${nextName}`);
  return svoDrySceneShader.slice(start, end);
}

test("contact visibility defaults on for beautiful presentation with a branch before secondary traversal", () => {
  assert.equal(SVO_CONTACT_VISIBILITY_CONTRACT.enabledByDefault, true);
  const contact = shaderFunction("dryContactVisibility", "dryEnvironment");
  const publicGate = contact.indexOf("if((dry.materialPublication.w&8u)==0u){return vec3f(1.0);}");
  const gate = contact.indexOf("if((dry.materialPublication.w&1u)==0u){return vec3f(1.0);}");
  const trace = contact.indexOf("svoTraceVisibility(");
  assert.ok(publicGate >= 0 && gate > publicGate && trace > gate,
    "the public option and exact-fallback gates must return before secondary SVO work");
  assert.match(drySceneSource, /this\.lightingOptions\.ambientOcclusionEnabled && scene\.contactVisibilityEnabled !== false/,
    "the user option enables AO unless a scene explicitly lacks the capability");
});

test("contact traversal has a fixed low sample and per-sample work budget", () => {
  const contact = shaderFunction("dryContactVisibility", "dryEnvironment");
  assert.match(contact, new RegExp(
    `sampleIndex<${SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount}u`,
  ));
  assert.match(contact, new RegExp(
    `SvoVisibilityBudget\\(${SVO_CONTACT_VISIBILITY_CONTRACT.maximumNodeVisitsPerSample}u,`
    + `${SVO_CONTACT_VISIBILITY_CONTRACT.maximumLeafVisitsPerSample}u,`
    + `${SVO_CONTACT_VISIBILITY_CONTRACT.maximumWorkItemsPerSample}u,`
    + `${SVO_CONTACT_VISIBILITY_CONTRACT.maximumIntersectionsPerSample}u\\)`,
  ));
  assert.match(contact, /result\.status==SVO_VIS_STATUS_INVALID\|\|result\.status==SVO_VIS_STATUS_EXHAUSTED[^]*return vec3f\(0\.0\)/,
    "invalid or exhausted secondary work must fail the complete estimate closed");
  assert.match(contact, /clamp\(visibility\/f32\(2\),vec3f\(0\.0\),vec3f\(1\.0\)\)/,
    "contact visibility must not add indirect energy");
});

test("contact radius, bias, and directions are finite, edge-aware, and temporally stable", () => {
  const radius = shaderFunction("dryContactVisibilityRadius", "dryContactVisibilityDirection");
  const direction = shaderFunction("dryContactVisibilityDirection", "dryContactVisibility");
  const contact = shaderFunction("dryContactVisibility", "dryEnvironment");
  assert.match(radius, /min\(sceneScale\*0\.06,max\(cellScale\*6\.0,sceneScale\*0\.01\)\)/);
  assert.match(contact, /select\(0\.025,0\.05,featureId!=SVO_FEATURE_SMOOTH\)/,
    "hard features need the larger self-intersection bias without changing their normal");
  assert.match(direction, /\(featureId&1u\)!=0u/);
  assert.match(direction, /normalize\(geometricNormal\+signValue\*\(\.55\*tangent\+\.2\*bitangent\)\)/);
  assert.doesNotMatch(`${direction}${contact}`, /uniforms\.(?:time|frame)|random|hash|noise/i,
    "the two directions must not shimmer with frame-varying noise");
});

test("contact visibility attenuates indirect diffuse only and adds no storage binding", () => {
  const shade = shaderFunction("shadeDryOpaque", "shadeThinGlass");
  assert.match(shade, /let contactVisibility=dryContactVisibility\(position,hit\.normal,hit\.featureId,hit\.ownerId\)/);
  assert.match(shade, /let diffuseEnvironment=[^;]*\*contactVisibility\/UNIFIED_PI/);
  assert.match(shade, /let specularEnvironment=dryEnvironment\(reflected,surface\.roughness\)\*fresnel/);
  assert.match(shade, /return max\(surface\.emissive\+diffuseEnvironment\+specularEnvironment\+direct,vec3f\(0\.0\)\)/);
  assert.doesNotMatch(shade, /(?:surface\.emissive|specularEnvironment|direct)\s*\*\s*contactVisibility/);

  const storageBindings = [...svoDrySceneShader.matchAll(/@group\(0\) @binding\((\d+)\) var<storage/g)]
    .map((match) => Number(match[1]));
  assert.equal(storageBindings.length, 10);
  assert.equal(new Set(storageBindings).size, storageBindings.length);
});
