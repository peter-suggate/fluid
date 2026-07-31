import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { SVO_CONTACT_VISIBILITY_CONTRACT } from "../lib/svo-contact-visibility";
import {
  SVO_DRY_SCENE_BINDING_CONTRACT,
  SVO_DRY_SCENE_MOVING_AO_CONE_SAMPLES,
  SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES,
  svoDrySceneShader,
} from "../lib/webgpu-svo-dry-scene";

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
  assert.match(contact,
    /SvoVisibilityBudget\(dry\.tuningCounts1\.w,dry\.tuningCounts2\.x,dry\.tuningCounts2\.y,dry\.tuningCounts2\.z\)/,
    "contact traversal must consume the live bounded render-tuning budgets");
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

test("cone AO drops to a single cone while the camera is changing, never to zero", () => {
  const contact = shaderFunction("dryContactVisibility", "dryEnvironment");
  // One cone while moving, not two and not zero. Disabling AO outright is
  // cheaper still, but its error lands in contiguous patches on contact
  // shading rather than as diffuse noise, so every settle would pop those
  // regions darker; the measured comparison is on the constant itself.
  assert.equal(SVO_DRY_SCENE_MOVING_AO_CONE_SAMPLES, 1);
  assert.equal(SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES, 4);
  assert.ok(SVO_DRY_SCENE_MOVING_AO_CONE_SAMPLES >= 1,
    "AO must stay present while moving so settling changes noise, not whether ambient exists");
  assert.ok(SVO_DRY_SCENE_MOVING_AO_CONE_SAMPLES < SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES);
  assert.match(contact,
    /coneSampleCount=select\(dry\.tuningCounts1\.z,dry\.tuningCounts1\.y,uniforms\.viewport\.w>=-1\.0\)/,
    "moving and settled cone counts must remain live GPU tuning state");
  assert.match(contact, new RegExp(
    `sampleIndex<${SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES}u[^]*sampleIndex>=coneSampleCount`,
  ));
  assert.match(contact, /visibility\/f32\(coneSampleCount\)/,
    "both quality levels must retain the same average visibility range");
  assert.doesNotMatch(contact, /pointer|mouse|click/i,
    "AO quality must follow camera stability rather than input-device state");
});

test("contact visibility attenuates indirect diffuse only and adds no storage binding", () => {
  const shade = shaderFunction("shadeDryOpaque", "shadeThinGlass");
  assert.match(shade, /let contactVisibility=dryContactVisibility\(position,hit\.normal,hit\.featureId,hit\.ownerId\)/);
  assert.match(shade, /let ignoredBodyOwner=select\(DRY_OWNER_NONE,hit\.ownerId,hit\.motionKind==DRY_GBUFFER_MOTION_RIGID\);let gi=dryGlobalIllumination\(position,hit\.normal,ignoredBodyOwner\)/);
  assert.match(shade, /let diffuseEnvironment=[^;]*\*contactVisibility\*gi\.visibility\*diffuseEnvironmentScale\/UNIFIED_PI/);
  assert.match(shade, /let specularEnvironment=dryEnvironment\(reflected,surface\.roughness\)\*fresnel/);
  assert.match(shade, /let indirectDiffuse=diffuseColor\*gi\.radiance/);
  assert.match(shade, /return max\(surface\.emissive\+diffuseEnvironment\+specularEnvironment\+direct\*directScale\+indirectDiffuse,vec3f\(0\.0\)\)/);
  assert.doesNotMatch(shade, /(?:surface\.emissive|specularEnvironment|direct)\s*\*\s*contactVisibility/);

  // "adds no storage binding" is the claim under test: contact visibility must
  // read what the pass already binds. Hold that against the published contract
  // rather than a literal, so a legitimate binding added elsewhere updates both
  // sides at once while a shader-only addition still fails.
  const storageBindings = [...svoDrySceneShader.matchAll(/@group\(0\) @binding\((\d+)\) var<storage/g)]
    .map((match) => Number(match[1]));
  assert.deepEqual([...storageBindings].sort((a, b) => a - b),
    SVO_DRY_SCENE_BINDING_CONTRACT
      .filter(({ type }) => type === "read-only-storage").map(({ binding }) => binding),
    "the dry pass must bind exactly the storage resources its contract publishes");
  assert.equal(new Set(storageBindings).size, storageBindings.length);
});
