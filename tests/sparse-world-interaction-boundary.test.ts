import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const solver = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts",
  import.meta.url,
), "utf8");
const adapter = readFileSync(new URL(
  "../lib/sparse-world/internal/cm12-adapter.ts",
  import.meta.url,
), "utf8");

const sourceBetween = (source: string, begin: string, end: string): string => {
  const start = source.indexOf(begin);
  const finish = source.indexOf(end, start + begin.length);
  assert.ok(start >= 0 && finish > start, `${begin} source range must be identifiable`);
  return source.slice(start, finish);
};

test("liquid-ball authoring uses only the public sparse-world edit API", () => {
  const injection = sourceBetween(solver,
    "  injectLiquidBall(ball: InjectedLiquidBall): void {",
    "  /**\n   * Adopt scene scalars");
  assert.match(injection, /this\.sparseWorld\.edit\(\{/);
  assert.match(injection, /kind: "liquid-ellipsoid"/);
  assert.doesNotMatch(injection, /sparseRuntime|encodeLiquidInjection/);

  const runtime = sourceBetween(adapter,
    "export interface CM12SparseWorldRuntime {",
    "export interface CM12SparseWorldDeveloperTrace");
  assert.doesNotMatch(runtime, /encodeLiquidInjection|encodeLiquidJetInjection/,
    "normal features must not regain an implementation-specific injection escape hatch");
});

test("a fluid edit owns one public world generation", () => {
  const interaction = sourceBetween(adapter,
    "  edit(edit: SparseWorldEdit): SparseWorldEditReceipt {",
    "  encodeStep(encoder: GPUCommandEncoder, input: SparseWorldStepInput): SparseWorldStep {");
  assert.match(interaction, /\(value - origin\[axis\]!\) \* inverseCell/,
    "world-space interaction centres must be translated by the world's origin");
  assert.match(interaction,
    /interaction\.radii_m\.map\(\(value\) => value \* inverseCell\)/);
  assert.match(interaction, /this\.resident\.encodeLiquidInjection\([\s\S]*this\.generation \+= 1;/);

  const step = sourceBetween(adapter,
    "  encodeStep(encoder: GPUCommandEncoder, input: SparseWorldStepInput): SparseWorldStep {",
    "  presentation(): SparseWorldPresentation");
  assert.doesNotMatch(step, /Interaction|encodeLiquidInjection/,
    "step encoding and interaction encoding need independent parameter lifetimes");
});

test("scene and rigid-body edits cross the same public world boundary", () => {
  const sceneEdit = sourceBetween(solver,
    "  applySceneUniforms(scene: SceneDescription): void {",
    "  /**\n   * Adopt the controls");
  assert.match(sceneEdit, /this\.sparseWorld\.edit\(\{ kind: "set-scene", scene \}\)/);
  assert.doesNotMatch(sceneEdit, /sparseRuntime|setSolidWorld|setRefinementRegionParameters/);

  const advance = sourceBetween(solver,
    "  advanceTo(time_s: number, bodies: RigidBodyState[]): boolean {",
    "  /** Publish the receipt");
  assert.match(advance, /this\.sparseWorld\.encodeStep\(encoder, \{[\s\S]*rigidBodies: activeBodies/);
  assert.match(advance, /liquidInflow,[\s\S]*this\.sparseWorld\.edit\(\{[\s\S]*kind: "liquid-jet"/);
  assert.doesNotMatch(advance, /outletFine|radiusFine|velocityFinePerSecond/,
    "hose features must remain in world-space SI units");
  assert.doesNotMatch(advance, /rigidSystem\?\.syncBodies|rigidSystem\?\.encode/);
  assert.doesNotMatch(advance, /sparseRuntime\.encodeLiquidJetInjection/);

  const runtime = sourceBetween(adapter,
    "export interface CM12SparseWorldRuntime {",
    "export interface CM12SparseWorldDeveloperTrace");
  assert.doesNotMatch(runtime,
    /encodeLiquidInjection|encodeLiquidJetInjection|setSolidWorld|setRefinementRegionParameters/,
  "application edits must not regain implementation-specific escape hatches");
});

test("a refinement-only scene edit does not rebuild SolidWorld", () => {
  const interaction = sourceBetween(adapter,
    "  edit(edit: SparseWorldEdit): SparseWorldEditReceipt {",
    "  encodeStep(encoder: GPUCommandEncoder, input: SparseWorldStepInput): SparseWorldStep {");
  assert.match(interaction,
    /if \(solidWorldChanged\) \{\s*this\.resident\.setSolidWorld\(/,
    "static collider uploads must be guarded by their own authority stamp");
  assert.match(interaction,
    /this\.resident\.setRefinementRegionParameters\(packSparseCM12RefinementRegions/,
    "the small refinement policy remains independently live");
});
