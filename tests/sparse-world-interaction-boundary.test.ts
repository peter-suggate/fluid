import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const solver = readFileSync(new URL(
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver.ts",
  import.meta.url,
), "utf8");
const adapter = readFileSync(new URL(
  "../lib/sparse-world/internal/adaptive-volume-adapter.ts",
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
  assert.doesNotMatch(injection, /encodeLiquidInjection/);
  assert.deepEqual([...injection.matchAll(/this\.sparseRuntime\.(\w+)/g)].map(match => match[1]),
    ["pendingLiquidInteractions"],
    "the runtime only reports whether the public edit needs a topology boundary");

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
  const encode = sourceBetween(adapter,
    "  private encodeInteraction(interaction: SparseWorldFluidEdit,",
    "  constructor(");
  assert.match(encode, /\(value - origin\[axis\]!\) \* inverseCell/,
    "world-space interaction centres must be translated by the world's origin");
  assert.match(encode,
    /interaction\.radii_m\.map\(value => value \* inverseCell\)/);
  assert.match(encode, /this\.resident\.encodeLiquidInjection\(/);
  assert.match(interaction, /this\.encodeInteraction\([\s\S]*this\.generation \+= 1;/);
  const completion = sourceBetween(adapter,
    "  completePendingLiquidInteractions(): void {",
    "  private encodeInteraction(");
  assert.doesNotMatch(completion, /this\.generation \+=/,
    "applying a prepared dose must not count the public edit a second time");

  const step = sourceBetween(adapter,
    "  encodeStep(encoder: GPUCommandEncoder, input: SparseWorldStepInput): SparseWorldStep {",
    "  presentation(): SparseWorldPresentation");
  assert.doesNotMatch(step, /Interaction|encodeLiquidInjection/,
    "step encoding and interaction encoding need independent parameter lifetimes");
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
