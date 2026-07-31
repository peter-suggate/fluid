import assert from "node:assert/strict";
import test from "node:test";
import {
  VISUALIZATION_STORAGE_BUFFERS_PER_STAGE,
  assertVisualizationProgram,
  visualizationBindGroupEntries,
  visualizationBindingBudget,
  visualizationBindingPreambleWGSL,
  visualizationProgramProblems,
  type VisualizationBinding,
  type VisualizationProgram,
} from "../lib/visualization-bindings";
import {
  OCTREE_LIFECYCLE_MEMBERSHIP_PROGRAM,
  OCTREE_TECHNIQUE_OVERLAY_CODES,
  OCTREE_TECHNIQUE_PROGRAMS,
  OCTREE_TECHNIQUE_PROGRAM_FOR_MODE,
  octreeTechniqueProgramForCode,
  type OctreeTechniqueOverlayMode,
} from "../lib/octree-technique-debug";
import {
  octreeTechniqueFaceShader,
  octreeTechniqueFineLifecycleShader,
  octreeTechniqueLifecycleShader,
  octreeTechniqueStructuredShader,
  octreeTechniqueTopologyShader,
} from "../lib/webgpu-octree-technique-overlay";

const storage = (name: string): VisualizationBinding =>
  ({ name, kind: "read-only-storage", type: "array<u32>", resource: name });

const program = (bindings: readonly VisualizationBinding[]): VisualizationProgram =>
  ({ id: "test", label: "Test program", bindings });

/* ------------------------------------------------------------------------- */
/* The budget                                                                 */
/* ------------------------------------------------------------------------- */

test("the budget counts storage buffers apart from uniforms and textures", () => {
  const budget = visualizationBindingBudget([
    { name: "u", kind: "uniform", type: "Uniforms", resource: "uniforms" },
    { name: "rows", kind: "texture-3d-uint", resource: "ownerRows" },
    storage("a"), storage("b"),
    { name: "out", kind: "storage", type: "array<u32>", resource: "out" },
  ]);
  assert.equal(budget.storageBuffers, 3);
  assert.equal(budget.uniformBuffers, 1);
  assert.equal(budget.textures, 1);
  assert.equal(budget.total, 5);
  assert.equal(budget.headroom, VISUALIZATION_STORAGE_BUFFERS_PER_STAGE - 3);
});

test("an eleventh storage buffer is a build failure, not a driver error", () => {
  // The whole reason the technique overlay is five programs rather than one.
  const atCap = program(Array.from(
    { length: VISUALIZATION_STORAGE_BUFFERS_PER_STAGE }, (_, index) => storage(`slot${index}`)));
  assert.deepEqual(visualizationProgramProblems(atCap), []);

  const overCap = program([...atCap.bindings, storage("oneTooMany")]);
  const problems = visualizationProgramProblems(overCap);
  assert.equal(problems.length, 1);
  assert.match(problems[0], /portable per-stage ceiling/);
  assert.throws(() => assertVisualizationProgram(overCap), /portable per-stage ceiling/);
});

test("a program cannot name one binding twice, or bind one resource twice", () => {
  assert.match(visualizationProgramProblems(program([storage("a"), storage("a")]))[0],
    /duplicate binding name/);
  assert.match(visualizationProgramProblems(program([
    { name: "left", kind: "read-only-storage", type: "array<u32>", resource: "same" },
    { name: "right", kind: "read-only-storage", type: "array<u32>", resource: "same" },
  ]))[0], /bound twice/);
});

test("a buffer binding without a WGSL type is caught before it reaches a shader", () => {
  assert.match(visualizationProgramProblems(program([
    { name: "rows", kind: "read-only-storage", resource: "rows" },
  ]))[0], /has no WGSL type/);
  // A texture needs none.
  assert.deepEqual(visualizationProgramProblems(program([
    { name: "rows", kind: "texture-3d-uint", resource: "rows" },
  ])), []);
});

/* ------------------------------------------------------------------------- */
/* Generation                                                                 */
/* ------------------------------------------------------------------------- */

test("slot numbers come from declaration order and nowhere else", () => {
  const preamble = visualizationBindingPreambleWGSL(program([
    { name: "u", kind: "uniform", type: "Uniforms", resource: "uniforms" },
    { name: "rows", kind: "texture-3d-uint", resource: "ownerRows" },
    { name: "headers", kind: "read-only-storage", type: "array<LeafHeader>", resource: "headers" },
    { name: "out", kind: "storage", type: "array<u32>", resource: "out" },
    { name: "depth", kind: "texture-depth-2d", resource: "depth" },
  ]));
  assert.deepEqual(preamble.split("\n"), [
    "@group(0) @binding(0) var<uniform> u:Uniforms;",
    "@group(0) @binding(1) var rows:texture_3d<u32>;",
    "@group(0) @binding(2) var<storage,read> headers:array<LeafHeader>;",
    "@group(0) @binding(3) var<storage,read_write> out:array<u32>;",
    "@group(0) @binding(4) var depth:texture_depth_2d;",
  ]);
});

test("bind-group entries follow the same order the preamble numbered", () => {
  const declared = program([storage("a"), storage("b"), storage("c")]);
  const entries = visualizationBindGroupEntries(declared, (resource) => `resolved:${resource}`);
  assert.deepEqual(entries, [
    { binding: 0, resource: "resolved:a" },
    { binding: 1, resource: "resolved:b" },
    { binding: 2, resource: "resolved:c" },
  ]);
});

test("a resource the frame cannot supply refuses the whole group", () => {
  // A partially bound program would read whatever the previous publication left
  // in that slot, which is a wrong picture rather than an error.
  const declared = program([storage("present"), storage("absent")]);
  assert.equal(
    visualizationBindGroupEntries(declared, (resource) => resource === "present" ? "ok" : undefined),
    undefined);
});

/* ------------------------------------------------------------------------- */
/* The declared octree programs                                               */
/* ------------------------------------------------------------------------- */

test("every declared overlay program fits the portable stage", () => {
  for (const declared of Object.values(OCTREE_TECHNIQUE_PROGRAMS)) {
    assert.deepEqual(visualizationProgramProblems(declared), [], declared.id);
  }
  assert.deepEqual(visualizationProgramProblems(OCTREE_LIFECYCLE_MEMBERSHIP_PROGRAM), []);
});

test("the fullest program still has headroom, and the report says how much", () => {
  const budgets = Object.values(OCTREE_TECHNIQUE_PROGRAMS)
    .map((declared) => ({ id: declared.id, ...visualizationBindingBudget(declared.bindings) }));
  const fullest = budgets.reduce((worst, next) => next.storageBuffers > worst.storageBuffers ? next : worst);
  assert.equal(fullest.id, "fine");
  // Seven of ten: the fine band is the tightest program, and the three slots
  // left are the budget any new fine-band view has to fit inside.
  assert.equal(fullest.storageBuffers, 7);
  assert.ok(fullest.headroom > 0, "a program at the ceiling cannot take another field");
});

test("the cost views cost no bindings, which is why they could be added at all", () => {
  // Both read only what the structured program already holds: the assembled row
  // width from the headers, and the owner map around each leaf. A cost view that
  // needed its own publication would have to argue for a slot instead.
  for (const mode of ["row-cost", "stencil-locality"] as const) {
    assert.equal(OCTREE_TECHNIQUE_PROGRAM_FOR_MODE[mode], "structured");
  }
  const names = new Set(OCTREE_TECHNIQUE_PROGRAMS.structured.bindings.map((binding) => binding.name));
  for (const needed of ["headers", "ownerRows", "metrics", "accepted", "structured"]) {
    assert.ok(names.has(needed), `structured no longer binds ${needed}`);
  }
  assert.ok(visualizationBindingBudget(OCTREE_TECHNIQUE_PROGRAMS.structured.bindings).headroom > 0);
});

test("each overlay shader embeds exactly the preamble its program declares", () => {
  // This is the pairing the resolver exists to guarantee: the slots the shader
  // reads and the slots the host binds come from one declaration.
  const shaders: readonly [keyof typeof OCTREE_TECHNIQUE_PROGRAMS, string][] = [
    ["topology", octreeTechniqueTopologyShader],
    ["face", octreeTechniqueFaceShader],
    ["structured", octreeTechniqueStructuredShader],
    ["lifecycle", octreeTechniqueLifecycleShader],
    ["fine", octreeTechniqueFineLifecycleShader],
  ];
  for (const [id, source] of shaders) {
    const preamble = visualizationBindingPreambleWGSL(OCTREE_TECHNIQUE_PROGRAMS[id]);
    assert.ok(source.includes(preamble), `${id} does not embed its declared preamble`);
    // And no slot is written by hand anywhere else in the program.
    const declarations = source.match(/@group\(0\) @binding\(/g) ?? [];
    assert.equal(declarations.length, OCTREE_TECHNIQUE_PROGRAMS[id].bindings.length, id);
  }
});

/* ------------------------------------------------------------------------- */
/* Dispatch                                                                   */
/* ------------------------------------------------------------------------- */

test("every technique mode resolves to a program, except the one that owns its pipelines", () => {
  for (const mode of Object.keys(OCTREE_TECHNIQUE_OVERLAY_CODES) as OctreeTechniqueOverlayMode[]) {
    const declared = OCTREE_TECHNIQUE_PROGRAM_FOR_MODE[mode];
    if (mode === "blast-radius") {
      // It floods the owner map on a compute pass before anything is shaded.
      assert.equal(declared, undefined);
      continue;
    }
    assert.ok(declared, `${mode} names no program`);
    assert.ok(declared! in OCTREE_TECHNIQUE_PROGRAMS, `${mode} names an unknown program`);
  }
});

test("a mode code resolves to the same program its mode declares", () => {
  for (const [mode, code] of Object.entries(OCTREE_TECHNIQUE_OVERLAY_CODES)) {
    assert.equal(
      octreeTechniqueProgramForCode(code),
      OCTREE_TECHNIQUE_PROGRAM_FOR_MODE[mode as OctreeTechniqueOverlayMode],
      mode);
  }
  assert.equal(octreeTechniqueProgramForCode(-1), undefined);
});
