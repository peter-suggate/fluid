import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createSparseCM12WGSLPruner, sparseCM12WGSLForEntryPoints } from
  "../lib/methods/adaptive-mass/sparse-cm12-wgsl-pruning";
import { referenceSparseCM12WGSLForEntryPoints as previous } from
  "./helpers/sparse-cm12-wgsl-pruning-reference";

const lexicalFixture = `// fn decoy(){ dangling((); } const ghost = 99u;
/* fn hidden() {} var<private> counterfeit: f32; { } */
const width = 4u; const widthExtra = 7u; override stride: u32 = width;
alias Scalar = f32; struct Cell { value: Scalar, };
struct Unused { value: u32, };
@group(0) @binding(0) var<storage, read_write> data: array<Cell, stride>;
@group(0) @binding(1) var<storage, read_write> spare: array<Unused, widthExtra>;
fn scale(v: Scalar) -> Scalar { let spare = v + 1.0; return spare; }
fn left(v: Scalar) -> Scalar { return right(v); }
fn right(v: Scalar) -> Scalar { if (v > 0.0) { return left(v - 1.0); } return scale(v); }
fn scaleExtra() -> u32 { return widthExtra; }
@compute @workgroup_size(width) fn first() { data[0].value = right(2.0); }
@compute @workgroup_size(1) fn second() { spare[0].value = scaleExtra(); }
@compute @workgroup_size(1) fn third() { var local: u32 = 1u; }
`;

test("a reused parse emits byte-identical slices for comments, globals, cycles and root order", () => {
  const prune = createSparseCM12WGSLPruner(lexicalFixture);
  const families = [[], ["missing"], ["first"], ["second"], ["third"],
    ["first", "second"], ["second", "first"], ["first", "first", "missing"]];
  for (const roots of [...families, ...families.toReversed()]) {
    const expected = previous(lexicalFixture, roots);
    assert.equal(prune(roots), expected, roots.join(","));
    assert.equal(sparseCM12WGSLForEntryPoints(lexicalFixture, roots), expected);
  }
  // Preserve the existing conservative identifier treatment, including local
  // names that shadow globals. This change only reuses parsing work.
  assert.match(prune(["first"]), /fn scale\(/);
  assert.doesNotMatch(prune(["first"]), /fn scaleExtra\(/);
  assert.doesNotMatch(prune(["second"]), /fn scale\(/);
});

test("different source owners and consecutive requests cannot share reachability", () => {
  const changed = lexicalFixture.replace("return widthExtra;", "return 99u;");
  const first = createSparseCM12WGSLPruner(lexicalFixture);
  const second = createSparseCM12WGSLPruner(changed);
  for (const roots of [["first"], ["second"], [], ["third"], ["second"]]) {
    assert.equal(first(roots), previous(lexicalFixture, roots));
    assert.equal(second(roots), previous(changed, roots));
  }
  assert.notEqual(first(["second"]), second(["second"]));
  assert.throws(() => createSparseCM12WGSLPruner("fn bad()"), /has no body/);
  assert.throws(() => createSparseCM12WGSLPruner("fn bad() {"), /unclosed body/);
});

test("every generated B8/P8 four-entry family remains byte-identical", () => {
  // Reuse the production shader checker's bounded layout fixture. Its explicit
  // emit mode never requests a GPU or constructs a solver/atlas.
  const environment = { ...process.env };
  delete environment.NODE_OPTIONS;
  delete environment.WEBGPU_NODE_MODULE;
  const generated = spawnSync(process.execPath, ["--import", "tsx",
    fileURLToPath(new URL("../tools/check-sparse-cm12-wgsl-dawn.ts", import.meta.url)),
    "--emit-source"], { env: environment, encoding: "utf8", maxBuffer: 4 * 1024 * 1024,
    timeout: 10_000 });
  assert.equal(generated.status, 0, generated.stderr || String(generated.error));
  const source = generated.stdout;
  const entries = [...source.matchAll(/@compute\s+@workgroup_size\([^)]*\)\s+fn\s+(\w+)/g)]
    .map(match => match[1]!);
  assert.ok(source.length > 100_000);
  assert.ok(entries.length > 100);
  const prune = createSparseCM12WGSLPruner(source);
  for (let index = 0; index < entries.length; index += 4) {
    const roots = entries.slice(index, index + 4);
    assert.equal(prune(roots), previous(source, roots), roots.join(","));
  }
  for (const roots of [[], [entries[0]!, entries.at(-1)!], entries, ["missing"]]) {
    assert.equal(prune(roots), previous(source, roots));
  }
});
