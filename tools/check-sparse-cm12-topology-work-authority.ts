#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  SPARSE_CM12_TWA_CAUSE,
  SPARSE_CM12_TWA_FAMILY,
  SPARSE_CM12_TWA_FAMILY_COUNT,
  SPARSE_CM12_TWA_FAULT,
  SPARSE_CM12_TWA_GLOBAL_PASS_AUDIT,
  SPARSE_CM12_TWA_INDIRECT_FAMILY_COUNT,
  SPARSE_CM12_TWA_LIVE_GLOBAL_TOKENS,
  appendSparseCM12TWAChange,
  beginSparseCM12TWA,
  commitSparseCM12TWA,
  createSparseCM12TWA,
  createSparseCM12TWAInitialWords,
  createSparseCM12TWALayout,
  publishSparseCM12TWAFamilyExecution,
  recordSparseCM12TWAUncoveredWrite,
  sealSparseCM12TWA,
  type SparseCM12TWA,
  type SparseCM12TWAChange,
  type SparseCM12TWATopology,
} from "../lib/methods/adaptive-mass/sparse-cm12-topology-work-authority";
import {
  SPARSE_CM12_TWA_ENTRY_POINTS,
  createSparseCM12TopologyWorkAuthorityWGSL,
} from "../lib/methods/adaptive-mass/sparse-cm12-topology-work-authority.wgsl";

const assert: (condition: unknown, message: string) => asserts condition =
  (condition, message): asserts condition => { if (!condition) throw new Error(message); };
const equal = (actual: unknown, expected: unknown, message: string): void => {
  if (actual !== expected) throw new Error(`${message}: ${String(actual)} != ${String(expected)}`);
};
const topology = (): SparseCM12TWATopology => ({ brickCapacity: 4,
  cellCapacity: 12, rowCapacity: 11,
  cellRows: Array.from({ length: 12 }, (_, cell) =>
    [cell - 1, cell].filter((row) => row >= 0 && row < 11)) });
const changes: readonly SparseCM12TWAChange[] = Object.freeze([
  { brick: 2, causeMask: SPARSE_CM12_TWA_CAUSE.rungChanged,
    oldCellFirst: 2, oldCellCount: 2, newCellFirst: 4, newCellCount: 2,
    oldRowFirst: 1, oldRowCount: 1, newRowFirst: 4, newRowCount: 1 },
  { brick: 0, causeMask: SPARSE_CM12_TWA_CAUSE.pageIdentity,
    oldCellFirst: 0, oldCellCount: 1, newCellFirst: 1, newCellCount: 1,
    oldRowFirst: 0, oldRowCount: 1, newRowFirst: 1, newRowCount: 1 },
]);

function plan(order: readonly SparseCM12TWAChange[]): { authority: SparseCM12TWA;
  signature: string } {
  const authority = createSparseCM12TWA({ topology: topology() });
  assert(beginSparseCM12TWA(authority, { generation: 1, topologyGeneration: 1 }),
    "TWA begin");
  order.forEach((change) => assert(appendSparseCM12TWAChange(authority, change),
    `TWA append ${change.brick}`));
  const result = sealSparseCM12TWA(authority); assert(result, "TWA seal");
  const signature = JSON.stringify({ bricks: result.changedBricks,
    cells: result.closureCells, rows: result.closureRows, families: result.families,
    indirect: result.indirect });
  return { authority, signature };
}

const forward = plan(changes); const reverse = plan([...changes].reverse());
equal(forward.signature, reverse.signature, "root append order changed stable packets");
const accepted = forward.authority; const acceptedPlan = accepted.plan!;
equal(acceptedPlan.changedBricks.join(","), "0,2", "changed brick packet order");
equal(acceptedPlan.closureCells.join(","), "0,1,2,3,4,5", "old/new cell union");
equal(acceptedPlan.closureRows.join(","), "0,1,2,3,4,5", "incidence row closure");
equal(acceptedPlan.families.transferFaces.join(","), "0,1,2,3,4,5,12,13,14,15,16,17",
  "stable brick-face packet IDs");
(Object.keys(SPARSE_CM12_TWA_FAMILY) as (keyof typeof SPARSE_CM12_TWA_FAMILY)[])
  .forEach((family) => assert(publishSparseCM12TWAFamilyExecution(accepted, family),
    `family receipt ${family}`));
assert(commitSparseCM12TWA(accepted), "complete TWA transaction rejected");
equal(accepted.acceptedGeneration, 1, "accepted generation");

const missing = plan(changes).authority;
(Object.keys(SPARSE_CM12_TWA_FAMILY) as (keyof typeof SPARSE_CM12_TWA_FAMILY)[])
  .slice(1).forEach((family) => publishSparseCM12TWAFamilyExecution(missing, family));
assert(!commitSparseCM12TWA(missing), "missing family receipt committed");
equal(missing.fault, SPARSE_CM12_TWA_FAULT.missingExecution, "missing receipt fault");
const uncovered = plan(changes).authority;
(Object.keys(SPARSE_CM12_TWA_FAMILY) as (keyof typeof SPARSE_CM12_TWA_FAMILY)[])
  .forEach((family) => publishSparseCM12TWAFamilyExecution(uncovered, family));
recordSparseCM12TWAUncoveredWrite(uncovered, 77);
assert(!commitSparseCM12TWA(uncovered), "uncovered topology write committed");
equal(uncovered.fault, SPARSE_CM12_TWA_FAULT.uncoveredWrite, "uncovered write fault");

const oracle = createSparseCM12TWA({ topology: topology(),
  constructionMode: "immutable-full-oracle" });
assert(!beginSparseCM12TWA(oracle, { generation: 1, topologyGeneration: 1,
  runtime: true }), "immutable full oracle became a runtime fallback");
equal(oracle.fault, SPARSE_CM12_TWA_FAULT.runtimeOracle, "runtime oracle fault");
const constructionOracle = createSparseCM12TWA({ topology: topology(),
  constructionMode: "immutable-full-oracle" });
assert(beginSparseCM12TWA(constructionOracle, { generation: 1, topologyGeneration: 1 }),
  "construction oracle begin");
const constructionPlan = sealSparseCM12TWA(constructionOracle);
assert(constructionPlan, "construction oracle seal");
equal(constructionPlan.changedBricks.length, 4, "construction oracle bricks");
equal(constructionPlan.closureCells.length, 12, "construction oracle cells");
equal(constructionPlan.closureRows.length, 11, "construction oracle rows");
(Object.keys(SPARSE_CM12_TWA_FAMILY) as (keyof typeof SPARSE_CM12_TWA_FAMILY)[])
  .forEach((family) => assert(publishSparseCM12TWAFamilyExecution(constructionOracle, family),
    `construction family receipt ${family}`));
assert(commitSparseCM12TWA(constructionOracle), "construction oracle commit");
assert(!beginSparseCM12TWA(constructionOracle, { generation: 2, topologyGeneration: 2 }),
  "construction full oracle ran more than once");
equal(constructionOracle.fault, SPARSE_CM12_TWA_FAULT.runtimeOracle,
  "second construction oracle fault");

const smallLayout = createSparseCM12TWALayout({ brickCapacity: 4,
  cellCapacity: 12, rowCapacity: 11 });
const initial = createSparseCM12TWAInitialWords({ layout: smallLayout });
equal(initial.length, smallLayout.totalWords, "TWA initial words");
equal(smallLayout.indirectBytes, 4 * 3 * SPARSE_CM12_TWA_INDIRECT_FAMILY_COUNT,
  "TWA indirect bytes");
const helpers = createSparseCM12TopologyWorkAuthorityWGSL({ layout: smallLayout });
const source = /* wgsl */`
@group(0) @binding(0) var<storage,read_write> twaArena:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> twaIndirectArgs:array<atomic<u32>>;
fn twaCellIncidenceRange(cell:u32)->vec2u{
  return vec2u(select(cell-1u,0u,cell==0u),select(2u,1u,cell==0u||cell==11u));
}
fn twaCellIncidenceRow(at:u32)->u32{return min(at,10u);}
${helpers}`;
SPARSE_CM12_TWA_ENTRY_POINTS.forEach((entry) => assert(source.includes(`fn ${entry}`),
  `missing TWA entry ${entry}`));
for (const token of ["acceptedTemplateCellInvocation", "acceptedTemplateRowInvocation",
  "templateCellCount", "templateRowCount", "packed.brickCount", "dispatchAccepted"]) {
  assert(!source.includes(token), `TWA standalone source contains global token ${token}`);
}
assert(!source.match(/fn twaFail\([^}]+twaZeroIndirect/s),
  "worker failure path depends on indirect storage binding");
const directory = mkdtempSync(join(tmpdir(), "cm12-twa1-"));
try {
  const path = join(directory, "topology-work-authority.wgsl"); writeFileSync(path, source);
  const result = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(result.stderr || result.stdout);
} finally { rmSync(directory, { recursive: true, force: true }); }

const residentPath = "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts";
const residentLines = readFileSync(resolve(residentPath), "utf8").split("\n");
const liveGlobalTokens = SPARSE_CM12_TWA_LIVE_GLOBAL_TOKENS.flatMap((token) => residentLines
  .map((line, index) => line.includes(token) ? { file: residentPath, line: index + 1, token,
    source: line.trim() } : undefined).filter((value) => value !== undefined));
SPARSE_CM12_TWA_GLOBAL_PASS_AUDIT.forEach(({ pass }) => assert(
  liveGlobalTokens.some((gap) => gap.token === pass), `missing live global gap token ${pass}`));
const production = createSparseCM12TWALayout({ brickCapacity: 600,
  cellCapacity: 537_220, rowCapacity: 1_574_783 });
console.log(JSON.stringify({ authority: "TWA1", configuration: "B16/P16",
  familyCount: SPARSE_CM12_TWA_FAMILY_COUNT,
  indirectFamilyCount: SPARSE_CM12_TWA_INDIRECT_FAMILY_COUNT,
  closure: "changed brick old/new cell+row ranges plus immutable incidence rows",
  deterministicPackets: true, receiptPolicy: "all non-empty families + no uncovered writes",
  constructionOracle: "immutable construction specialization; runtime selection rejected",
  cpuOracle: "stable order + missing receipt + uncovered write pass",
  naga: "valid", productionLayoutBytes: production.totalBytes,
  indirectBytes: production.indirectBytes, globalPassAudit: SPARSE_CM12_TWA_GLOBAL_PASS_AUDIT,
  liveGlobalTokens, wgslBytes: source.length }, null, 2));
