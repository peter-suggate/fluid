#!/usr/bin/env node
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER,
  compareSparseCM12PressureSolveAuthorityQA,
  createSparseCM12PressureSolveAuthorityInitialWords,
  createSparseCM12PressureSolveAuthorityLayout,
  sparseCM12PressureSolveAuthorityIndirectByteOffset,
  sparseCM12PressureSolveTailIndirectByteOffset,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-solve-authority";
import { createSparseCM12PressureSolveAuthorityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-solve-authority.wgsl";

const brickCapacity = 513;
const hierarchyLevelCounts = [96, 12, 2] as const;
const layout = createSparseCM12PressureSolveAuthorityLayout({
  brickCapacity, hierarchyLevelCounts, baseWords: 128,
});
const initial = createSparseCM12PressureSolveAuthorityInitialWords(layout);
assert.equal(initial.byteLength, layout.totalBytes);
assert.equal(initial[layout.baseWords + SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER.totalWords],
  layout.totalWords);
assert.deepEqual(layout.hierarchyLevelOffsets, [0, 96, 108]);
assert.equal(layout.hierarchyNodeCapacity, 110);
assert(sparseCM12PressureSolveAuthorityIndirectByteOffset(layout, "brick", "work") > 0);
assert(sparseCM12PressureSolveTailIndirectByteOffset(layout, 1, "hierarchyNode")
  > sparseCM12PressureSolveTailIndirectByteOffset(layout, 0, "hierarchyNode"));

const parent0 = Array.from({ length: brickCapacity }, (_, brick) => brick % 96);
const parent1 = Array.from({ length: brickCapacity }, (_, brick) => brick % 12);
const parent2 = Array.from({ length: brickCapacity }, (_, brick) => brick % 2);
const brickWet = Array.from({ length: brickCapacity }, (_, brick) =>
  brick === 0 || brick === 127 || brick === 300 || brick === 512);
const brickBits = new Uint32Array(Math.ceil(brickCapacity / 32));
const nodeBits = new Uint32Array(Math.ceil(layout.hierarchyNodeCapacity / 32));
const setBit = (words: Uint32Array, id: number) => { words[id >>> 5]! |= 1 << (id & 31); };
brickWet.forEach((wet, brick) => {
  if (!wet) return;
  setBit(brickBits, brick);
  setBit(nodeBits, layout.hierarchyLevelOffsets[0]! + parent0[brick]!);
  setBit(nodeBits, layout.hierarchyLevelOffsets[1]! + parent1[brick]!);
  setBit(nodeBits, layout.hierarchyLevelOffsets[2]! + parent2[brick]!);
});
const stableWords = brickWet.map((wet, brick) => [wet ? 0x3f80_0000 : 0, brick >>> 0]);
const qa = compareSparseCM12PressureSolveAuthorityQA({
  brickWet, hierarchyParentByLevel: [parent0, parent1, parent2],
  localBrickBits: brickBits, localHierarchyNodeBits: nodeBits,
  fullBrickWords: stableWords, localBrickWords: stableWords.map((words) => [...words]),
});
assert.equal(qa.exact, true);
assert.equal(qa.wetBrickCount, 4);
const missingBits = brickBits.slice();
missingBits[300 >>> 5]! &= ~(1 << (300 & 31));
const missing = compareSparseCM12PressureSolveAuthorityQA({
  brickWet, hierarchyParentByLevel: [parent0, parent1, parent2],
  localBrickBits: missingBits, localHierarchyNodeBits: nodeBits,
});
assert.equal(missing.firstMismatchFamily, "brick-membership");
assert.equal(missing.firstMismatchId, 300);

const wgslFor = (qaFullOracle: boolean): string => {
  const checkedLayout = qaFullOracle
    ? createSparseCM12PressureSolveAuthorityLayout({
      brickCapacity, hierarchyLevelCounts, baseWords: 128, qaFullOracle: true,
    }) : layout;
  const helpers = createSparseCM12PressureSolveAuthorityWGSL({
    layout: checkedLayout, arenaName: "arena", prefix: "psa", workgroupSize: 64,
  });
  return /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> arena:array<atomic<u32>>;
fn psaFrameGeneration()->u32{return 2u;}
fn psaTopologyGeneration()->u32{return 3u;}
fn psaPCMGeneration()->u32{return 4u;}
fn psaPCFGeneration()->u32{return 5u;}
fn psaExpectedProducerReceipts()->u32{return 0u;}
fn psaPCMCellWorkgroupCount()->u32{return 9u;}
fn psaCellBrick(cell:u32)->u32{return cell%513u;}
fn psaBrickWetExact(brick:u32)->bool{return brick==0u||brick==127u||brick==300u||brick==512u;}
fn psaHierarchyParent(level:u32,brick:u32)->u32{
  if(level==0u){return brick%96u;}if(level==1u){return brick%12u;}return brick%2u;}
fn psaHierarchyNodeActiveExact(node:u32)->bool{let address=psaHierarchyNodeAddress(node);
  if(address.x==0xffffffffu){return false;}
  for(var brick=0u;brick<513u;brick+=1u){
    if(psaBrickWetExact(brick)&&psaHierarchyParent(address.x,brick)==address.y){return true;}}
  return false;}
fn psaPressureArithmeticActive()->bool{return true;}
fn psaPressureAddressingReady()->bool{return true;}
${helpers}
@compute @workgroup_size(1)
fn exercisePSA(){_=psaMarkPCMCellTransition(300u,1u,false);
  _=psaMarkTopologyBrickBlast(127u,false);_=psaWetBrickInvocation(0u);
  _=psaActiveHierarchyNodeInvocation(0u);_=psaActiveHierarchyNodeAddress(0u);
  _=psaWetBrickContains(0u);_=psaActiveHierarchyNodeContains(0u);
  _=psaAcceptedGeneration();_=psaFaultCode();_=psaWetBrickCount();
  _=psaActiveHierarchyNodeCount();}
`;
};
const directory = mkdtempSync(join(tmpdir(), "sparse-cm12-psa1-"));
try {
  for (const qaFullOracle of [false, true]) {
    const path = join(directory, qaFullOracle ? "psa1-qa.wgsl" : "psa1.wgsl");
    writeFileSync(path, wgslFor(qaFullOracle));
    const checked = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
    if (checked.error) throw checked.error;
    if (checked.status !== 0) throw new Error(checked.stderr || checked.stdout);
  }
} finally {
  rmSync(directory, { recursive: true, force: true });
}

process.stdout.write(`${JSON.stringify({
  abi: "PSA1/v1", brickCapacity: layout.brickCapacity,
  hierarchyNodeCapacity: layout.hierarchyNodeCapacity,
  wetBrickCount: qa.wetBrickCount,
  activeHierarchyNodeCount: qa.activeHierarchyNodeCount,
  totalWords: layout.totalWords, cpuOracle: "bit-exact",
  naga: "production+full-oracle valid",
}, null, 2)}\n`);
