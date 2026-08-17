#!/usr/bin/env node
/** Naga-validate the binding-free A4D2 scheduler for B4/B8/B16. */
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { createSparseCM12ProductionActivityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-production-activity";
import { SPARSE_CM12_PRODUCTION_ACTIVITY_REQUIRED_CERTIFICATE } from
  "../lib/methods/adaptive-mass/sparse-cm12-production-activity";
import { createSparseCM12ProductionActivityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-production-activity.wgsl";

const directory = mkdtempSync(join(tmpdir(), "fluid-a4d2-wgsl-"));
try {
  for (const brickFineResolution of [4, 8, 16] as const) {
    const layout = createSparseCM12ProductionActivityLayout({
      brickCapacity: 509, brickFineResolution,
    });
    const generated = createSparseCM12ProductionActivityWGSL(layout);
    const source = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> activity:array<atomic<u32>>;
fn cm12ActivityCandidateGeneration()->u32{return 7u;}
fn cm12ActivityCandidateBrickCount()->u32{return 17u;}
fn cm12ActivityCandidateListGeneration()->u32{return 7u;}
fn cm12ActivityCandidateBrickInvocation(candidate:u32)->u32{return candidate;}
fn cm12ActivityTopologyGeneration()->u32{return 3u;}
fn cm12ActivityFramePlanGeneration()->u32{return 7u;}
fn cm12ActivityBrickTopologySignature(brick:u32)->u32{return brick^3u;}
fn cm12ActivityBrickTopologyChanged(brick:u32)->bool{return brick==0xffffffffu;}
fn cm12ActivityBuildTileTrigger(brick:u32,tile:u32)->vec4u{
  return vec4u(7u,brick|tile,${(SPARSE_CM12_PRODUCTION_ACTIVITY_REQUIRED_CERTIFICATE | 1) >>> 0}u,
    brick^3u);
}
fn cm12ActivityExpectedTileContributionCount(brick:u32,tile:u32)->u32{
  return select(1u,0u,(brick|tile)==0xffffffffu);
}
fn cm12ActivityExpectedTileCheck(brick:u32,tile:u32)->u32{return brick^tile;}
${generated}
fn cm12ActivityRebuildExactTile(brick:u32,tile:u32)->A4D2TileSummary{
  return A4D2TileSummary(vec4i(i32(brick),i32(tile),0,0),
    vec4f(f32(brick),f32(tile),0.0,0.0),0u,0u,0u,1u,brick^tile);
}
fn cm12ActivityPublishFramePlanRoot(brick:u32,tile:u32,directStages:u32,
 closureStages:u32,cause:u32,depth:u32,generation:u32)->bool{
  return (brick|tile|directStages|closureStages|cause|depth|generation)!=0xffffffffu;
}
fn cm12ActivityPublishExactBrick(brick:u32,moments:vec4i,metrics:vec4f,
 masks:vec4u,prior:vec4u)->vec4u{
  return vec4u(u32(max(0,moments.x)),masks.x,prior.z,
    select(0u,1u,brick<${layout.brickCapacity}u&&metrics.x>=0.0));
}
`;
    const path = join(directory, `a4d2-b${brickFineResolution}.wgsl`);
    writeFileSync(path, source);
    const result = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
    if (result.status !== 0) {
      throw new Error(`A4D2 B${brickFineResolution}:\n${result.stderr || result.stdout}`);
    }
    console.log(`validated A4D2 B${brickFineResolution} (${layout.tilesPerBrick} tiles/brick)`);
  }
} finally {
  rmSync(directory, { recursive: true, force: true });
}
