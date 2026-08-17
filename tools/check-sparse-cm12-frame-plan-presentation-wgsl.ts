#!/usr/bin/env node
/** Naga-validate the binding-free FPP1 executor for B4/B8/B16. */
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { createSparseCM12FramePlanLayout } from "../lib/core/sparse-cm12-frame-plan";
import { createSparseCM12FramePlanWGSL } from "../lib/core/sparse-cm12-frame-plan.wgsl";
import { createSparseCM12FramePlanPresentationLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation";
import { createSparseCM12FramePlanPresentationWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation.wgsl";

const directory = mkdtempSync(join(tmpdir(), "fluid-fpp1-wgsl-"));
try {
  for (const brickFineResolution of [4, 8, 16] as const) {
    const capacity = 17;
    const frame = createSparseCM12FramePlanLayout({ brickCapacity: capacity,
      brickFineResolution, packetCount: 6 });
    const packet = createSparseCM12FramePlanPresentationLayout({
      pageCapacity: capacity, brickFineResolution,
    });
    const samples = capacity * brickFineResolution ** 3;
    const source = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> framePlan:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> presentationPacket:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> candidateSamples:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> candidateControl:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> acceptedSamples:array<atomic<u32>>;
fn cm12PresentationPageMatches(
  page:u32,brick:u32,logicalKey:u32,topologyGeneration:u32)->bool{
  return page==brick&&brick==logicalKey&&topologyGeneration!=0xffffffffu;
}
fn cm12PresentationPreparePage(
  brick:u32,page:u32,lane:u32,generation:u32,causeMask:u32)->u32{
  return select(1u,0u,brick==page&&lane<64u&&(generation|causeMask)!=0xffffffffu);
}
fn cm12PresentationExactSample(
  brick:u32,page:u32,tile:u32,sample:u32,generation:u32,causeMask:u32)->vec2u{
  return vec2u(brick^page^tile^sample^generation^causeMask,0u);
}
fn cm12PresentationStoreCandidate(
  page:u32,localIndex:u32,generation:u32,payload:u32){
  let at=page*${brickFineResolution ** 3}u+localIndex;
  if(at<${samples}u){atomicStore(&candidateSamples[at],payload^generation);}
}
fn cm12PresentationLoadCandidate(page:u32,localIndex:u32,generation:u32)->u32{
  let at=page*${brickFineResolution ** 3}u+localIndex;
  return select(0u,atomicLoad(&candidateSamples[at])^generation,at<${samples}u);
}
fn cm12PresentationStoreAccepted(page:u32,localIndex:u32,payload:u32){
  let at=page*${brickFineResolution ** 3}u+localIndex;
  if(at<${samples}u){atomicStore(&acceptedSamples[at],payload);}
}
fn cm12PresentationCommitCandidate(page:u32,generation:u32){
  if(page<${capacity}u){atomicStore(&candidateControl[page],generation);}
}
${createSparseCM12FramePlanWGSL({ layout: frame })}
${createSparseCM12FramePlanPresentationWGSL({ layout: packet })}
`;
    const path = join(directory, `fpp1-b${brickFineResolution}.wgsl`);
    writeFileSync(path, source);
    const result = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
    if (result.status !== 0) {
      throw new Error(`FPP1 B${brickFineResolution}:\n${result.stderr || result.stdout}`);
    }
    console.log(`validated FPP1 B${brickFineResolution}/P${brickFineResolution}`);
  }
} finally {
  rmSync(directory, { recursive: true, force: true });
}
