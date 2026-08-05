import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import test from "node:test";
import {
  coarsePhiBFECCEnabled,
  FLUID_COARSE_PHI_BFECC_ENV,
} from "../lib/webgpu-octree-fine-levelset-transport";
import { fineLevelSetRigidCarveWGSL } from "../lib/webgpu-octree-fine-levelset-rigid-carve";
import { webgpuOctreeLosassoBackendWGSL } from "../lib/webgpu-octree-losasso-backend.wgsl";
import { octreeLosassoConditionedOperatorWGSL } from "../lib/webgpu-octree-losasso-conditioned-operator";
import { octreeLosassoFineTransportWGSL } from "../lib/webgpu-octree-losasso-fine-transport.wgsl";
import { WebGPUOctreeLosassoFineTransport }
  from "../lib/webgpu-octree-losasso-fine-transport";
import {
  fineSurfaceFilteredNormalsEnabled,
  FLUID_FINE_FILTERED_NORMALS_ENV,
} from "../lib/webgpu-water-global-fine-tetra";

const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

test("fully enclosed solid owners leave the Losasso pressure frontier", () => {
  const projection = read("../lib/webgpu-octree.ts");
  const wet = projection.slice(projection.indexOf("fn currentPressureOwnerWet"),
    projection.indexOf("fn previousFrontierHasExactIdentity"));
  assert.match(wet, /solidAt\(vec3i\(origin\+vec3u\(x,y,z\)\)\)\.fraction<0\.999999/);
  assert.match(wet, /if\(fullySolid\)\{wet=false;\}/);

  const backend = read("../lib/webgpu-octree-losasso-backend.wgsl.ts");
  const sort = backend.slice(backend.indexOf("fn sortLosassoIncidences"),
    backend.indexOf("fn clearLosassoFaceDirectory"));
  assert.match(sort, /valid&&allSolidBlocked[\s\S]*diagonal\[row\]=1\.;rhs\[row\]=0\.;return;/);
});

test("Losasso rigid faces use analytic face-plane area and refresh every substep", () => {
  const backend = read("../lib/webgpu-octree-losasso-backend.wgsl.ts");
  assert.match(backend, /for\(var sample=0u;sample<4u;sample\+=1u\)/);
  assert.match(backend, /aperture=clamp\(\.5\+rigidSdf\(body,world\)/);
  const host = read("../lib/webgpu-octree.ts");
  assert.match(host, /encodeRigidBoundaryRefresh\(broker\)[\s\S]*encodeAdvection\(broker, step\)/);
  const integrator = read("../lib/webgpu-uniform-eulerian.ts");
  assert.match(integrator, /this\.rigidSystem\.encode\(encoder, dt, cellVolume, 1/);
});

test("resident fine phi is carved by the analytic rigid SDF before redistance", () => {
  assert.match(fineLevelSetRigidCarveWGSL,
    /fineWritePackedPhi\(index,max\(old,-solidPhi\)\)/);
  const host = read("../lib/webgpu-octree.ts");
  assert.match(host,
    /rigidCarve\?\.encode\(redistanceBroker\);\s*pending\.redistance\.encode/);
  assert.match(host,
    /this\.scene\.rigidBodies\.length > 0[\s\S]*pending\.volume\.encodeMeasurement/);
});

test("pressure reaction owns buoyancy while occupancy drives drag and added mass", () => {
  const reaction = read("../lib/webgpu-octree-losasso-rigid-pressure-reaction.ts");
  assert.match(reaction, /atomicMax\(&rigidExchange\[base\+10u\],1\)/);
  assert.match(reaction, /addWetSurfaceEstimate/);
  assert.match(reaction, /let blocked=1\.-face\.openFraction/);
  const rigid = read("../lib/webgpu-rigid-body.ts");
  assert.match(rigid, /pressureCoupled=atomicLoad\(&exchange\[base\+10u\]\)!=0/);
  assert.match(rigid, /buoyancy=select\(-rho\*displaced\*params\.gravity\.xyz,vec3f\(0\),pressureCoupled\)/);
});

test("factor-four fine normals and MacCormack remain opt-in", () => {
  assert.equal(fineSurfaceFilteredNormalsEnabled(1, {}), true);
  assert.equal(fineSurfaceFilteredNormalsEnabled(4, {}), false);
  assert.equal(fineSurfaceFilteredNormalsEnabled(4, { [FLUID_FINE_FILTERED_NORMALS_ENV]: "1" }), true);
  assert.equal(coarsePhiBFECCEnabled(4, {}), false);
  assert.equal(coarsePhiBFECCEnabled(4, { [FLUID_COARSE_PHI_BFECC_ENV]: "1" }), true);
  assert.equal(coarsePhiBFECCEnabled(8, { [FLUID_COARSE_PHI_BFECC_ENV]: "1" }), false);
  assert.match(octreeLosassoFineTransportWGSL, /fn reverseLosassoFinePhi/);
  assert.match(octreeLosassoFineTransportWGSL, /fn correctLosassoFinePhi/);
  const constructor = WebGPUOctreeLosassoFineTransport.toString().replace(/\s+/g, "");
  assert.match(constructor, /this\.storageBindingCount=10/);
  assert.match(constructor, /reverseLosassoFinePhi[\s\S]*layout:"auto"/);
  assert.match(constructor, /dispatchWorkgroupsIndirect\(this\.liveDispatch,0\)/);
  const host = read("../lib/webgpu-octree-losasso-fine-transport.ts");
  const mainLayout = host.slice(host.indexOf("Losasso fine transport reduced layout"),
    host.indexOf("this.pipelineLayout"));
  assert.doesNotMatch(mainLayout, /binding: 12/,
    "the optional reverse field must not push the main transport over ten storage bindings");
});

test("fine sizing evidence is advected, decayed, and reaches pressure refinement", () => {
  assert.match(octreeLosassoFineTransportWGSL,
    /evaluatedSizingScore[\s\S]*laplacian[\s\S]*diagonalDerivative/);
  assert.match(octreeLosassoFineTransportWGSL,
    /max\(advectedSizingScore\(departure\),evaluatedSizingScore\(departure,transported\)\)/);
  assert.match(octreeLosassoFineTransportWGSL, /pow\(\.9,p\.dt\/\.01\)/);
  const summary = read("../lib/webgpu-octree-fine-levelset-summary-direct.ts");
  assert.match(summary, /SIZING_REFINEMENT:u32=0x40000000u/);
  assert.match(summary, /metadata\[4u\*page\+3u\]&PAGE_SIZING_MASK/);
  const projection = read("../lib/webgpu-octree.ts");
  assert.match(projection, /if \(summary\.sizingRefinement\) \{ return true; \}/);
});

test("HUD diagnostics expose per-source frame counts", () => {
  const pipeline = read("../lib/webgpu-water-pipeline.ts");
  const renderer = read("../lib/webgpu-renderer.ts");
  const panel = read("../components/DiagnosticsPanel.tsx");
  assert.match(pipeline, /surfaceSourceFrameCounts/);
  assert.match(renderer, /sourceFrameCounts: water\.sourceFrameCounts/);
  assert.match(panel, /frames fine/);
});

test("new rigid coupling shaders validate as WGSL", () => {
  for (const [name, source] of [
    ["fine-rigid-carve", fineLevelSetRigidCarveWGSL],
    ["losasso-face-publication", webgpuOctreeLosassoBackendWGSL],
    ["losasso-conditioned-operator", octreeLosassoConditionedOperatorWGSL],
    ["losasso-fine-transport", octreeLosassoFineTransportWGSL],
  ] as const) {
    const result = spawnSync(process.env.NAGA ?? "naga",
      ["--stdin-file-path", `${name}.wgsl`], { input: source, encoding: "utf8" });
    assert.equal(result.status, 0, `${name}: ${result.stderr || result.stdout}`);
  }
});
