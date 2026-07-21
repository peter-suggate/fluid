import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { planOctreeSurfaceStateAllocation } from "../lib/octree-surface-allocation";
import { WebGPUQuadtreeSurfaceState } from "../lib/webgpu-quadtree-builder";

test("adaptive surface authority retains one publication phi and removes every legacy volume-scaled auxiliary", () => {
  const plan = planOctreeSurfaceStateAllocation([320, 96, 80], true);
  assert.equal(plan.cellCount, 2_457_600);
  assert.equal(plan.publicationBytes, 9_830_400);
  assert.equal(plan.denseBaselineBytes, 78_643_200);
  assert.equal(plan.allocatedBytes, plan.publicationBytes);
  assert.equal(plan.persistentAllocatedBytes, plan.publicationBytes);
  assert.equal(plan.savedBytes, 68_812_800);
  assert.equal(plan.legacyAuxiliaryBytes, 3 * plan.publicationBytes + 2 * plan.cellCount * 8);
});

test("direct paged authority retains only one format-compatible phi texel after bootstrap", () => {
  const plan = planOctreeSurfaceStateAllocation([320, 96, 80], true, true);
  assert.equal(plan.publicationBytes, 9_830_400);
  assert.equal(plan.persistentPublicationBytes, 4);
  assert.equal(plan.allocatedBytes, 9_830_400, "bootstrap peak includes the uploaded dense field");
  assert.equal(plan.persistentAllocatedBytes, 4);
  assert.equal(plan.savedBytes, plan.denseBaselineBytes - 4);
});

test("analytic sparse bootstrap never allocates the box-sized publication peak", () => {
  const plan = planOctreeSurfaceStateAllocation([320, 96, 80], true, true, true);
  assert.equal(plan.publicationBytes, 9_830_400, "dense compatibility size remains measurable");
  assert.equal(plan.allocatedBytes, 4);
  assert.equal(plan.persistentAllocatedBytes, 4);
  assert.equal(plan.savedBytes, plan.denseBaselineBytes - 4);
});

test("surface-state savings scale with volume for the large target domain", () => {
  const medium = planOctreeSurfaceStateAllocation([320, 96, 80], true);
  const large = planOctreeSurfaceStateAllocation([640, 192, 160], true);
  assert.equal(large.cellCount, medium.cellCount * 8);
  assert.equal(large.publicationBytes, 78_643_200);
  assert.equal(large.denseBaselineBytes, 629_145_600);
  assert.equal(large.savedBytes, 550_502_400);
  assert.equal(large.savedBytes, medium.savedBytes * 8);
});

test("legacy mode accounts all four phi textures and both seed arenas", () => {
  const plan = planOctreeSurfaceStateAllocation([7, 5, 3], false);
  assert.equal(plan.allocatedBytes, 7 * 5 * 3 * 32);
  assert.equal(plan.savedBytes, 0);
  assert.throws(() => planOctreeSurfaceStateAllocation([0, 5, 3], true), RangeError);
});

test("presentation-only dense phi is released exactly once after bootstrap submission", async () => {
  let destroys = 0;
  const state = {
    presentationOnly: true,
    presentationTextureReleased: false,
    device: { queue: { onSubmittedWorkDone: () => Promise.resolve() } },
    texture: { destroy: () => { destroys += 1; } },
    dims: { nx: 320, ny: 96, nz: 80 },
  };
  const release = WebGPUQuadtreeSurfaceState.prototype.releasePresentationTexture;
  assert.equal(release.call(state as never), 9_830_400);
  assert.equal(release.call(state as never), 0);
  await Promise.resolve();
  assert.equal(destroys, 1);
});

test("format-only phi placeholder remains live until solver destruction", async () => {
  let destroys = 0;
  const state = {
    presentationOnly: true,
    placeholderOnly: true,
    presentationTextureReleased: false,
    device: { queue: { onSubmittedWorkDone: () => Promise.resolve() } },
    texture: { destroy: () => { destroys += 1; } },
    dims: { nx: 60, ny: 45, nz: 40 },
  };
  const release = WebGPUQuadtreeSurfaceState.prototype.releasePresentationTexture;
  assert.equal(release.call(state as never), 0);
  await Promise.resolve();
  assert.equal(destroys, 0, "recurring bind groups retain the format placeholder");
  assert.equal(state.presentationTextureReleased, false);
});

test("solver retires bootstrap phi only after submission and exposes the 1x1 fallback", () => {
  const octree = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  assert.match(uniform, /queue\.submit\(\[encoder\.finish\(\)\]\);[\s\S]*?releaseDenseBootstrapPhi\(\)/,
    "the dense bootstrap texture must remain alive through submission");
  assert.match(octree, /get levelSetTexture\(\) \{ return this\.denseBootstrapPhiReleased \? this\.levelSetFallbackTexture! : this\.surfaceState\.texture; \}/);
  assert.match(octree, /this\.pagedPhiDifferential \|\| this\.diagnosticGroups[\s\S]*?this\.globalFineLevelSet && !this\.globalFineBootstrapped[\s\S]*?this\.scene\.rigidBodies\.length > 0 \|\| sceneHasTerrain\(this\.scene\)/,
    "dense-only diagnostic and solid-coupling paths must retain their publication");
  assert.match(octree,
    /if \(this\.directPagedTopology && !this\.surfacePagesBootstrapped && this\.pagedGroups\)/,
    "global-fine mode must perform the same one-texel paged-phi rebinding before release");
  assert.doesNotMatch(octree,
    /this\.directPagedTopology && !this\.globalFineLevelSet && !this\.surfacePagesBootstrapped/);
  assert.match(octree,
    /this\.adaptiveSurfacePages\?\.releaseDensePublicationBinding\(\);[\s\S]*this\.surfaceState\.releasePresentationTexture\(\)/,
    "the one-shot dense publication bind group must be dropped before destroying its texture");
  assert.match(octree,
    /this\.couplingGroups = \{[\s\S]*this\.createCouplingGroup\(this\.pressureA, this\.levelSetFallbackTexture!\)[\s\S]*this\.createCouplingGroup\(this\.pressureB, this\.levelSetFallbackTexture!\)/,
    "dormant no-body coupling groups must not retain views of the destroyed dense texture");
  const couplingGroup = octree.slice(octree.indexOf("private createCouplingGroup"), octree.indexOf("private descriptor"));
  assert.match(couplingGroup, /binding: 6, resource: levelSet\.createView\(\)/);
  assert.equal((couplingGroup.match(/\{ binding:/g) ?? []).length, 9,
    "fallback rebinding must reuse the nine-binding coupling layout without increasing stage limits");
});

const modulePath = process.env.WEBGPU_NODE_MODULE;

test("Dawn presentation-only surface state allocates no legacy transport volumes or seeds", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU surface allocation checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  const textureLabels: string[] = [], bufferLabels: string[] = [], shaderLabels: string[] = [];
  const wrapped = new Proxy(device, {
    get(target, property) {
      if (property === "createTexture") return (descriptor: GPUTextureDescriptor) => {
        textureLabels.push(String(descriptor.label ?? "")); return target.createTexture(descriptor);
      };
      if (property === "createBuffer") return (descriptor: GPUBufferDescriptor) => {
        bufferLabels.push(String(descriptor.label ?? "")); return target.createBuffer(descriptor);
      };
      if (property === "createShaderModule") return (descriptor: GPUShaderModuleDescriptor) => {
        shaderLabels.push(String(descriptor.label ?? "")); return target.createShaderModule(descriptor);
      };
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  }) as GPUDevice;
  const velocity = device.createTexture({
    size: [1, 1, 1], dimension: "3d", format: "rgba32float",
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  const state = new WebGPUQuadtreeSurfaceState(
    wrapped, { nx: 4, ny: 3, nz: 2 }, { x: 1, y: 1, z: 1 }, velocity,
    new Float32Array(24).fill(-0.5), undefined, undefined, false, false, true, true,
    undefined, undefined, true,
  );
  try {
    assert.deepEqual(textureLabels, ["Resident quadtree level set"]);
    assert.deepEqual(bufferLabels, []);
    assert.deepEqual(shaderLabels, [], "presentation authority must not compile the dormant dense surface pipeline");
    assert.equal((await state.readVolumeDiagnostics()).volumeCells, 15);
  } finally {
    state.destroy(); velocity.destroy(); device.destroy();
  }
});
