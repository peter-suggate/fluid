import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { NBFLIP_STAIR_CUBE_ID } from "../lib/core/nbflip-stair-cube-scene";
import { getScenePreset } from "../lib/core/scenes";
import { interactionScene } from "../lib/interaction/adapters/scene";
import { createUniformTroughScene } from "../lib/core/uniform-trough-scenes";
import type { SceneDescription } from "../lib/core/model";
import { auditUniformGPUAllocations, uniformGPUAllocationClass } from "../tools/uniform-gpu-allocation-audit";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

// The census start's construction (docs/plans/bulk16-plan-2026-10-28.md, V2b): the h stores hold
// slots from the start, sized by the slot rule, so no h detail texture of the whole block grid at h
// (the h start's dense store) is created at any point of construction. The h start is the control:
// its h-resolution detail textures (at least the box's cells; not the 4h fields or the 1-texel
// stubs) are dense, and the census start's of the same label is smaller.
const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };

type Built = { largest: Record<string, number>; peak: number; detail: number; cells: number };
async function construct(scene: SceneDescription, values: Record<string, unknown>): Promise<Built> {
  let built: Built | undefined;
  await withUniformDevice("census construction", async raw => {
    // The largest h detail texture of each label created, live or not by the end.
    const largest: Record<string, number> = {};
    const audit = auditUniformGPUAllocations(raw);
    const device = new Proxy(audit.device, { get(target, key) {
      if (key === "createTexture") return (d: GPUTextureDescriptor) => {
        const e = Symbol.iterator in d.size ? [...d.size as Iterable<number>] : [(d.size as GPUExtent3DDict).width, (d.size as GPUExtent3DDict).height ?? 1, (d.size as GPUExtent3DDict).depthOrArrayLayers ?? 1];
        const label = d.label ?? "", texels = e.reduce((n, x) => n * x, 1);
        if (uniformGPUAllocationClass(label) === "h detail fields") largest[label] = Math.max(largest[label] ?? 0, texels);
        return target.createTexture({ ...d, size: e });
      };
      const value = Reflect.get(target, key);
      return typeof value === "function" ? value.bind(target) : value;
    } });
    const solver = await createUniformSolver(device, scene, values as never);
    try {
      assert.equal(solver.info.simulationPipelineError, undefined);
      const s = audit.snapshot();
      const detail = s.resources.filter(r => uniformGPUAllocationClass(r.label) === "h detail fields").reduce((n, r) => n + r.bytes, 0);
      built = { largest, peak: s.peakBytes, detail, cells: solver.info.nx * solver.info.ny * solver.info.nz };
    } finally { solver.destroy(); }
  });
  return built!;
}

const scenes: readonly [string, () => SceneDescription, Record<string, unknown>][] = [
  ["the sealed rigid vessel (stair cube)", () => interactionScene(getScenePreset(NBFLIP_STAIR_CUBE_ID).create()), { rigidCoupling: "on" }],
  ["the trough dam-break", () => createUniformTroughScene("dam-break"), { volumeControlSeconds: 0 }],
];
for (const [name, make, values] of scenes) {
  test(`a census start creates no dense h detail texture during construction: ${name}`, gpu, async () => {
    const at = (start: "h" | "census") => { const scene = make(); scene.fluid.tiers = { ...scene.fluid.tiers, start }; return construct(scene, values); };
    const h = await at("h"), census = await at("census");
    console.log(JSON.stringify({ scene: name, h, census }));
    const dense = Object.keys(h.largest).filter(label => h.largest[label]! >= h.cells);
    for (const label of dense)
      assert.ok(census.largest[label] !== undefined && census.largest[label]! < h.largest[label]!, `${label}: census ${census.largest[label]} texels against the h start's ${h.largest[label]}`);
    for (const [label, texels] of Object.entries(census.largest)) assert.ok(texels < h.cells || dense.includes(label), `${label}: census ${texels} texels, as large as the box (${h.cells} cells)`);
    assert.ok(dense.length > 0, "the h start's dense h stores");
    assert.ok(census.peak < h.peak, `construction peak: census ${census.peak} against h ${h.peak}`);
  });
}
