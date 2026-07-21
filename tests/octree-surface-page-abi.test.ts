import assert from "node:assert/strict";
import test from "node:test";
import {
  planOctreeSurfacePages,
  validateOctreeSurfacePagePlan,
  validateOctreeSurfacePageSource,
} from "../lib/webgpu-octree-surface-pages";

test("single host ABI validator accepts exactly coherent 2-cubed and 4-cubed plans", () => {
  for (const pageResolution of [2, 4] as const) {
    const plan = planOctreeSurfacePages(8, [8, 8, 8], { pageResolution, maximumPages: 2 });
    assert.doesNotThrow(() => validateOctreeSurfacePagePlan(plan));
    assert.equal(plan.samplesPerPage, pageResolution ** 3);
  }
  const plan = planOctreeSurfacePages(8, [8, 8, 8], { maximumPages: 2 });
  assert.throws(() => validateOctreeSurfacePagePlan({ ...plan, samplesPerPage: 64 }), /cubed/);
  assert.throws(() => validateOctreeSurfacePagePlan({ ...plan, phiBOffsetWords: plan.phiBOffsetWords + 1 }), /offsets/);
});

test("binding validator rejects shape/offset drift before GPU fallback can hide it", () => {
  const plan = planOctreeSurfacePages(8, [8, 8, 8], { maximumPages: 2 });
  const buffer = {} as GPUBuffer;
  const source = {
    plan, arena: { buffer }, leaves: { buffer }, params: { buffer },
    phiAOffsetBytes: plan.phiAOffsetWords * 4, pageTableOffsetBytes: plan.pageTableOffsetWords * 4,
  } as Parameters<typeof validateOctreeSurfacePageSource>[0];
  assert.doesNotThrow(() => validateOctreeSurfacePageSource(source));
  assert.throws(() => validateOctreeSurfacePageSource({ ...source, phiAOffsetBytes: source.phiAOffsetBytes + 4 }), /byte offsets/);
  assert.throws(() => validateOctreeSurfacePageSource({ ...source, arena: undefined as never }), /binding is missing/);
});
