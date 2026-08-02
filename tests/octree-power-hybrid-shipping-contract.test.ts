import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  decodeOctreePowerHybridRuntimeCensus,
  octreePowerHybridWorkVerdict,
} from "../lib/octree-power-hybrid";
import {
  octreePowerHybridDryIdentityEnabled,
} from "../lib/webgpu-octree-structured-boundary";
import {
  OCTREE_PERSISTENT_MGPCG_HYBRID_CENSUS_MARKER,
  decodeOctreePersistentMGPCGHybridCensus,
} from "../lib/webgpu-octree-persistent-mgpcg";
import { octreePersistentMGPCGWGSL } from "../lib/webgpu-octree-persistent-mgpcg.wgsl";

const source = (file: string) => readFileSync(new URL(`../lib/${file}`, import.meta.url), "utf8");

function header(epoch: number, count: number, capacity: number): Uint32Array {
  const blocks = Math.ceil(count / 64);
  const x = Math.max(1, Math.min(65_535, blocks));
  return Uint32Array.of(epoch, count, capacity, 3, x, Math.ceil(blocks / x), 1);
}

test("runtime census accepts only an exact four-class row partition", () => {
  const members = [Uint32Array.of(0, 2, 4, 6, 8, 10, 11, 12), Uint32Array.of(1),
    Uint32Array.of(3, 5), Uint32Array.of(7, 9)];
  const headers = members.map((list) => header(41, list.length, 13));
  const census = decodeOctreePowerHybridRuntimeCensus({
    acceptedRows: 13, acceptedEpoch: 41, headers, members,
  });
  assert.ok(census);
  assert.deepEqual(census.classCounts, [8, 1, 2, 2]);
  assert.equal(census.regularRows, 8);
  assert.equal(census.powerRows, 5);
  assert.equal(census.fullPageSlotChains, 18 * 13);
  assert.equal(census.hybridPageSlotChains, 18 * 5);
  assert.equal(census.machineryReduction, 13 / 5);

  const stale = headers.map((value) => value.slice()); stale[2]![0] = 40;
  assert.equal(decodeOctreePowerHybridRuntimeCensus({
    acceptedRows: 13, acceptedEpoch: 41, headers: stale, members,
  }), null);
  const duplicate = members.map((value) => value.slice()); duplicate[3]![1] = 8;
  assert.equal(decodeOctreePowerHybridRuntimeCensus({
    acceptedRows: 13, acceptedEpoch: 41, headers, members: duplicate,
  }), null, "equal total counts must not hide one duplicate plus one omission");
  const outside = members.map((value) => value.slice()); outside[3]![1] = 13;
  assert.equal(decodeOctreePowerHybridRuntimeCensus({
    acceptedRows: 13, acceptedEpoch: 41, headers, members: outside,
  }), null);

  const allRegularMembers = [Uint32Array.of(0, 1, 2), new Uint32Array(),
    new Uint32Array(), new Uint32Array()];
  const allRegular = decodeOctreePowerHybridRuntimeCensus({
    acceptedRows: 3,
    acceptedEpoch: 42,
    headers: allRegularMembers.map((list) => header(42, list.length, 3)),
    members: allRegularMembers,
  });
  assert.equal(allRegular?.powerRows, 0);
  assert.equal(allRegular?.machineryReduction, Number.POSITIVE_INFINITY);
});

test("the live work gate fails closed and scores exact expensive machinery", () => {
  assert.deepEqual(octreePowerHybridWorkVerdict(undefined, 2), {
    accepted: false, reason: "shipping GPU census was not published",
  });
  const census = {
    liveRows: 100, regularRows: 51, powerRows: 49, classCounts: [51, 0, 0, 49] as const,
    fullDescriptorRows: 100, hybridDescriptorRows: 49,
    fullCatalogLookups: 100, hybridCatalogLookups: 49,
    fullPageSlotChains: 1_800, hybridPageSlotChains: 882,
    machineryReduction: 100 / 49,
  };
  assert.deepEqual(octreePowerHybridWorkVerdict(census, 2), { accepted: true });
  const belowThreshold = { ...census, regularRows: 49, powerRows: 51,
    classCounts: [49, 0, 0, 51] as const, hybridPageSlotChains: 918, machineryReduction: 100 / 51 };
  assert.match(octreePowerHybridWorkVerdict(belowThreshold, 2).reason ?? "",
  /below 2\.000000x/);
  const corrupt = { ...census, hybridPageSlotChains: 881 };
  assert.match(octreePowerHybridWorkVerdict(corrupt, 2).reason ?? "",
    /inconsistent expensive-work totals/);
  assert.match(octreePowerHybridWorkVerdict(census, Number.NaN).reason ?? "", /minimum reduction/);

  const withDryIdentityRows = { ...census, liveRows: 1_000 };
  assert.deepEqual(octreePowerHybridWorkVerdict(withDryIdentityRows, 2), { accepted: true },
    "dry identity elimination is Bet 1 work and must not inflate the liquid-only Bet 4 score");
  const inflatedByAir = { ...belowThreshold, liveRows: 1_000 };
  assert.match(octreePowerHybridWorkVerdict(inflatedByAir, 2).reason ?? "", /below 2\.000000x/);
});

test("dry identity classification defaults on and has an isolated A/B switch", () => {
  assert.equal(octreePowerHybridDryIdentityEnabled({}), true);
  assert.equal(octreePowerHybridDryIdentityEnabled({ FLUID_POWER_HYBRID_DRY_IDENTITY: "1" }), true);
  assert.equal(octreePowerHybridDryIdentityEnabled({ FLUID_POWER_HYBRID_DRY_IDENTITY: "0" }), false);
});

test("dynamic boundary classes fail closed around every non-regular pressure face", () => {
  const text = source("webgpu-octree-structured-boundary.ts");
  const dynamics = source("webgpu-octree-structured-dynamics.ts");
  assert.match(text, /var powerBand=candidateLiquid\[row\]==0u/);
  assert.match(text, /powerBand=powerBand\|\|hi==INVALID\|\|aperture!=1\.\|\|scale!=1\.\|\|\(other<control\.rows&&candidateLiquid\[other\]!=candidateLiquid\[row\]\)/);
  assert.match(text, /\(worksetClasses\[row\]&HYBRID_POWER_BAND\)!=0u/);
  assert.match(text, /return select\(select\(0u,2u,boundary\),select\(1u,3u,boundary\),rowTransition\(row\)\)/);
  assert.match(text, /dryIdentity=liquidAt\(row\)==0u&&rows\[base\]==bitcast<u32>\(1\.\)/);
  assert.match(text, /dryIdentity=dryIdentity&&rows\[base\+channel\]==0u/);
  assert.match(text, /candidateLiquid\[other\]!=candidateLiquid\[row\]/,
    "a liquid row incident on dry identity remains in the power seam");
  assert.match(dynamics, /if\(liquidAt\(row\)==0u\)\{rhs\[row\]=0\.;return;\}/,
    "the dynamics publisher must author exact zero RHS for every dry identity row");
  assert.match(dynamics, /if\(liquidAt\(lo\)!=0u\)\{pressureLo=pressure\[lo\];\}/);
  assert.match(dynamics, /if\(hi!=INVALID&&liquidAt\(hi\)!=0u\)\{pressureHi=pressure\[hi\];\}/,
    "projection coupling must never sample a dry pressure unknown");
  assert.match(dynamics, /if\(row>=acc\(2u\)\|\|liquidAt\(row\)==0u\)\{return true;\}/,
    "dry rows must be excluded from wet projection-energy attribution");
});

test("topology case zero bypasses the per-row descriptor/catalog walk", () => {
  const text = source("webgpu-octree-power-topology.ts");
  const prepare = text.indexOf("if(!rowTemplateValid(0u))");
  const resolve = text.indexOf("fn resolvePowerTopology");
  const regular = text.indexOf("if(geometry==REGULAR_DESCRIPTOR&&boundary==0u)", resolve);
  const general = text.indexOf("let found=resolveDescriptor(descriptor)", resolve);
  assert.ok(prepare >= 0 && resolve >= 0 && regular > resolve && general > regular);
  assert.doesNotMatch(text.slice(regular, general), /rowTemplateValid|resolveDescriptor/);
});

test("regular operator-image rows skip the general page-slot chain but keep dynamic coefficients", () => {
  const text = source("webgpu-octree-spgrid-vcycle.ts");
  const begin = text.indexOf("fn resolveRegularChannel");
  const end = text.indexOf("fn reportChannelCode", begin);
  assert.ok(begin >= 0 && end > begin);
  const body = text.slice(begin, end);
  assert.match(body, /absolute\.x\+absolute\.y\+absolute\.z!=1/);
  assert.match(body, /pageNeighbour/);
  assert.match(body, /brickRecord/);
  assert.doesNotMatch(body, /pageSlotCoded|resolveDirectChannel/);
  assert.match(text, /if\(regular\)\{operatorRows\[image\+word\]=resolveRegularChannel/,
    "WGSL select evaluates both arms and would retain the general walk");
  assert.match(text, /let c=section63Coefficients\[base\+1u\+channel\]/,
    "regular and seam terms must retain the boundary-authored coefficient");
  assert.match(text, /operatorRows\[image\]&~HYBRID_REGULAR_ROW/);
});

test("shipping persistent solver consumes the exact dynamic boundary memberships", () => {
  const kernel = octreePersistentMGPCGWGSL({ maximumIterations: 10 });
  const host = source("webgpu-octree-persistent-mgpcg.ts");
  const owner = source("webgpu-octree.ts");
  assert.match(kernel, /binding\(9\).*dynamicWorksets/);
  assert.match(kernel, /binding\(10\).*acceptedBoundary/);
  assert.match(kernel, /fn validateAcceptedRowPartition\(\)->u32/);
  assert.match(kernel, /for\(var expected=0u;expected<rows\(\);expected\+=1u\)/);
  assert.match(kernel, /if\(best!=expected\|\|owner==INVALID\)\{return 0x201u;\}/);
  assert.match(kernel, /if\(cls==0u\)\{applyRegularRow\(row,inCh,outCh\);\}else if\(cls==4u\)\{applyIdentityRow\(row,inCh,outCh\);\}else\{applyRow\(row,inCh,outCh\);\}/);
  assert.match(kernel, /coefficients\[base\]!=1\.0\|\|vload\(CH_RHS,row\)!=0\.0/);
  assert.match(kernel, /coefficients\[base\+1u\+channel\]!=0\.0/);
  const regular = kernel.slice(kernel.indexOf("fn applyRegularRow"), kernel.indexOf("fn applyAllRows"));
  assert.match(regular, /channel=0u;channel<6u/);
  assert.match(regular, /coefficients\[base\+1u\+channel\]/);
  assert.doesNotMatch(regular, /opPageSlot|finerAdjoint|resolveDescriptor/);
  assert.doesNotMatch(host.slice(host.indexOf("encodeSolve("), host.indexOf("private bindGroup")),
    /source\.acceptedAuthority|source\.worksets/,
    "accepted classes are bound live; they must not regress to staging copies");
  assert.match(host, /0, 0, 0, 0,\s*source\.worksetStrideWords, source\.worksetBankStrideWords, 0, 0,/,
    "the workset strides must follow, rather than be overwritten by, numerics vec4f");
  assert.match(owner, /acceptedAuthority: this\.structuredBoundary\.control/);
  assert.match(owner, /worksets: this\.structuredBoundary\.worksets/);
  assert.doesNotMatch(owner.slice(owner.indexOf("this\.persistentMGPCG = new"),
    owner.indexOf("this\.topologyEpoch = new")), /section63Source\.worksets/);
});

test("shipping census decodes only internally consistent GPU-proved work", () => {
  const words = Uint32Array.of(900, 400, 100, 1_000, 100,
    18_000, 1_800, 73, 5, OCTREE_PERSISTENT_MGPCG_HYBRID_CENSUS_MARKER);
  const census = decodeOctreePersistentMGPCGHybridCensus(words);
  assert.ok(census);
  assert.equal(census.identityRows, 400);
  assert.equal(census.liquidRows, 1_000);
  assert.equal(census.liveRows, 1_400);
  assert.equal(census.machineryReduction, 10);
  const corrupt = words.slice(); corrupt[6] += 1;
  assert.equal(decodeOctreePersistentMGPCGHybridCensus(corrupt), null);
  const wrongClassCount = words.slice(); wrongClassCount[8] = 4;
  assert.equal(decodeOctreePersistentMGPCGHybridCensus(wrongClassCount), null);
  const unstamped = words.slice(); unstamped[9] = 0;
  assert.equal(decodeOctreePersistentMGPCGHybridCensus(unstamped), null);
});

test("Dawn compiles the shipping persistent hybrid kernel", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for persistent hybrid validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: octreePersistentMGPCGWGSL({ maximumIterations: 10 }) });
  const info = await module.getCompilationInfo();
  assert.deepEqual(info.messages.filter((message) => message.type === "error")
    .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`), []);
  device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "persistentMGPCG" } });
  const error = await device.popErrorScope();
  assert.equal(error, null, error?.message);
  device.destroy();
});
