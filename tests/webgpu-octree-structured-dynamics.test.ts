import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { unpackOctreePowerRowTemplateSlot } from "../lib/octree-power-catalog";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { structuredBoundaryCoefficientWGSL } from "../lib/webgpu-octree-structured-boundary";
import {
  decodeStructuredProjectionEnergy,
  STRUCTURED_PROJECTION_ENERGY_WORDS,
  structuredVelocityDynamicsWGSL,
} from "../lib/webgpu-octree-structured-dynamics";

const dynamicsSource = readFileSync(new URL("../lib/webgpu-octree-structured-dynamics.ts", import.meta.url), "utf8");
const dynamicsHost = dynamicsSource.slice(0, dynamicsSource.indexOf("export const structuredVelocityDynamicsWGSL"));

/** Every `name[...]` subscript expression in a WGSL source, in source order. */
function indexExpressions(wgsl: string, name: string): readonly string[] {
  const found: string[] = [];
  const opening = new RegExp(String.raw`(?<![A-Za-z0-9_])${name}\[`, "g");
  for (let match = opening.exec(wgsl); match; match = opening.exec(wgsl)) {
    let depth = 0, end = opening.lastIndex;
    while (end < wgsl.length && !(wgsl[end] === "]" && depth === 0)) {
      if (wgsl[end] === "[") depth += 1; else if (wgsl[end] === "]") depth -= 1;
      end += 1;
    }
    found.push(wgsl.slice(opening.lastIndex, end));
  }
  return found;
}

/** Evaluate a stage's `sbase()` body with its own uniform names resolved, so a
 * reader and a writer that disagree about the slot bank stride cannot both pass. */
function slotBankBase(wgsl: string, bank: number, slotCapacity: number): number {
  const body = /fn sbase\(\)->u32\{return ([^;]+);\}/.exec(wgsl);
  assert.ok(body, "both structured stages must name the slot bank base `sbase()`");
  const resolved = body[1]!.replace(/control\.bank|bank\(\)/g, String(bank))
    .replace(/p\.counts\.y|p\.slotCapacity/g, String(slotCapacity));
  assert.match(resolved, /^[0-9* ]+$/,
    `the slot bank base must be a pure bank-times-slot-capacity product, got ${body[1]}`);
  return resolved.split("*").reduce((product, term) => product * Number(term), 1);
}

const entries = ["prepareStructuredDynamics",
  "summarizeStructuredPreProjectionEnergy", "summarizeStructuredPostProjectionEnergy",
  ...[5, 6, 7, 8].flatMap((value) => [`advectStructuredClass${value}`,
    `commitAdvectedStructuredClass${value}`, `forceStructuredClass${value}`,
    `projectStructuredClass${value}`]),
  ...[0, 1, 2, 3].flatMap((value) => [`divergenceStructuredClass${value}`,
    `separateStructuredClass${value}`, `reconstructStructuredClass${value}`]),
];

test("structured dynamics owns destination writes and has no general face/incidence graph", () => {
  assert.match(structuredVelocityDynamicsWGSL,
    /worksetBankStride:u32,dimensionX:u32,dimensionY:u32,dimensionZ:u32,closedMask:u32/,
    "uniform dimensions stay scalar-packed so all authority offsets retain their host word indices");
  assert.doesNotMatch(structuredVelocityDynamicsWGSL, /dimensions:vec3u/,
    "a vec3 uniform member would insert two hidden words before every authority offset");
  assert.doesNotMatch(structuredVelocityDynamicsWGSL, /PowerFaceRecord|incidence|atomicAdd/i);
  assert.match(structuredVelocityDynamicsWGSL, /fn regularSample\(/);
  assert.match(structuredVelocityDynamicsWGSL,
    /sampleX=clamp\(x,vec3f\(\.5\*h\),vec3f\(d\)\*p\.physical\.x-vec3f\(\.5\*h\)\)/,
    "regular transport must use the explicit constant physical-boundary extension");
  assert.match(structuredVelocityDynamicsWGSL,
    /let weight=[^;]+;[\s\S]*if\(weight<=0\.\)\{continue;\}[\s\S]*taggedVelocity\(regularTag\(anchor,offset\)\)/,
    "zero-weight exterior corners must not be mistaken for missing live topology");
  assert.doesNotMatch(structuredVelocityDynamicsWGSL, /fn axisNeighbor[^}]*for\s*\(/,
    "regular interpolation must load publisher-resolved axis handles in O(1)");
  assert.match(structuredVelocityDynamicsWGSL, /p\.rowAxisOffset\+6u\*row\+direction/);
  assert.match(structuredVelocityDynamicsWGSL, /fn transitionSample\(/);
  assert.match(structuredVelocityDynamicsWGSL,
    /selectorAt=p\.selectorOffsetWords\+row\*p\.selectorStride\+selectorIndex[\s\S]*other=supportWord\(selectorAt\)/,
    "transition interpolation must resolve the complete Delaunay selector set, including non-face vertices");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn supportPublicationValid\(\)[\s\S]*supportWord\(base\+2u\)==acc\(3u\)[\s\S]*supportWord\(base\+3u\)==bank\(\)[\s\S]*supportWord\(base\+4u\)==boundaryControl\[4u\][\s\S]*SUPPORT_VALID/,
    "every tagged velocity must be generation-, bank-, and boundary-coherent");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn taggedVelocity\(tag:u32\)[\s\S]*tag==INVALID\|\|!supportPublicationValid\(\)[\s\S]*tag&SUPPORT_TAG[\s\S]*supportVectorOffsetWords/,
    "support tags remain unusable until a complete support-vector publication exists");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn advect\([\s\S]*if\(!supportPublicationValid\(\)\)\{transportMetrics\[handle\]=invalidVector\(\);rejectSample\(1u,handle\);return;\}/,
    "every momentum destination must reject an incomplete face-extension transaction, even when its local interpolant needs only direct rows");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn regularTag\(row:u32,offset:vec3i\)[\s\S]*regularTagOffsetWords\+27u\*row/,
    "regular trilinear corners consume the direct 3x3x3 tag publication");
  assert.doesNotMatch(structuredVelocityDynamicsWGSL,
    /fn selectorVelocity[\s\S]*for\(var local=0u;local<h\.y/,
    "transition interpolation must not mistake the power-face list for the Delaunay vertex adjacency");
  assert.match(structuredVelocityDynamicsWGSL,
    /positive=max\(weights,vec4f\(0\.\)\)[\s\S]*positive\/=positiveSum[\s\S]*if\(positive\.y>0\.\)\{let v1=selectorVelocity[\s\S]*if\(positive\.w>0\.\)\{let v3=selectorVelocity/,
    "zero-weight tetra vertices must not require velocity extrapolation, while positive contributors remain strict");
  assert.match(structuredVelocityDynamicsWGSL, /fn divergenceRow\(/);
  assert.match(structuredVelocityDynamicsWGSL, /fn projectFamily\(/);
  assert.match(structuredVelocityDynamicsWGSL,
    /transportMetrics\[handle\]=vec4f\(projected\*n\*area,.5\*area\*max\(0.,prior\*prior-projected\*projected\)\)/,
    "each destination retains failure-local transported momentum and dissipation evidence");
  assert.doesNotMatch(structuredVelocityDynamicsWGSL,
    /summarizeStructuredTransportMetrics|transportStats|metricRegular|metricTransition/,
    "unused O(N) transport diagnostics must not run on every substep");
  assert.match(structuredVelocityDynamicsWGSL, /fn boundaryValid\(\)/);
  assert.match(structuredVelocityDynamicsWGSL,
    /boundaryControl\[6\]==acc\(3u\)[\s\S]*worksets\[base\]!=acc\(3u\)/,
    "hot dynamics must consume only the current committed dynamic boundary worksets");
});

test("transition barycentric sampling requires non-face selector adjacency", () => {
  const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  const catalog = decodeGeneratedOctreePowerCatalog(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const caseId = 6999;
  const tetraFirst = catalog.tetrahedronHeaders[caseId * 3]!;
  const tetraCount = catalog.tetrahedronHeaders[caseId * 3 + 1]!;
  const tetrahedra = Array.from({ length: tetraCount }, (_, local) => {
    const packed = catalog.tetrahedronData[tetraFirst + local]!;
    return [packed & 0xff, (packed >>> 8) & 0xff, (packed >>> 16) & 0xff];
  });
  const slotFirst = catalog.rowTemplateHeaders[caseId * 4]!;
  const slotCount = catalog.rowTemplateHeaders[caseId * 4 + 1]!;
  const faceSelectors = Array.from(
    catalog.rowTemplateSlots.slice(slotFirst, slotFirst + slotCount),
    (packed) => unpackOctreePowerRowTemplateSlot(packed).neighborSelector,
  );

  assert.deepEqual(faceSelectors, [30, 14, 34, 36, 37, 67]);
  assert.deepEqual(tetrahedra[8], [15, 67, 74]);
  assert.deepEqual(tetrahedra[8]!.filter((selector) => !faceSelectors.includes(selector)), [15, 74],
    "the accepted transition sample cannot be reconstructed from power-face row slots alone");
});

test("projection energy uses one coherent face-weighted pair around projection", () => {
  // Four eight-word stage records: start-of-step (post-remap), post-advection,
  // post-force (pre-projection), post-projection; each also carries the
  // wet-only sum and the stage-1 sampler-path census. The host cross-checks
  // generation, bank, and family coverage across all four.
  assert.equal(STRUCTURED_PROJECTION_ENERGY_WORDS, 32);
  assert.match(structuredVelocityDynamicsWGSL,
    /binding\(23\)var<storage,read_write>projectionEnergyStats:array<u32>/);
  assert.match(structuredVelocityDynamicsWGSL,
    /dualVolume=area\/inverseDistance[\s\S]*\.5\*aperture\*dualVolume\*sample\*sample/,
    "all stage reductions must use the identical open-face kinetic-energy measure");
  const startAt = dynamicsHost.indexOf('this.encodeProjectionEnergy(broker, params, "start")');
  const advectAt = dynamicsHost.indexOf("Advect structured family class");
  const advectedAt = dynamicsHost.indexOf('this.encodeProjectionEnergy(broker, params, "advected")');
  const forceAt = dynamicsHost.indexOf("Force and constrain structured family class");
  const preAt = dynamicsHost.indexOf('this.encodeProjectionEnergy(broker, params, "pre")');
  const divergenceAt = dynamicsHost.indexOf("Fuse structured divergence RHS class");
  const projectionAt = dynamicsHost.indexOf("Project structured family class");
  const postAt = dynamicsHost.indexOf('this.encodeProjectionEnergy(broker, params, "post")');
  const reconstructAt = dynamicsHost.indexOf("Reconstruct projected structured rows");
  assert.ok(startAt >= 0 && advectAt > startAt && advectedAt > advectAt,
    "the start probe precedes advection and the advected probe follows its commit");
  assert.ok(forceAt > advectedAt && preAt > forceAt && divergenceAt > preAt);
  assert.ok(projectionAt >= 0 && postAt > projectionAt && reconstructAt > postAt);
  assert.match(structuredVelocityDynamicsWGSL,
    /let base=8u\*stage;[\s\S]*projectionEnergyStats\[base\+1u\]=\(generation<<1u\)\|bank\(\)/,
    "every stage record self-identifies with its generation and bank for host cross-checking");
});

test("projection energy decoder fails closed on partial pairs", () => {
  const energyBits = (value: number) => new Uint32Array(new Float32Array([value]).buffer)[0]!;
  const epochAndBank = (11 << 1) | 1;
  const stage = (all: number, wet: number, theta: number, census = 0) =>
    [0, epochAndBank, 24, energyBits(all), 20, energyBits(wet), energyBits(theta), census];
  const coherent = new Uint32Array([
    ...stage(5, 4.5, 4), ...stage(4.75, 4.25, 3.75, 17), ...stage(4, 3.5, 3), ...stage(3, 2.75, 2.5),
  ]);
  assert.deepEqual(decodeStructuredProjectionEnergy(coherent), {
    sample: {
      epoch: 11, activeBank: 1, familySampleCount: 24,
      startKineticEnergyProxy: 5,
      postAdvectionKineticEnergyProxy: 4.75,
      preProjectionKineticEnergyProxy: 4,
      postProjectionKineticEnergyProxy: 3,
      wetStartKineticEnergyProxy: 4.5,
      wetPostAdvectionKineticEnergyProxy: 4.25,
      wetPreProjectionKineticEnergyProxy: 3.5,
      wetPostProjectionKineticEnergyProxy: 2.75,
      wetStartThetaEnergyProxy: 4,
      wetPostAdvectionThetaEnergyProxy: 3.75,
      wetPreProjectionThetaEnergyProxy: 3,
      wetPostProjectionThetaEnergyProxy: 2.5,
      wetFaceCount: 20,
      staggeredPathCount: 17,
      projectionEnergyRatio: 0.75,
    },
    blocker: null,
  });
  const mismatchedCount = new Uint32Array(coherent);
  mismatchedCount[8 * 3 + 2] = 23;
  const mismatchedEpoch = new Uint32Array(coherent);
  mismatchedEpoch[8 * 2 + 1] = (10 << 1) | 1;
  const failedStage = new Uint32Array(coherent);
  failedStage[8 * 1] = 2;
  for (const words of [
    coherent.subarray(0, 31),
    mismatchedCount,
    mismatchedEpoch,
    failedStage,
  ]) {
    const decoded = decodeStructuredProjectionEnergy(words);
    assert.equal(decoded.sample, null);
    assert.ok(decoded.blocker);
  }
});

test("smoke stability envelope consumes only explicit structured energy pairs", () => {
  const smoke = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  assert.match(smoke,
    /structuredEnergySamples = sample\.structuredProjectionEnergySampleCount[\s\S]*structuredEnergyRatio = sample\.structuredProjectionEnergyRatio/);
  assert.match(smoke,
    /stabilityEnvelope\.projectionEnergySampleCount \+= structuredEnergySamples/);
  assert.doesNotMatch(smoke,
    /maximumProjectionEnergyRatio[^\n]*transport(?:Stats|Dissipation)/i);
});

test("pooled stats readback decodes the exact 32-byte pair and clears rejected samples", () => {
  const projection = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  assert.match(projection,
    /get structuredProjectionEnergyStats\(\): GPUBuffer \| undefined \{[\s\S]*this\.structuredDynamics\?\.projectionEnergyStats/);
  assert.match(uniform,
    /copyBufferToBuffer\([\s\S]*structuredProjectionEnergy, 0, buffer, 16, STRUCTURED_PROJECTION_ENERGY_WORDS \* 4/);
  assert.match(uniform,
    /decodeStructuredProjectionEnergy\(new Uint32Array\([\s\S]*getMappedRange\(16, STRUCTURED_PROJECTION_ENERGY_WORDS \* 4\)/);
  assert.match(uniform,
    /structuredProjectionEnergySampleCount = 1;[\s\S]*else \{[\s\S]*structuredProjectionEnergyRatio = undefined;[\s\S]*structuredProjectionEnergySampleCount = undefined;/,
    "an incoherent pair must not retain the preceding step's energy sample");
});

test("uniform telemetry reads the exact paired projection report and clears blockers", () => {
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  const readStats = uniform.slice(uniform.indexOf("async readStats()"), uniform.indexOf("\n  destroy()"));
  assert.match(uniform, /size: 16 \+ STRUCTURED_PROJECTION_ENERGY_WORDS \* 4/,
    "the pooled readback reserves exactly one eight-word energy report");
  assert.match(readStats,
    /structuredProjectionEnergyStats[\s\S]*copyBufferToBuffer\([\s\S]*16, STRUCTURED_PROJECTION_ENERGY_WORDS \* 4/,
    "telemetry copies the producer-owned report only after submitted projection work");
  assert.match(readStats, /decodeStructuredProjectionEnergy\(/);
  assert.match(readStats,
    /structuredProjectionEnergySampleCount = 1[\s\S]*structuredProjectionEnergySampleCount = undefined/,
    "only a coherent pair becomes a sample; every blocker removes stale values");
});

test("projection leaves Section 5 extrapolation to the committed face producer", () => {
  assert.doesNotMatch(structuredVelocityDynamicsWGSL,
    /rowCpt|rowScratch|neighborAverage|extendStructuredClass/,
    "the retired row-average field cannot remain as a shadow air authority");
  assert.match(dynamicsHost, /"Reconstruct projected structured rows"/);
  assert.doesNotMatch(dynamicsHost,
    /extendAtoB|extendBtoA|rowExtensionScratch/);
});

test("newly wet topology faces inherit the accepted Section 5 air extension", () => {
  assert.match(dynamicsHost,
    /words\[54\] = resources\.airSupportLayout\.ownerDirectoryOffsetWords;[\s\S]*words\[55\] = resources\.airSupportLayout\.ownerDirectoryCellCapacity;/,
    "topology transfer must receive the same dense owner directory as fine transport");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn extendedOwnerVelocity\(point:vec3f\)[\s\S]*supportPublicationValid\(\)[\s\S]*ownerDirectoryOffsetWords\+4u\*cell[\s\S]*taggedVelocity\(tag\)/,
    "a new liquid face must resolve its old value from the accepted extrapolated owner vector");
  const transfer = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn transferStructuredTopologyCandidate("),
    structuredVelocityDynamicsWGSL.indexOf("// Characteristic sources"));
  assert.match(transfer,
    /if\(!vectorValid\(old\)\)\{old=extendedOwnerVelocity\(point\);\}[\s\S]*if\(!vectorValid\(old\)\).*rejectCandidateTransfer/,
    "the extended field is used only after exact old-row transfer fails, and missing support still rejects");
  assert.doesNotMatch(transfer, /old=vec4f\(0/,
    "newly wet faces must never be admitted with an invented zero velocity");
});

test("momentum advection consumes the projected extended field on air rows", () => {
  assert.match(structuredVelocityDynamicsWGSL,
    /fn velocitySample\([\s\S]*sample=rowVelocity\[rbase\(\)\+row\]/,
    "momentum characteristics consume the canonical projected-and-extended row field");
  const advection = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn advect("),
    structuredVelocityDynamicsWGSL.indexOf("fn forceFamily("));
  assert.doesNotMatch(advection, /rowCpt/,
    "the hot advection call graph must not add a separate CPT storage binding");
  // The liquid classification IS part of the advection graph now: the carry
  // gate re-traces the interface band and carries deep-interior liquid, the
  // way main's geometryCode-keyed DELTA_CARRIED identity behaves. The stage
  // stays within WebGPU's ten-storage-buffer limit because advection no
  // longer reads the prescribed solid-normal field (aperture-0 faces keep
  // their staged prior; forceFamily re-imposes the solid value first).
  assert.match(dynamicsHost,
    /this\.advection, FAMILY_CLASSES, \[0, 1, 2, 3, 4, 5, 6, 11, 16, 17, 18\]/,
    "advection binds exactly ten storage buffers including the liquid mask");
});

test("regular advection sampling is per-axis face-based with the cube basis as fallback", () => {
  // Aanjaneya et al. 2017, Section 5: regular regions away from level
  // transitions use standard staggered per-axis face interpolation. The
  // cell-vector cube basis alone applied a [1,2,1]/4 filter to every face's
  // own value each substep, which measured as ~17% mechanical-energy loss by
  // t=0.24 s on the mini dam break.
  assert.match(structuredVelocityDynamicsWGSL, /fn staggeredSample\(/);
  assert.match(structuredVelocityDynamicsWGSL, /fn cellSample\(/);
  const wrapper = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn regularSample("),
    structuredVelocityDynamicsWGSL.indexOf("fn selectorVelocity"));
  assert.match(wrapper, /staggeredSample\(anchor,x\)/,
    "regular samples must try the staggered face basis first");
  assert.match(wrapper, /return cellSample\(anchor,x\);/,
    "the paper's cube/tetra interpolant remains the transition/unextended fallback");
  const advection = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn advect("),
    structuredVelocityDynamicsWGSL.indexOf("fn forceFamily("));
  // The face-centre sample stays on the owner's closure (the staggered basis
  // reproduces the face's own value there); the midpoint and departure
  // re-resolve the containing row first, as main's sampleOld resolved
  // owner(x) per sample point, with the pinned owner row as the fallback.
  assert.ok(advection.includes("adv=regularSample(row,x)"),
    "the face-centre sample must route regular rows through the staggered-first sampler");
  for (const site of ["middle=characteristicSample(row,midpoint)",
    "transported=characteristicSample(row,departure)"]) {
    assert.ok(advection.includes(site),
      `trace samples must resolve the containing element per point (${site})`);
  }
  const characteristic = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn characteristicSample("),
    structuredVelocityDynamicsWGSL.indexOf("fn rowTouchesDry("));
  assert.match(characteristic, /acceptedRowContaining\(point\)/,
    "per-point resolution must consult the accepted row directory");
  assert.match(characteristic, /regularSample\(row,point\)/,
    "the pinned incident row remains the fallback when the directory misses");
  assert.match(advection,
    /let centerTag=regularTag\(row,vec3i\(0\)\);[\s\S]*let useTransition=centerTag!=row/,
    "the producer's published closure marker alone selects the basis: a wall-touching row has a nonzero caseId but keeps axis-normal cube faces and must stay staggered");
  // (row, axis, side) resolves through the publisher's O(1) family-slot map,
  // never a per-slot scan: classifyStructuredCatalogSlots writes
  // rowFamilyHandles[6*row+family] and publishSection63Rows fills
  // rowFamilySlots[base+orientation] for both incident sides of every face.
  assert.match(structuredVelocityDynamicsWGSL, /p\.rowFamilyHandleOffset\+6u\*row\+family/);
  assert.match(structuredVelocityDynamicsWGSL, /p\.rowFamilySlotOffset\+slotBase\+orientation/);
  const handleResolver = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn regularFaceHandle("),
    structuredVelocityDynamicsWGSL.indexOf("fn faceAxisValue("));
  assert.doesNotMatch(handleResolver, /for\s*\(/,
    "the family-slot handle lookup must stay O(1)");
  // The stored degree of freedom is u dot n; the world-axis component must be
  // recovered with the face's own normal sign, exactly.
  assert.match(structuredVelocityDynamicsWGSL, /select\(-sample,sample,n\[axis\]>0\.\)/);
  const planeResolver = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn staggeredPlaneValue("),
    structuredVelocityDynamicsWGSL.indexOf("fn staggeredComponent("));
  // A neighbour qualifies by same size, not caseId==0: wall rows carry a
  // nonzero caseId yet keep exact axis-normal cube faces, and faceAxisValue
  // rejects any genuinely non-axis-normal face below.
  assert.match(planeResolver, /rowGeometry\[rbase\(\)\+tag\]\.y!=rg\.y\)\{return vec2f\(0\.,0\.\);\}/,
    "a size mismatch anywhere in the stencil disqualifies the staggered basis");
  assert.match(planeResolver, /taggedVelocity\(tag\)/,
    "an air support cell contributes its published Section 5 extended vector");
  assert.match(planeResolver, /if\(!vectorValid\(extended\)\)\{return vec2f\(0\.,0\.\);\}/,
    "air without a valid published extension disqualifies the sample instead of substituting zero");
  assert.doesNotMatch(planeResolver, /vec2f\(0\.,1\.\)|return vec2f\(support\/|\(face\+support\)/,
    "no averaged or zeroed face value may be presented as valid");
});

/** CPU transcription of `staggeredComponent`'s weight/topology enumeration
 * (clamping, snapping, corner weights); the WGSL lines it mirrors are pinned
 * by regex below so the two cannot drift apart silently. */
function staggeredCorners(origin: readonly number[], h: number, extent: readonly number[],
  x: readonly number[], axis: number): readonly { plane: number; offset: number[]; weight: number }[] {
  const sample = [...x];
  sample[axis] = Math.min(Math.max(sample[axis]!, 0), extent[axis]!);
  for (let other = 0; other < 3; other += 1) {
    if (other === axis) continue;
    sample[other] = Math.min(Math.max(sample[other]!, 0.5 * h), extent[other]! - 0.5 * h);
  }
  const along = (sample[axis]! - origin[axis]!) / h;
  const plane = Math.min(Math.max(Math.floor(along), -1), 1);
  let tAlong = Math.min(Math.max(along - plane, 0), 1);
  if (tAlong < 1e-5) tAlong = 0; else if (tAlong > 1 - 1e-5) tAlong = 1;
  const center = origin.map((value) => value + 0.5 * h);
  const low = [0, 0, 0]; const tTransverse = [0, 0, 0];
  for (let other = 0; other < 3; other += 1) {
    if (other === axis) continue;
    if (sample[other]! < center[other]!) low[other] = -1;
    let t = Math.min(Math.max((sample[other]! - (center[other]! + low[other]! * h)) / h, 0), 1);
    if (t < 1e-5) t = 0; else if (t > 1 - 1e-5) t = 1;
    tTransverse[other] = t;
  }
  const corners: { plane: number; offset: number[]; weight: number }[] = [];
  for (let corner = 0; corner < 8; corner += 1) {
    let weight = (corner & 1) !== 0 ? tAlong : 1 - tAlong;
    const offset = [0, 0, 0]; let bit = 1;
    for (let other = 0; other < 3; other += 1) {
      if (other === axis) continue;
      const high = (corner & (1 << bit)) !== 0;
      weight *= high ? tTransverse[other]! : 1 - tTransverse[other]!;
      offset[other] = low[other]! + (high ? 1 : 0);
      bit += 1;
    }
    if (weight <= 0) continue;
    corners.push({ plane: plane + (corner & 1), offset, weight });
  }
  return corners;
}

test("staggered weights reproduce a face's own value exactly at its centre", () => {
  // Pin the WGSL arithmetic the CPU transcription mirrors.
  assert.match(structuredVelocityDynamicsWGSL, /let along=\(sample\[axis\]-origin\[axis\]\)\/h;/);
  assert.match(structuredVelocityDynamicsWGSL, /let plane=clamp\(i32\(floor\(along\)\),-1,1\);/);
  assert.match(structuredVelocityDynamicsWGSL,
    /if\(tAlong<1e-5\)\{tAlong=0\.;\}else if\(tAlong>1\.-1e-5\)\{tAlong=1\.;\}/,
    "the measure-zero snap keeps weight one on a face's own value under floating-point dust");
  assert.match(structuredVelocityDynamicsWGSL, /var weight=select\(1\.-tAlong,tAlong,\(corner&1u\)!=0u\);/);
  assert.match(structuredVelocityDynamicsWGSL,
    /sample\[axis\]=clamp\(sample\[axis\],0\.,f32\(d\[axis\]\)\*p\.physical\.x\);/,
    "the axis-normal face lattice reaches the domain walls exactly");
  assert.match(structuredVelocityDynamicsWGSL,
    /sample\[other\]=clamp\(sample\[other\],\.5\*h,f32\(d\[other\]\)\*p\.physical\.x-\.5\*h\);/,
    "transverse axes keep the cell-centred .5h constant physical-boundary extension");

  const origin = [3, 2, 5], h = 1, extent = [16, 16, 16];
  // Sampling AT the +x face centre: weight 1 on that face, nothing else.
  // This self-consistency is exactly what removes the per-substep filter.
  const own = staggeredCorners(origin, h, extent, [4, 2.5, 5.5], 0);
  assert.deepEqual(own, [{ plane: 1, offset: [0, 0, 0], weight: 1 }]);
  // Floating-point dust at the centre must snap back to the exact face.
  const dusty = staggeredCorners(origin, h, extent, [4 + 1e-7, 2.5 - 1e-7, 5.5], 0);
  assert.deepEqual(dusty, [{ plane: 1, offset: [0, 0, 0], weight: 1 }]);
  // The transverse component at that face centre is the standard MAC average
  // of the four transverse faces of the two incident cells.
  const transverse = staggeredCorners(origin, h, extent, [4, 2.5, 5.5], 1);
  assert.equal(transverse.length, 4);
  for (const corner of transverse) assert.ok(Math.abs(corner.weight - 0.25) < 1e-12);
  assert.deepEqual(transverse.map((corner) => [corner.plane, corner.offset[0]]).sort(),
    [[0, 0], [0, 1], [1, 0], [1, 1]]);
  // A quarter-cell characteristic displacement splits 3:1 between the two
  // bracketing same-axis face planes.
  const quarter = staggeredCorners(origin, h, extent, [4.25, 2.5, 5.5], 0);
  assert.deepEqual(quarter.map((corner) => [corner.plane, corner.weight]),
    [[1, 0.75], [2, 0.25]]);
  // Weights always partition unity, including at the clamped domain margin.
  for (const point of [[3.3, 2.9, 5.1], [4, 2.5, 5.5], [3.01, 0.01, 15.99]] as const) {
    for (let axis = 0; axis < 3; axis += 1) {
      const total = staggeredCorners(origin, h, extent, point, axis)
        .reduce((sum, corner) => sum + corner.weight, 0);
      assert.ok(Math.abs(total - 1) < 1e-12, `axis ${axis} at ${point.join(",")}`);
    }
  }
});

test("advection destinations stage into the inactive bank and commit after a fence", () => {
  // The staggered sampler reads neighbouring face degrees of freedom, so a
  // lane writing its destination into the accepted bank mid-dispatch would
  // race the reads and advect some faces through a partially updated field.
  const advection = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn advect("),
    structuredVelocityDynamicsWGSL.indexOf("fn commitAdvected("));
  assert.doesNotMatch(advection, /setValue\(/,
    "advect must never write the accepted value bank it samples from");
  // Aperture-0 faces keep the staged prior (forceFamily re-imposes the exact
  // solid value before any divergence consumer), which keeps advection inside
  // the ten-storage-buffer stage limit.
  for (const staged of ["setNextValue(handle,prior);",
    "setNextValue(handle,projected);"]) {
    assert.ok(advection.includes(staged), `every advect outcome must stage (${staged})`);
  }
  assert.match(structuredVelocityDynamicsWGSL,
    /fn nextValueAt\(handle:u32\)->u32\{return \(1u-bank\(\)\)\*p\.authorityWords\+p\.valuesOffset\+handle;\}/,
    "staging must use the inactive authority bank, which every future candidate publication rewrites");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn commitAdvected\([\s\S]*?if\(!supportPublicationValid\(\)\)\{return;\}[\s\S]*?a\[abase\(\)\+p\.valuesOffset\+handle\]=a\[nextValueAt\(handle\)\];/,
    "commit mirrors the advect gate and copies the staged words back bit-exactly");
  const encodeAdvection = dynamicsHost.slice(dynamicsHost.indexOf("encodeAdvection("),
    dynamicsHost.indexOf("encodeForcesAndDivergence("));
  const advectAt = encodeAdvection.indexOf("Advect structured family class");
  const fenceAt = encodeAdvection.indexOf("broker.fence(");
  const commitAt = encodeAdvection.indexOf("Commit advected structured family class");
  assert.ok(advectAt >= 0 && fenceAt > advectAt && commitAt > fenceAt,
    "the commit dispatches must sit behind a fence that closes the race window");
  assert.match(encodeAdvection, /this\.advectionCommit, FAMILY_CLASSES, \[0, 1, 2, 11, 17, 18\]/,
    "commit binds exactly the workset, authority, and support-control interface");
});

test("gravity is a body force on the liquid, never on dry extension-carrier faces", () => {
  // Dry faces only carry Section 5 extended/advected air values. Integrating
  // gravity into them built a field growing by g*dt every substep in air --
  // never projected, never reset by the extension march -- which the
  // staggered stencil would ingest at the free surface.
  assert.match(structuredVelocityDynamicsWGSL,
    /let wet=liquidAt\(lo\)!=0u\|\|\(hi!=INVALID&&liquidAt\(hi\)!=0u\);\s*if\(!wet\)\{return;\}/,
    "only faces with a liquid incident row may integrate gravity");
  const force = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn forceFamily("),
    structuredVelocityDynamicsWGSL.indexOf("fn divergenceRow("));
  assert.ok(force.indexOf("if(aperture==0.){setValue(handle,solid);return;}")
    < force.indexOf("let wet="),
    "prescribed solid faces keep their exact value before any wetness gate");
  assert.match(dynamicsHost,
    /this\.force, FAMILY_CLASSES, \[0, 1, 2, 11, 16, 17, 22\], params/,
    "the force stage must bind the accepted liquid classification it gates on");
});

test("moving-solid flux is mandatory and liquid classification follows the accepted bank", () => {
  assert.match(structuredVelocityDynamicsWGSL,
    /fn lbase\(\)->u32\{return bank\(\)\*p\.rowCapacity;\}[\s\S]*fn liquidAt\(row:u32\)->u32\{return liquid\[lbase\(\)\+row\];\}/);
  assert.doesNotMatch(structuredVelocityDynamicsWGSL, /liquid\[(?!lbase\(\)\+row)/);
  assert.match(structuredVelocityDynamicsWGSL,
    /binding\(22\)var<storage,read>solidNormalVelocities:array<f32>/);
  assert.match(dynamicsHost,
    /resources\.solidNormalVelocities\.size < structured\.plan\.slotCapacity \* 2 \* 4/,
    "the reader must require both banks, because bank 1 is the one it reads on odd generations");
  assert.match(dynamicsHost, /resources\.liquidMask\.size < structured\.plan\.rowCapacity \* 2 \* 4/);
  assert.match(structuredVelocityDynamicsWGSL, /if\(aperture==0\.\)\{setValue\(handle,solid\);return;\}/);
  assert.match(structuredVelocityDynamicsWGSL,
    /flux\+=sign\*area\*\(aperture\*sample\+\(1\.-aperture\)\*solid\)/);
  assert.match(structuredVelocityDynamicsWGSL,
    /projected=value\(handle\)-p\.physical\.y\*\(pressureHi-pressureLo\)\*inv\*scale\/p\.physical\.z/);
  assert.doesNotMatch(structuredVelocityDynamicsWGSL,
    /pressureHi-pressureLo\)[^;\n]*\*aperture/,
    "aperture belongs in the flux, not twice in the pressure-gradient projection");
});

test("prescribed solid normal velocities are read from the bank the boundary wrote", () => {
  // `beginStructuredPublication` alternates the active bank on every accepted
  // publication (webgpu-octree-structured-velocity-gpu.ts, activeBank =
  // 1u-acceptedControl[4]), so an unbanked read silently consumes the previous
  // generation's slot numbering. A static container hides it because bank 0 is
  // all zeros and zero is the correct prescribed value; a moving solid does not.
  assert.deepEqual(indexExpressions(structuredBoundaryCoefficientWGSL, "solidVelocity"),
    ["sbase()+h"], "the boundary producer must publish into the active structured bank");
  assert.deepEqual(indexExpressions(structuredVelocityDynamicsWGSL, "solidNormalVelocities"),
    ["sbase()+handle"],
    "every prescribed-solid read must go through the single banked accessor, not a raw handle");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn solidVelocityAt\(handle:u32\)->f32\{return solidNormalVelocities\[sbase\(\)\+handle\];\}/,
    "advection, forcing, divergence, and projection must share one banked accessor");
  // Forcing, divergence, and projection read the prescribed solid value;
  // advection no longer does (aperture-0 faces keep their staged prior and
  // forceFamily constrains them before the divergence integrates anything).
  assert.equal((structuredVelocityDynamicsWGSL.match(/solidVelocityAt\(handle\)/g) ?? []).length, 3);
  for (const bank of [0, 1]) {
    assert.equal(slotBankBase(structuredVelocityDynamicsWGSL, bank, 4096),
      slotBankBase(structuredBoundaryCoefficientWGSL, bank, 4096),
      "reader and writer must derive the solid-velocity slot base from the same expression");
  }
  assert.equal(slotBankBase(structuredVelocityDynamicsWGSL, 1, 4096), 4096,
    "bank 1 must be one whole slot capacity away, never aliased onto bank 0");
  assert.match(structuredVelocityDynamicsWGSL, /fn sbase\(\)->u32\{return bank\(\)\*p\.slotCapacity;\}/,
    "the reader's bank must be the accepted structured bank that boundaryValid() pins");
});

test("invalid divergence inputs publish a finite neutral RHS and invalidate the authority", () => {
  const divergence = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn divergenceRow"),
    structuredVelocityDynamicsWGSL.indexOf("fn projectFamily"),
  );
  assert.doesNotMatch(divergence, /0x7fc00000|bitcast<f32>\([^)]*nan/i,
    "WGSL must not construct a NaN literal rejected by Dawn/Naga parsing");
  assert.match(divergence, /rhs\[row\]=0\.;rejectSample\(2[0-3]u,row\);return;/,
    "a rejected row must remain finite while the atomic authority flag fails closed");
  assert.match(divergence,
    /!finite\(p\.physical\.y\)\|\|p\.physical\.y<0\.[\s\S]*!finite\(p\.physical\.z\)\|\|p\.physical\.z<=0\.[\s\S]*if\(p\.physical\.y==0\.\)\{rhs\[row\]=0\.;return;\}[\s\S]*rhs\[row\]=p\.physical\.z\*flux\/p\.physical\.y/,
    "Eq. (3)/(4) uses the integrated flux scaled by rho/dt, matching area/distance A and dt/rho projection");
  assert.doesNotMatch(divergence, /flux\/\(volume\*p\.physical\.y\)/,
    "cell-volume normalization would break the shared variational equation across adaptive leaf sizes");
  assert.match(dynamicsHost,
    /this\.divergence, ROW_CLASSES, \[0, 1, 2, 5, 6, 11, 14, 16, 17, 22\], params/,
    "the divergence bind group must exactly match its reflected Eq. (3)/(4) interface");
  assert.doesNotMatch(dynamicsHost,
    /this\.divergence, ROW_CLASSES, \[[^\]]*\b3\b[^\]]*\], params/,
    "integrated divergence no longer reads rowGeometry, so its auto-layout cannot accept binding 3");

  // With A p = -(rho/dt) flux, the exact projection update leaves zero
  // integrated Eq. (4) flux. This checks the production sign and dimensions.
  const flux = 0.125, dt = 0.004, density = 1000;
  const operatorImage = -density * flux / dt;
  const projectedFlux = flux + dt * operatorImage / density;
  assert.equal(projectedFlux, 0);
});

test("the divergence RHS and the projection are given the same density", () => {
  // The cancellation above only holds when BOTH halves receive the same rho.
  // A literal density in either `update` call silently rescales every pressure
  // gradient: with rhs built at rho=1 and the projection dividing by the scene
  // density, the projection removed 1/rho of the divergence, so the dam column
  // free-fell but its bottom front never advanced. Pin the plumbing, not just
  // the arithmetic.
  assert.match(dynamicsHost,
    /encodeForcesAndDivergence\(broker: PassBroker, dt: number, density: number,/,
    "encodeForcesAndDivergence must take the density from its caller");
  for (const stage of [1, 2]) {
    assert.match(dynamicsHost, new RegExp(`this\\.update\\(${stage}, dt, density,`),
      `stage ${stage} must forward the caller's density, never a literal`);
  }

  const host = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  for (const call of ["encodeForcesAndDivergence", "encodeProjection"]) {
    const at = host.indexOf(`dynamics.${call}(`);
    assert.ok(at > 0, `${call} must be called from the production encode path`);
    assert.match(host.slice(at, at + 240), /this\.scene\.fluid\.density_kg_m3/,
      `${call} must be passed the scene density`);
  }
});

test("structured sampling rejects incomplete selectors, neighbors, and tetrahedra", () => {
  assert.match(structuredVelocityDynamicsWGSL, /fn axisNeighbor\([\s\S]*if\(other>=acc\(2u\)\)\{return INVALID;\}/);
  assert.match(structuredVelocityDynamicsWGSL, /fn selectorVelocity\([\s\S]*return invalidVector\(\);/);
  assert.match(structuredVelocityDynamicsWGSL, /fn transitionSample\([\s\S]*return invalidVector\(\);/);
  assert.match(structuredVelocityDynamicsWGSL,
    /fn rejectSample\(stage:u32,index:u32\)[\s\S]*atomicOr\(&accepted\[0\],ERROR_SAMPLE\);[\s\S]*atomicMin\(&accepted\[1\],\(stage<<24u\)\|\(index&0x00ffffffu\)\)/);
  assert.doesNotMatch(structuredVelocityDynamicsWGSL, /return velocity\(|select\(row,other|select\(vec3f\(0\),velocity/);
  assert.doesNotMatch(structuredVelocityDynamicsWGSL,
    /let adv=select\(|let transported=select\(/,
    "WGSL select evaluates both sampling paths and must not execute a substitute interpolant");
});

test("structured boundary advection is explicit and each class runs once", () => {
  assert.match(structuredVelocityDynamicsWGSL,
    /if\(aperture==0\.\)\{[\s\S]*setValue\(handle,solid\);[\s\S]*return;/,
    "fully prescribed faces must not enter a characteristic sampler");
  assert.match(structuredVelocityDynamicsWGSL,
    /if\(all\(selectorCenter>=lower-vec3f\(tolerance\)\)[\s\S]*return vec4f\(f32\(selectorIndex\),3\.,f32\(other\),-1\.\);[\s\S]*return velocitySample\(row\);/,
    "only catalog selectors proven exterior may use the boundary extension");
  assert.match(dynamicsHost,
    /this\.advection[\s\S]*\[0, 1, 2, 3, 4, 5, 6, 11, 16, 17, 18\]/,
    "advection binds the liquid mask for the carry gate; the solid-normal field belongs to forcing/divergence/projection");
  const encodeClasses = dynamicsHost.slice(dynamicsHost.indexOf("private encodeClasses"),
    dynamicsHost.indexOf("encodeAdvection"));
  assert.equal((encodeClasses.match(/dispatchWorkgroupsIndirect/g) ?? []).length, 1,
    "duplicating destination-owned dispatches would apply gravity and pressure twice");
});

test("projection adds no row-average extension dispatches or pass boundary", () => {
  const projection = dynamicsHost.slice(dynamicsHost.indexOf("encodeProjection("),
    dynamicsHost.indexOf("destroy(): void"));
  assert.doesNotMatch(projection, /extend|layer/,
    "the dedicated face producer owns extrapolation and its publication boundaries");
  assert.doesNotMatch(projection, /broker\.fence\("(?!algorithm diagnostic)/,
    "production projection must not add a pass boundary; opt-in timestamp partitions are diagnostic-only");
});

test("Dawn Metal compiles all structured dynamics variants", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WGSL validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
  } });
  const shaderModule = device.createShaderModule({ code: structuredVelocityDynamicsWGSL });
  const errors = (await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error");
  assert.deepEqual(errors, []);
  device.pushErrorScope("validation");
  entries.forEach((entryPoint) => device.createComputePipeline({
    layout: "auto", compute: { module: shaderModule, entryPoint },
  }));
  const error = await device.popErrorScope(); assert.equal(error, null, error?.message);
  device.destroy();
});
