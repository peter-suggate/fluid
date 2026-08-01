import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { unpackOctreePowerRowTemplateSlot } from "../lib/octree-power-catalog";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { structuredBoundaryCoefficientWGSL } from "../lib/webgpu-octree-structured-boundary";
import { PassBroker } from "../lib/webgpu-pass-broker";
import {
  decodeStructuredProjectionEnergy,
  STRUCTURED_BOUNDARY_DRY_PROBE_DISPATCH_OFFSET_BYTES,
  STRUCTURED_PROJECTION_ENERGY_WORDS,
  WebGPUStructuredVelocityDynamics,
  structuredBoundaryAdvectionFlatteningEnabled,
  structuredDynamicsPlainStoragePassCompactionEnabled,
  structuredRowTouchesDryProbeOracle,
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

const entries = ["prepareStructuredDynamics", "classifyStructuredBoundaryDryProbes",
  "summarizeStructuredPreProjectionEnergy", "summarizeStructuredPostProjectionEnergy",
  ...[5, 6, 7, 8].flatMap((value) => [`advectStructuredClass${value}`,
    `commitAdvectedStructuredClass${value}`, `forceStructuredClass${value}`,
    `projectStructuredClass${value}`]),
  "advectStructuredFlattenedClass7", "advectStructuredFlattenedClass8",
  ...[0, 1, 2, 3].flatMap((value) => [`divergenceStructuredClass${value}`,
    `reconstructStructuredClass${value}`,
    `exchangeStructuredBodyImpulseClass${value}`]),
];

test("overhead contact is unilateral: tension marks separation, the next rebuild opens the face", () => {
  // The paper's Batty-style cut-cell coupling is bilateral and balances a
  // liquid sheet on the tank ceiling with sustained suction — the sheet
  // hangs (measured: 76 vs 13 wet top-layer cells at t=1.5 s on the minimal
  // dam without/with this stage). Contact pressure obeys p >= 0: the
  // projection stage marks rows holding tension against gravity-opposed
  // closed world faces, and the NEXT boundary rebuild opens exactly those
  // faces so the solve itself resolves separation with a p = 0 ghost.
  assert.match(structuredVelocityDynamicsWGSL,
    /fn markSeparationRow[\s\S]*let contact=\.25\*p\.physical\.z\*weight\*p\.physical\.x;[\s\S]*let opening=finite\(solved\)&&solved<contact;[\s\S]*dot\(normal\(handle\),up\)>\.5/,
    "sub-hydrostatic contact pressure must release a gravity-opposed world face immediately");
  assert.doesNotMatch(structuredVelocityDynamicsWGSL,
    /let opening=finite\(solved\)&&solved<0\./,
    "the first stationary top layer must not wait a substep for strict tension");
  assert.match(dynamicsHost, /Mark structured overhead separation/);
  assert.match(structuredBoundaryCoefficientWGSL,
    /if\(boundaryBit!=0u&&\(p\.resolved\.z&boundaryBit\)!=0u\)\{[\s\S]*?aperture=select\(0\.,1\.,separationFresh\(lo,boundaryBit\)\);/,
    "a closed world face opens only under a fresh separation mark");
  assert.match(structuredBoundaryCoefficientWGSL, /fn separationFresh[\s\S]*age<=2u/,
    "stale marks re-close so the active set tracks the solved tension");
  // Complementarity hysteresis: a separated film solves p ~ 0, not p < 0, so
  // a mark that required fresh tension every epoch re-welded the film on an
  // open/closed duty cycle. Renewal may keep faces tension opened but never
  // open new ones.
  assert.match(structuredVelocityDynamicsWGSL,
    /let contact=\.25\*p\.physical\.z\*weight\*p\.physical\.x;[\s\S]*renewing=finite\(solved\)&&previousBits!=0u&&age<=2u&&solved<contact/,
    "an open separation face renews while contact pressure stays below the hydrostatic contact scale");
  assert.match(structuredVelocityDynamicsWGSL,
    /if\(!opening\)\{faceBits&=previousBits;\}/,
    "renewal must not open faces that tension never opened");
});

test("structured dynamics owns destination writes and has no general face/incidence graph", () => {
  assert.match(structuredVelocityDynamicsWGSL,
    /worksetBankStride:u32,dimensionX:u32,dimensionY:u32,dimensionZ:u32,closedMask:u32/,
    "uniform dimensions stay scalar-packed so all authority offsets retain their host word indices");
  assert.doesNotMatch(structuredVelocityDynamicsWGSL, /dimensions:vec3u/,
    "a vec3 uniform member would insert two hidden words before every authority offset");
  assert.doesNotMatch(structuredVelocityDynamicsWGSL, /PowerFaceRecord|incidence/i);
  // Scatter is banned in the field graph, where a destination owns its own
  // write. The one exception is the rigid exchange: many cut faces reduce onto
  // one of at most twelve bodies, which has no destination-owned form.
  assert.deepEqual(
    [...structuredVelocityDynamicsWGSL.matchAll(/atomicAdd\(&([A-Za-z0-9_]+)/g)]
      .map((match) => match[1]).filter((name, index, all) => all.indexOf(name) === index),
    ["rigidExchange"],
    "no field stage may scatter; only the bounded per-body impulse reduction may");
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
    /demandCount=supportWord\(base\+8u\)\+supportWord\(base\+9u\)[\s\S]*count==0u\|\|\(demandCount>0u&&faces>0u&&seeds>0u\)/,
    "regular-only support publications must not require transition-selector demand");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn taggedVelocity\(tag:u32\)[\s\S]*tag==INVALID\|\|!supportPublicationValid\(\)[\s\S]*tag&SUPPORT_TAG[\s\S]*supportVectorOffsetWords/,
    "support tags remain unusable until a complete support-vector publication exists");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn advect\([\s\S]*if\(!supportPublicationValid\(\)\)\{[\s\S]*transportMetrics\[handle\]=invalidVector\(\);[\s\S]*rejectVector\(1u,handle,[\s\S]*return;\}/,
    "every momentum destination must reject an incomplete face-extension transaction, even when its local interpolant needs only direct rows");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn advect\([\s\S]*let prescribed=inflowNormalVelocity\(handle\);[\s\S]*if\(prescribed\.y>0\.\)\{[\s\S]*setNextValue\(handle,prescribed\.x\);[\s\S]*return;[\s\S]*let midpoint=/,
    "prescribed source faces must bypass Section 5 backtracing while ordinary faces retain it");
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

test("velocity reconstruction is permutation invariant and exactly odd under reflection", () => {
  assert.match(structuredVelocityDynamicsWGSL,
    /fn canonicalReconstructionSum\(values:array<f32,31>,count:u32\)/);
  assert.match(structuredVelocityDynamicsWGSL,
    /abs\(sorted\[j-1u\]\)<=abs\(value\)[\s\S]*balance\+=1[\s\S]*balance-=1[\s\S]*f32\(balance\)\*magnitude/,
    "equal-magnitude terms must collapse before summation so negating every term negates the exact f32 result");
  assert.match(structuredVelocityDynamicsWGSL,
    /termsX\[local\]=term\.x;termsY\[local\]=term\.y;termsZ\[local\]=term\.z[\s\S]*canonicalReconstructionSum\(termsX,header\.y\)/,
    "all catalog incidences must enter the canonical component fold");
});

test("divergence flux is independent of reflected incidence order", () => {
  const divergence = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn divergenceRow("),
    structuredVelocityDynamicsWGSL.indexOf("fn divergenceStructuredClass0("));
  assert.match(divergence, /var fluxTerms:array<f32,31>/);
  assert.match(divergence,
    /fluxTerms\[local\]=sign\*area\*boundaryVelocity[\s\S]*canonicalReconstructionSum\(fluxTerms,count\)/,
    "the paper's integrated face flux must not depend on catalog slot order");
  assert.doesNotMatch(divergence, /flux\+=/);
});

test("advection interpolation folds reflected corners canonically", () => {
  assert.match(structuredVelocityDynamicsWGSL,
    /fn snapInterpolationCoordinate\(value:f32\)->f32\{\s*return round\(value\*65536\.\)\/65536\.;/,
    "world-space construction dust must resolve to one dyadic canonical coordinate");
  const staggered = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn staggeredComponent("),
    structuredVelocityDynamicsWGSL.indexOf("fn staggeredSample("));
  assert.match(staggered,
    /weight=canonicalProduct3\(alongWeight,transverseWeights\.x,transverseWeights\.y\)/,
    "staggered trilinear weights must not depend on which world transverse axis is visited first");
  assert.match(staggered,
    /terms\[termCount\]=weight\*resolved\.x;termCount\+=1u;[\s\S]*canonicalInterpolation8\(terms,termCount\)/);
  const cell = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn cellSample("),
    structuredVelocityDynamicsWGSL.indexOf("fn regularSample("));
  assert.match(cell, /weight=canonicalProduct3\(select\(1\.-t\.x[\s\S]*select\(1\.-t\.z/,
    "cube fallback weights must be invariant to horizontal axis permutation");
  assert.match(cell,
    /termsX\[termCount\]=term\.x;termsY\[termCount\]=term\.y;termsZ\[termCount\]=term\.z[\s\S]*canonicalInterpolation8\(termsX,termCount\)/);
  const transition = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn transitionSample("),
    structuredVelocityDynamicsWGSL.indexOf("fn interpolationElementSample("));
  assert.match(structuredVelocityDynamicsWGSL, /fn transitionFanSample[\s\S]*canonicalInterpolation4\(termsX\)/);
  assert.match(transition,
    /let local=snapInterpolationCoordinates\(\s*powerTransform\(\(x-center\)\/extent/);
  assert.match(structuredVelocityDynamicsWGSL,
    /fn canonicalDeterminant\(a:vec3f,b:vec3f,c:vec3f\)->f32[\s\S]*canonicalProduct3[\s\S]*canonicalInterpolation8\(terms,6u\)/,
    "tetrahedral determinants must be invariant to permutation of world axes");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn canonicalVelocityDot\(a:vec3f,b:vec3f\)->f32[\s\S]*canonicalInterpolation4[\s\S]*projected=canonicalVelocityDot\(transported\.xyz,n\)/,
    "power-face projection must not depend on the order of swapped world axes");
  assert.match(structuredVelocityDynamicsWGSL,
    /positiveSum=canonicalInterpolation4\(array<f32,4>\(positive\.x,positive\.y,positive\.z,positive\.w\)\)/,
    "barycentric normalization must not depend on transformed selector order");
  assert.match(structuredVelocityDynamicsWGSL,
    /canonicalVertical=abs\(powerTransform\(vec3f\(0\.,1\.,0\.\),rowTransform\)\)[\s\S]*symmetryMask=\(catalog\[thAt\+2u\]>>\(8u\+8u\*fixedAxis\)\)&255u[\s\S]*array<u32,24>[\s\S]*d4\[8u\*fixedAxis\+symmetry\][\s\S]*canonicalInterpolation8\(xTerms,sampleCount\)/,
    "a symmetric co-spherical link must average the D4 fan conjugated through the row transform");
  assert.doesNotMatch(structuredVelocityDynamicsWGSL, /tetraFanClosedUnderD4Transform|tetraFanUsesSelector/,
    "immutable fan closure must be generated once, never reproved inside every sample");
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
  assert.match(dynamicsHost,
    /this\.group\(pipeline, \[0, 1, 2, 3, 5, 11, 16, 17, 18, 23\], params\)/,
    "the energy bind group must match its auto layout and omit reconstruction-only row velocity");
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
  const smoke = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
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
    /dispatchWorkgroupsIndirect\(this\.resources\.structured\.liveRowDispatch,[\s\S]*OCTREE_STRUCTURED_TOPOLOGY_TRANSFER_DISPATCH_OFFSET_BYTES\)/,
    "topology transfer must launch from the producer's compact changed-face record");
  assert.match(dynamicsHost,
    /words\[54\] = resources\.airSupportLayout\.ownerDirectoryOffsetWords;[\s\S]*words\[55\] = resources\.airSupportLayout\.ownerDirectoryCellCapacity;/,
    "topology transfer must receive the same dense owner directory as fine transport");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn extendedOwnerTag\(q:vec3u\)[\s\S]*ownerDirectoryOffsetWords\+4u\*cell[\s\S]*fn extendedOwnerVelocity\(point:vec3f\)[\s\S]*supportPublicationValid\(\)[\s\S]*taggedVelocity\(tag\)/,
    "a new liquid face must resolve its old value from the accepted extrapolated owner vector");
  const transfer = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn transferStructuredTopologyCandidate("),
    structuredVelocityDynamicsWGSL.indexOf("// Characteristic sources"));
  assert.match(structuredVelocityDynamicsWGSL,
    /@compute @workgroup_size\(128\)fn transferStructuredTopologyCandidate[\s\S]*let index=g\.x\+g\.y\*65535u;let handle=candidateTransferItem\(index\)/,
    "each changed face must own a wide workgroup resolved from the compact class-4 list");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn oldAnchor\([\s\S]*lid<5u[\s\S]*lid<125u[\s\S]*reduceTransferProbe/,
    "ordered backtrace probes and the complete 5^3 neighbourhood must reduce cooperatively");
  assert.match(transfer,
    /if\(!vectorValid\(transferOld\)\)\{transferOld=extendedOwnerVelocity\(point\);\}[\s\S]*if\(!vectorValid\(transferOld\)\).*rejectCandidateTransfer/,
    "the extended field is used only after exact old-row transfer fails, and missing support still rejects");
  assert.doesNotMatch(transfer, /old=vec4f\(0/,
    "newly wet faces must never be admitted with an invented zero velocity");
});

test("Section 5 owner-directory sampling canonically folds exact grid seams", () => {
  const shader = structuredVelocityDynamicsWGSL.replace(/\s+/g, "");
  const extension = shader.slice(shader.indexOf("fnextendedOwnerTag"), shader.indexOf("fnaxisNeighbor"));
  assert.match(extension, /seamMask.*abs\(grid\[axis\]-rounded\[axis\]\)<=1e-5/s,
    "float32-near integer coordinates must resolve as one geometric seam");
  assert.match(extension, /tags:array<u32,8>.*tags\[prior\]==tag.*duplicate=true/s,
    "a coarse owner reached through several finest cells must contribute once");
  assert.match(extension,
    /canonicalInterpolation8\(xTerms,count\).*canonicalInterpolation8\(yTerms,count\).*canonicalInterpolation8\(zTerms,count\)/s,
    "incident seam limits must be folded in a permutation-invariant, reflection-odd sum");
  assert.match(extension, /if\(seamMask==0u\).*extendedOwnerTag\(vec3u\(floor\(grid\)\)\)/s,
    "off-seam extension samples must retain the single-directory-owner fast path");
  assert.match(shader,
    /fncandidateRowCenterInsideOld.*doubled=2u\*origin\+vec3u\(rg\.y\).*incident=acceptedRowContainingFinestCell.*incident!=resolved.*returnfalse/s,
    "a candidate centre is old-liquid interior only when every incident finest cell names one old leaf");
  assert.match(shader,
    /ownerInsideOld=candidateRowCenterInsideOld\(candidateBank,candidateOwner\).*neighborInsideOld=uniformNeighbor!=INVALID&&candidateRowCenterInsideOld\(candidateBank,uniformNeighbor\).*if\(!ownerInsideOld&&!neighborInsideOld\)/s,
    "topology transfer must classify the unordered incident pair, not one arbitrary stored owner");
  assert.match(shader,
    /vectorValid\(ownerField\)&&vectorValid\(neighborField\).*canonicalInterpolation4.*ownerField\.x,neighborField\.x/s,
    "the genuinely two-sided old-field ambiguity must use a D4-canonical fold");
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
  // A reflection exchanges the arbitrary low-coordinate owner of a shared
  // face. The face-centre seam therefore resolves both incident dual-element
  // limits, while midpoint and departure samples re-resolve their containing
  // row as main's sampleOld resolved owner(x) per sample point.
  assert.ok(advection.includes("adv=faceCharacteristicSample(handle,x)"),
    "the face-centre sample must not depend on the storage face's owner side");
  const faceCharacteristic = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn faceCharacteristicSample("),
    structuredVelocityDynamicsWGSL.indexOf("fn rowTouchesDryDirection("));
  assert.match(faceCharacteristic,
    /incidentInterpolationElementSample\(low,point\)[\s\S]*incidentInterpolationElementSample\(high,point\)[\s\S]*canonicalInterpolation4/,
    "a shared face must canonically fold both valid incident element limits");
  for (const site of ["middle=characteristicSample(midpointRow,midpoint)",
    "transported=characteristicSample(departureRow,departure)"]) {
    assert.ok(advection.includes(site),
      `trace samples must resolve the containing element per point (${site})`);
  }
  assert.match(advection,
    /midpointRow=characteristicIncidentRow\(handle,adv\.xyz\)[\s\S]*midpoint=characteristicPoint\(midpointRow,x,\.5\*p\.physical\.y,adv\.xyz\)[\s\S]*departureRow=characteristicIncidentRow\(handle,middle\.xyz\)[\s\S]*departure=characteristicPoint\(departureRow,x,p\.physical\.y,middle\.xyz\)/,
    "both midpoint stages must construct characteristics in the reflection-equivariant grid frame");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn characteristicIncidentRow\(handle:u32,velocity:vec3f\)->u32[\s\S]*normalVelocity=canonicalVelocityDot\(velocity,normal\(handle\)\)[\s\S]*select\(high,low,normalVelocity>=0\.\)/,
    "a shared face must choose the physical upstream incident row, never its arbitrary storage owner");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn characteristicPoint\(row:u32,x:vec3f,dt:f32,velocity:vec3f\)->vec3f[\s\S]*local=snapInterpolationCoordinates\(powerTransform\(\(x-center\)\/extent,transform\)\)[\s\S]*displacement=snapInterpolationCoordinates\(dt\*powerTransform\(velocity,transform\)\/extent\)[\s\S]*traced=snapInterpolationCoordinates\(local-displacement\)[\s\S]*inverseTransform\(traced,transform\)/,
    "world-space translation must not introduce a side-dependent final ulp before Section 5 interpolation");
  const characteristic = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn characteristicSample("),
    structuredVelocityDynamicsWGSL.indexOf("fn rowTouchesDry("));
  assert.match(characteristic, /acceptedRowContaining\(point\)/,
    "per-point resolution must consult the accepted row directory");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn seamInterpolationSample\(point:vec3f\)[\s\S]*abs\(grid\[axis\]-round\(grid\[axis\]\)\)<=1e-5[\s\S]*acceptedRowContaining\(probe\)[\s\S]*incidentInterpolationElementSample\(candidate,point\)[\s\S]*canonicalInterpolation8[\s\S]*let seam=seamInterpolationSample\(point\)/,
    "a characteristic on an octree seam must fold every distinct incident local Delaunay limit");
  assert.match(characteristic, /interpolationElementSample\(row,point\)/,
    "the pinned incident row remains the fallback when the directory misses");
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

test("characteristics resolve the paper's dual element rather than equating it with an octree leaf", () => {
  // Aanjaneya et al. 2017, Section 5
  // (`docs/papers/aanjaneya-2017-power-liquids.txt`) assumes that arbitrary
  // velocity samples are interpolated on the dual mesh: trilinear in cubes
  // and barycentric in locally Delaunay tetrahedra. At a T-junction the
  // octree leaf containing a point need not own a tetrahedron containing it,
  // so the exact fallback must traverse published Delaunay adjacency.
  const element = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn interpolationElementSample("),
    structuredVelocityDynamicsWGSL.indexOf("// Resolve the accepted leaf"));
  assert.match(element,
    /regularTag\(anchor,vec3i\(0\)\)!=anchor\)\{return transitionSample\(anchor,point\);\}[\s\S]*return regularSample\(anchor,point\);/,
    "each candidate element must retain the paper's tetrahedron/cube interpolation rule");
  assert.match(element,
    /p\.selectorOffsetWords\+anchor\*p\.selectorStride\+selectorIndex[\s\S]*tag&SUPPORT_TAG[\s\S]*interpolationElementSample\(tag,point\)/,
    "a rejected seed star must search only its published Delaunay neighbours");
  assert.doesNotMatch(element, /velocitySample\(anchor\)|taggedVelocity\(tag\)/,
    "the adjacency retry must not disguise a missing element as constant velocity");

  const characteristic = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn characteristicSample("),
    structuredVelocityDynamicsWGSL.indexOf("fn rowTouchesDry("));
  assert.match(characteristic,
    /interpolationElementSample\(anchor,point\)[\s\S]*adjacentInterpolationElementSample\(anchor,point\)/,
    "the containing leaf is a search seed, not proof that its local star contains the point");
  assert.match(characteristic,
    /interpolationElementSample\(row,point\)[\s\S]*adjacentInterpolationElementSample\(row,point\)/,
    "the incident-row seam fallback must follow the same exact dual-element search");
  assert.match(characteristic,
    /if\(anchor==INVALID\)\{[\s\S]*extendedOwnerVelocity\(point\)[\s\S]*var pinned=interpolationElementSample\(row,point\)/,
    "a point outside the liquid dual mesh must consume the paper's published closest-face extension before an incident-row interpolant can clamp it");
  assert.match(characteristic,
    /adjacentInterpolationElementSample\(row,point\)[\s\S]*extendedOwnerVelocity\(point\)[\s\S]*return pinned;/,
    "a directory seam retains the closest-face extension as its final fail-closed fallback");
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
  const snap = (value: number) => Math.round(value * 65_536) / 65_536;
  const center = origin.map((value) => value + 0.5 * h);
  const local = sample.map((value, component) => snap((value - center[component]!) / h));
  const along = local[axis]! + 0.5;
  const plane = Math.min(Math.max(Math.floor(along), -1), 1);
  let tAlong = Math.min(Math.max(along - plane, 0), 1);
  if (tAlong < 1e-5) tAlong = 0; else if (tAlong > 1 - 1e-5) tAlong = 1;
  const low = [0, 0, 0]; const tTransverse = [0, 0, 0];
  for (let other = 0; other < 3; other += 1) {
    if (other === axis) continue;
    if (local[other]! < 0) low[other] = -1;
    let t = Math.min(Math.max(local[other]! - low[other]!, 0), 1);
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
  assert.match(structuredVelocityDynamicsWGSL,
    /let local=snapInterpolationCoordinates\(\(sample-center\)\/h\);[\s\S]*let along=local\[axis\]\+\.5;/);
  assert.match(structuredVelocityDynamicsWGSL, /let plane=clamp\(i32\(floor\(along\)\),-1,1\);/);
  assert.match(structuredVelocityDynamicsWGSL,
    /if\(tAlong<1e-5\)\{tAlong=0\.;\}else if\(tAlong>1\.-1e-5\)\{tAlong=1\.;\}/,
    "the measure-zero snap keeps weight one on a face's own value under floating-point dust");
  assert.match(structuredVelocityDynamicsWGSL,
    /let alongWeight=select\(1\.-tAlong,tAlong,\(corner&1u\)!=0u\);[\s\S]*weight=canonicalProduct3\(alongWeight,transverseWeights\.x,transverseWeights\.y\)/,
    "the three basis factors must be multiplied independently of world-axis visitation order");
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

test("class-7/8 carry probes flatten the exact order-free Boolean reduction", () => {
  assert.equal(structuredBoundaryAdvectionFlatteningEnabled({}), true,
    "the measured boundary optimization is production-default");
  assert.equal(structuredBoundaryAdvectionFlatteningEnabled({
    FLUID_STRUCTURED_BOUNDARY_ADVECT_FLAT: "0",
  }), false, "zero retains the serial trajectory oracle");
  assert.equal(structuredBoundaryAdvectionFlatteningEnabled({
    FLUID_STRUCTURED_BOUNDARY_ADVECT_FLAT: "1",
  }), true);
  assert.equal(STRUCTURED_BOUNDARY_DRY_PROBE_DISPATCH_OFFSET_BYTES, 48,
    "the flattened schedule owns the otherwise-unused class-4 record");

  // Exhaust the six independent directional outcomes. The scalar reference
  // is deliberately written without Array.some so this checks the exported
  // oracle's exact Boolean operation rather than repeating its implementation.
  for (const liquid of [false, true]) {
    for (let mask = 0; mask < 64; mask += 1) {
      const probes = Array.from({ length: 6 }, (_, direction) =>
        (mask & (1 << direction)) !== 0);
      let serial = !liquid;
      for (let direction = 0; direction < 6 && !serial; direction += 1) {
        if (probes[direction]) serial = true;
      }
      assert.equal(structuredRowTouchesDryProbeOracle(liquid, probes), serial,
        `liquid=${liquid}, mask=${mask}`);
    }
  }
  assert.throws(() => structuredRowTouchesDryProbeOracle(true, [false]), /six directional/);

  const prepare = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn prepareStructuredDynamics("),
    structuredVelocityDynamicsWGSL.indexOf("fn value("));
  assert.match(prepare,
    /base7=wbase\(7u\);let base8=wbase\(8u\)[\s\S]*flatCount=select\(0u,worksets\[base7\+1u\]\+worksets\[base8\+1u\],[\s\S]*indirect\[12u\]=flatX/,
    "class 4 must concatenate the accepted class-7/8 counts into one face-wide dispatch");

  const flattened = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn rowTouchesDryDirection("),
    structuredVelocityDynamicsWGSL.indexOf("fn advect("));
  assert.match(flattened,
    /lid<16u[\s\S]*local=lid&7u[\s\S]*rowTouchesDryDirection\(row,local\)/,
    "owner and neighbour directions must occupy separate lanes");
  assert.match(flattened,
    /for\(var width=4u;width>0u;width>>=1u\)[\s\S]*dryProbePartial\[lid\]\|=dryProbePartial\[lid\+width\]/,
    "the only combine is an order-free Boolean OR tree");
  assert.match(flattened,
    /\(1u-bank\(\)\)\*p\.authorityWords\+p\.ownerOffset\+handle[\s\S]*\(1u-bank\(\)\)\*p\.authorityWords\+p\.neighborOffset\+handle/,
    "cache words must live in inactive-bank channels that the next candidate rewrites");

  const host = dynamicsHost.slice(dynamicsHost.indexOf("encodeAdvection("),
    dynamicsHost.indexOf("encodeForcesAndDivergence("));
  const flatAt = host.indexOf("Flatten structured boundary carry probes");
  const advectAt = host.indexOf("Advect structured family class");
  assert.ok(flatAt >= 0 && advectAt > flatAt,
    "the cache dispatch must precede its class-7/8 consumers");
  assert.doesNotMatch(host.slice(flatAt, advectAt), /broker\.fence/,
    "the extra dispatch remains in the existing advection pass");
  assert.match(dynamicsHost,
    /this\.encodedAdvectionDispatchCount = this\.flattenedBoundaryAdvection \? 10 : 9/,
    "the A/B accounting is exactly one dispatch and zero implicit work");
});

test("advection destinations stage into the inactive bank and commit in dispatch order", () => {
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
  const commitAt = encodeAdvection.indexOf("Commit advected structured family class");
  assert.ok(advectAt >= 0 && commitAt > advectAt);
  assert.match(encodeAdvection,
    /if \(!this\.compactPlainStoragePass\) \{[\s\S]*structured advected destinations staged[\s\S]*\}[\s\S]*Commit advected structured family class/,
    "only the legacy A/B splits the ordinary-storage handoff");
  assert.match(encodeAdvection, /this\.advectionCommit, FAMILY_CLASSES, \[0, 1, 2, 11, 17, 18\]/,
    "commit binds exactly the workset, authority, and support-control interface");
});

test("structured plain-storage handoffs remove exactly two recurring passes", () => {
  assert.equal(structuredDynamicsPlainStoragePassCompactionEnabled({}), true);
  assert.equal(structuredDynamicsPlainStoragePassCompactionEnabled({
    FLUID_STRUCTURED_DYNAMICS_COMPACT_PASS: "0",
  }), false);

  const encoder = (events: string[]) => ({
    beginComputePass(descriptor?: GPUComputePassDescriptor) {
      events.push(`begin:${descriptor?.label}`);
      return {
        setPipeline() {},
        setBindGroup() {},
        dispatchWorkgroups() { events.push("dispatch"); },
        dispatchWorkgroupsIndirect() { events.push("indirect"); },
        end() { events.push("end"); },
      };
    },
    finish() { events.push("finish"); return {}; },
  } as unknown as GPUCommandEncoder);

  const advectionPasses = (compactPass: boolean) => {
    const events: string[] = [];
    const broker = new PassBroker(encoder(events), { isolateLabels: false });
    const dynamics = {
      destroyed: false,
      compactPlainStoragePass: compactPass,
      update() { return {} as GPUBuffer; },
      encodePrepare(passBroker: PassBroker) {
        passBroker.compute({ label: "Prepare" }).dispatchWorkgroups(1);
        passBroker.fence("indirect published");
      },
      censusTick() {},
      encodeProjectionEnergy() {},
      boundaryDryProbe: undefined,
      advection: [],
      advectionCommit: [],
      encodeClasses(passBroker: PassBroker, _pipelines: unknown, _classes: unknown,
        _bindings: unknown, _params: unknown, label: string) {
        passBroker.compute({ label }).dispatchWorkgroupsIndirect({} as GPUBuffer, 0);
      },
    } as unknown as WebGPUStructuredVelocityDynamics;
    WebGPUStructuredVelocityDynamics.prototype.encodeAdvection.call(dynamics, broker, 1 / 60);
    broker.finish();
    return { passes: broker.computePassCount, events };
  };
  const compactAdvection = advectionPasses(true);
  const splitAdvection = advectionPasses(false);
  assert.equal(compactAdvection.passes, 2);
  assert.equal(splitAdvection.passes, 3);
  assert.deepEqual(compactAdvection.events, [
    "begin:Prepare", "dispatch", "end",
    "begin:Advect structured family class", "indirect", "indirect", "end", "finish",
  ]);

  const transferPasses = (compactPass: boolean) => {
    const events: string[] = [];
    const broker = new PassBroker(encoder(events), { isolateLabels: false });
    const dynamics = {
      destroyed: false,
      compactPlainStoragePass: compactPass,
      params: [{}],
      topologyTransfer: {},
      resources: { structured: { liveRowDispatch: {} } },
      group() { return {}; },
    } as unknown as WebGPUStructuredVelocityDynamics;
    WebGPUStructuredVelocityDynamics.prototype.encodeTopologyTransferCandidate.call(
      dynamics, broker,
    );
    broker.compute({ label: "Reconstruct transferred structured rows" })
      .dispatchWorkgroupsIndirect({} as GPUBuffer, 0);
    broker.finish();
    return { passes: broker.computePassCount, events };
  };
  const compactTransfer = transferPasses(true);
  const splitTransfer = transferPasses(false);
  assert.equal(compactTransfer.passes, 1);
  assert.equal(splitTransfer.passes, 2);
  assert.deepEqual(compactTransfer.events, [
    "begin:Transfer accepted velocity to changed topology faces",
    "indirect", "indirect", "end", "finish",
  ]);
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

test("newly wetted topology faces activate extrapolated velocity at the temporal midpoint", () => {
  const transfer = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("const NEWLY_WET_VELOCITY_RAMP"),
    structuredVelocityDynamicsWGSL.indexOf("fn nextValueAt("),
  );
  assert.match(transfer, /const NEWLY_WET_VELOCITY_RAMP:f32=\.25;/,
    "newly liquid degrees of freedom must center both support and velocity activation");
  assert.match(transfer,
    /!ownerInsideOld[\s\S]*extendedOwnerVelocity\(point\)[\s\S]*NEWLY_WET_VELOCITY_RAMP\*extended\.xyz/,
    "only newly wetted owners ramp the accepted air-extension velocity");
});

test("authored inflow is prescribed before divergence and restored after projection", () => {
  assert.match(dynamicsHost,
    /writeBuffer\(this\.params\[stage\], 224, inflowBytes\)/,
    "every structured stage must receive the same per-step nozzle state");
  assert.match(dynamicsHost,
    /for \(let component = 0; component < 3; component \+= 1\)[\s\S]*desired\[component\] \*= inflow\.apertureScale/,
    "flux normalization must preserve the full authored nozzle direction");
  assert.match(structuredVelocityDynamicsWGSL,
    /fn inflowNormalVelocity\(handle:u32\)->vec2f/);
  assert.doesNotMatch(structuredVelocityDynamicsWGSL, /effectiveRadius/,
    "the velocity boundary must retain the authored nozzle cross-section");
  assert.match(structuredVelocityDynamicsWGSL,
    /let direction=p\.inflowVelocity\.xyz\/speed[\s\S]*let axial=dot\(delta,direction\)[\s\S]*if\(axial<-\.55\*h\|\|axial>2\.55\*h\|\|coverage<=0\.\)/,
    "an oriented nozzle must prescribe every u-dot-n degree of freedom in its short source plug");
  assert.doesNotMatch(structuredVelocityDynamicsWGSL, /fn inflowNormalVelocity[\s\S]*?let axis=/,
    "source-face selection must not rotate through a dominant Cartesian axis");
  const force = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn forceFamily("),
    structuredVelocityDynamicsWGSL.indexOf("fn divergenceRow("));
  assert.match(force,
    /let prescribed=inflowNormalVelocity\(handle\);[\s\S]*setValue\(handle,prescribed\.x\);return;/,
    "the divergence RHS must consume prescribed nozzle-normal velocity");
  assert.ok(force.indexOf("let prescribed=inflowNormalVelocity(handle)")
    < force.indexOf("if(aperture==0.)"),
  "the authored nozzle mouth must override the visual nozzle's closed solid aperture");
  const divergence = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn divergenceRow("),
    structuredVelocityDynamicsWGSL.indexOf("fn divergenceStructuredClass0("));
  assert.match(divergence,
    /boundaryVelocity=select\(aperture\*sample\+\(1\.-aperture\)\*solid,prescribed\.x,prescribed\.y>0\.\)/,
  "equation (4) must integrate the prescribed source flux even through the visual solid");
  const projection = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn projectFamily("),
    structuredVelocityDynamicsWGSL.indexOf("fn reconstructRow("));
  assert.match(projection,
    /inflowNormalVelocity\(handle\)[\s\S]*projected=prescribed\.x/,
    "pressure projection must not erase the authored boundary source");
  assert.doesNotMatch(projection, /aperture>0\.&&prescribed\.y>0\./,
    "source restoration must not be masked by the nozzle body's solid aperture");
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
    /boundaryVelocity=select\(aperture\*sample\+\(1\.-aperture\)\*solid,prescribed\.x,prescribed\.y>0\.\)[\s\S]*fluxTerms\[local\]=sign\*area\*boundaryVelocity/);
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

test("the body-impulse exchange is the exact adjoint of the divergence solid term", () => {
  const divergence = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn divergenceRow("),
    structuredVelocityDynamicsWGSL.indexOf("fn divergenceStructuredClass0"));
  assert.match(divergence,
    /boundaryVelocity=select\(aperture\*sample\+\(1\.-aperture\)\*solid,prescribed\.x,prescribed\.y>0\.\)/,
    "non-source faces must retain the aperture-weighted solid flux differentiated by the adjoint");
  const exchange = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn bodyImpulseRow("),
    structuredVelocityDynamicsWGSL.indexOf("fn exchangeStructuredBodyImpulseClass0"));
  // p * area * (1 - aperture) along the row's outward normal, times dt.
  assert.match(exchange, /p\.physical\.y\*solved\*area\*\(1\.-aperture\)\*sign\*n/,
    "the impulse must be the pressure over the blocked share of the same face, along the same outward normal");
  assert.match(exchange, /liquidAt\(row\)==0u\){return;}/,
    "only a liquid row carries a pressure that can push on a solid");
  assert.match(exchange, /if\(aperture>=1\.\){continue;}/,
    "an uncut face transmits nothing and must not cost a body search");
  assert.match(exchange, /sdf<half&&sdf<best/,
    "attribution must use the same nearest-body-within-half-a-face predicate as the boundary producer");
  assert.doesNotMatch(exchange, /setValue|rhs\[|outputVector/,
    "the adjoint reads the solved pressure and writes only the resident rigid exchange");
});

test("zero-body structured dynamics omits body-impulse pipelines", () => {
  const dynamicsHost = readFileSync(new URL("../lib/webgpu-octree-structured-dynamics.ts", import.meta.url), "utf8");
  assert.match(dynamicsHost,
    /this\.resources\.bodyCount === 0 \? \[\][\s\S]*exchangeStructuredBodyImpulseClass/,
    "a scene without rigid bodies must not create the four impulse pipelines");
  assert.match(dynamicsHost, /count > this\.resources\.bodyCount/,
    "runtime coupling cannot exceed the construction-time pipeline roster");
  assert.match(dynamicsHost,
    /if \(couplingBodyCount > 0\) \{[\s\S]*this\.bodyImpulse/,
    "the existing encode gate must remain conditional on active bodies");
});

test("structured dynamics can defer and sequentially initialize its pipelines", () => {
  assert.match(dynamicsHost,
    /constructor\([\s\S]*deferPipelineCompilation = false\)[\s\S]*if \(!deferPipelineCompilation\) this\.createPipelinesSync\(\)/,
    "existing direct construction must retain synchronous pipeline creation by default");
  const initialize = dynamicsHost.slice(dynamicsHost.indexOf("async initializePipelines("),
    dynamicsHost.indexOf("private requirePipelines()"));
  assert.match(initialize,
    /for \(let index = 0; index < entryPoints\.length; index \+= 1\) \{[\s\S]*await this\.device\.createComputePipelineAsync/,
    "deferred compilation must apply backpressure after every individual pipeline");
  assert.match(initialize,
    /onProgress\(entryPoint, index, entryPoints\.length\)[\s\S]*createComputePipelineAsync[\s\S]*onProgress\(entryPoint, index \+ 1, entryPoints\.length\)/,
    "the caller must see which individual program is pending inside the driver");
  assert.doesNotMatch(initialize, /Promise\.all/,
    "deferred compilation must not fan out driver work");
  assert.match(initialize, /this\.assignPipelines\(pipelines\)/,
    "the complete pipeline set must be published only after sequential compilation");
  assert.match(dynamicsHost,
    /this\.resources\.bodyCount === 0 \? \[\][\s\S]*ROW_CLASSES\.map\(\(value\) => `exchangeStructuredBodyImpulseClass/,
    "the deferred manifest must preserve zero-body pruning");
  assert.match(dynamicsHost,
    /\.\.\.\(this\.projectionEnergyProbe \? \[[\s\S]*summarizeStructuredPreProjectionEnergy[\s\S]*\] : \[\]\)/,
    "disabled energy diagnostics must not compile four unreachable programs");
});

test("only integrating bodies put the adjoint on the command graph", () => {
  const projection = dynamicsHost.slice(dynamicsHost.indexOf("encodeProjection("),
    dynamicsHost.indexOf("destroy(): void"));
  assert.match(projection, /if \(couplingBodyCount > 0\) \{[\s\S]*this\.bodyImpulse/,
    "a scene of authored static solids must not encode an exchange nobody reads");
  const octree = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  assert.match(octree, /this\.dynamicCouplingBodyCount = hasDynamicBodies \? bounded : 0;/,
    "static-only rosters must resolve to a zero coupling count");
  assert.match(octree, /\], pressure, this\.dynamicCouplingBodyCount, this\.surfaceInflow\);/,
    "the projection encode must carry the live coupling roster");
});

test("the solid boundary samples rigid poses in the centred world frame", () => {
  assert.match(structuredBoundaryCoefficientWGSL,
    /fn solidWorld\(x:vec3f\)->vec3f\{return x-vec3f\(\.5\*f32\(p\.dimensions\.x\)\*p\.physical\.x,0\.,\.5\*f32\(p\.dimensions\.z\)\*p\.physical\.x\);\}/,
    "lattice-origin centroids must be centred before they index an authored rigid pose");
  const resolve = structuredBoundaryCoefficientWGSL.slice(
    structuredBoundaryCoefficientWGSL.indexOf("fn resolveStructuredSolidSlots"));
  assert.match(resolve, /let world=solidWorld\(x\);/);
  assert.match(resolve, /rigidSdf\(rb,world\)/,
    "sampling the uncentred point looks for every body half a domain away in x and z");
  assert.doesNotMatch(resolve, /rigidSdf\(rb,x\)/);
});
