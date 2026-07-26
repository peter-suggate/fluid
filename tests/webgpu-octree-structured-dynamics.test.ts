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
  ...[5, 6, 7, 8].flatMap((value) => [`advectStructuredClass${value}`, `forceStructuredClass${value}`,
    `projectStructuredClass${value}`]),
  ...[0, 1, 2, 3].flatMap((value) => [`divergenceStructuredClass${value}`,
    `reconstructStructuredClass${value}`]),
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
  assert.equal(STRUCTURED_PROJECTION_ENERGY_WORDS, 8);
  assert.match(structuredVelocityDynamicsWGSL,
    /binding\(23\)var<storage,read_write>projectionEnergyStats:array<u32>/);
  assert.match(structuredVelocityDynamicsWGSL,
    /dualVolume=area\/inverseDistance[\s\S]*\.5\*aperture\*dualVolume\*sample\*sample/,
    "pre and post reductions must use the identical open-face kinetic-energy measure");
  const forceAt = dynamicsHost.indexOf("Force and constrain structured family class");
  const preAt = dynamicsHost.indexOf('this.encodeProjectionEnergy(broker, params, "pre")');
  const divergenceAt = dynamicsHost.indexOf("Fuse structured divergence RHS class");
  const projectionAt = dynamicsHost.indexOf("Project structured family class");
  const postAt = dynamicsHost.indexOf('this.encodeProjectionEnergy(broker, params, "post")');
  const reconstructAt = dynamicsHost.indexOf("Reconstruct projected structured rows");
  assert.ok(forceAt >= 0 && preAt > forceAt && divergenceAt > preAt);
  assert.ok(projectionAt >= 0 && postAt > projectionAt && reconstructAt > postAt);
  assert.match(structuredVelocityDynamicsWGSL,
    /projectionEnergyStats\[7\]=select\(generation,0u,failed\|\|!pairMatches\)/,
    "the pair publishes only after post energy matches the pre generation, bank, and coverage");
});

test("projection energy decoder fails closed on partial pairs", () => {
  const bits = new Uint32Array(new Float32Array([4, 3]).buffer);
  const coherent = new Uint32Array([0, 11, 1, 24, 24, bits[0]!, bits[1]!, 11]);
  assert.deepEqual(decodeStructuredProjectionEnergy(coherent), {
    sample: {
      epoch: 11, activeBank: 1, familySampleCount: 24,
      preProjectionKineticEnergyProxy: 4,
      postProjectionKineticEnergyProxy: 3,
      projectionEnergyRatio: 0.75,
    },
    blocker: null,
  });
  for (const words of [
    coherent.subarray(0, 7),
    new Uint32Array([...coherent.slice(0, 7), 10]),
    new Uint32Array([...coherent.slice(0, 4), 23, ...coherent.slice(5)]),
    new Uint32Array([1, ...coherent.slice(1)]),
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

test("momentum advection consumes the projected extended field on air rows", () => {
  assert.match(structuredVelocityDynamicsWGSL,
    /fn velocitySample\([\s\S]*sample=rowVelocity\[rbase\(\)\+row\]/,
    "momentum characteristics consume the canonical projected-and-extended row field");
  const advection = structuredVelocityDynamicsWGSL.slice(
    structuredVelocityDynamicsWGSL.indexOf("fn advect("),
    structuredVelocityDynamicsWGSL.indexOf("fn forceFamily("));
  assert.doesNotMatch(advection, /rowCpt|liquidAt/,
    "the hot advection call graph must not add separate CPT/liquid storage bindings");
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
  assert.equal((structuredVelocityDynamicsWGSL.match(/solidVelocityAt\(handle\)/g) ?? []).length, 4);
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
    /this\.advection[\s\S]*\[0, 1, 2, 3, 4, 5, 6, 11, 17, 18, 22\]/,
    "advection must bind the prescribed solid-normal field and complete selector adjacency");
  const encodeClasses = dynamicsHost.slice(dynamicsHost.indexOf("private encodeClasses"),
    dynamicsHost.indexOf("encodeAdvection"));
  assert.equal((encodeClasses.match(/dispatchWorkgroupsIndirect/g) ?? []).length, 1,
    "duplicating destination-owned dispatches would apply gravity and pressure twice");
});

test("projection adds no row-average extension dispatches or pass boundary", () => {
  const projection = dynamicsHost.slice(dynamicsHost.indexOf("encodeProjection("),
    dynamicsHost.indexOf("destroy(): void"));
  assert.doesNotMatch(projection, /extend|layer|broker\.fence\(/,
    "the dedicated face producer owns extrapolation and its publication boundaries");
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
