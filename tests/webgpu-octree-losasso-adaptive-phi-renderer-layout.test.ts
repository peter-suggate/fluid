import assert from "node:assert/strict";
import test from "node:test";
import {
  adaptivePhiRendererDirectoryLayout,
  OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS,
  OCTREE_LOSASSO_ADAPTIVE_PHI_RENDERER_ROW_WORDS,
  unpackAdaptivePhiReceipt,
  WebGPUOctreeLosassoAdaptivePhi,
} from "../lib/webgpu-octree-losasso-adaptive-phi";
import {
  octreeLosassoAdaptivePhiEvidenceWGSL,
  octreeLosassoAdaptivePhiBacktraceWGSL,
  octreeLosassoAdaptivePhiCorrectionWGSL,
  octreeLosassoAdaptivePhiHandoffWGSL,
  octreeLosassoAdaptivePhiRedistanceFinishWGSL,
  octreeLosassoAdaptivePhiRedistanceInitializeWGSL,
  octreeLosassoAdaptivePhiRedistanceProjectAWGSL,
  octreeLosassoAdaptivePhiRedistanceProjectBWGSL,
  octreeLosassoAdaptivePhiPredictorSnapshotWGSL,
  octreeLosassoAdaptivePhiReverseBacktraceWGSL,
  octreeLosassoAdaptivePhiScheduleWGSL,
  octreeLosassoAdaptivePhiTransportWGSL,
  octreeLosassoAdaptivePhiWorklistConstraintMarkWGSL,
  octreeLosassoAdaptivePhiWorklistIndependentWGSL,
  octreeLosassoAdaptivePhiWorklistInflowWGSL,
  octreeLosassoAdaptivePhiWorklistPrepareWGSL,
  octreeLosassoAdaptivePhiWorklistProjectWGSL,
  octreeLosassoAdaptivePhiWorklistReachWGSL,
  octreeLosassoAdaptivePhiWGSL,
}
  from "../lib/webgpu-octree-losasso-adaptive-phi.wgsl";
import { WebGPUOctreeProjection } from "../lib/webgpu-octree";
import { makeOctreePowerCoarseLevelSetSampleWGSL } from
  "../lib/webgpu-octree-power-coarse-levelset";

const compact = (source: string) => source.replace(/\s+/g, "");
const f32 = Math.fround;
const permutations = <T>(values: readonly T[]): T[][] => values.length <= 1
  ? [Array.from(values)]
  : values.flatMap((value, index) => permutations(values.filter((_, at) => at !== index))
    .map((tail) => [value, ...tail]));
const orderedConstraintOracle = (masters: readonly (readonly [number, number])[]): number => {
  const terms = masters.map(([numerator, value]) => f32(f32(numerator) * f32(value)));
  if (terms.length === 2) return f32(terms[0]! + terms[1]!);
  terms.sort((a, b) => a - b || new Uint32Array(new Float32Array([a]).buffer)[0]!
    - new Uint32Array(new Float32Array([b]).buffer)[0]!);
  return f32(f32(terms[0]! + terms[1]!) + f32(terms[2]! + terms[3]!));
};

test("adaptive phi renderer auxiliary records follow live primary rows", () => {
  const liveRows = 3;
  const capacity = 11;
  const layout = adaptivePhiRendererDirectoryLayout(liveRows, capacity);

  assert.equal(layout.primaryWordOffset, 8);
  assert.equal(layout.auxiliaryWordOffset,
    layout.primaryWordOffset + OCTREE_LOSASSO_ADAPTIVE_PHI_RENDERER_ROW_WORDS * liveRows);
  assert.equal(layout.auxiliaryWordOffset, 32);
  assert.notEqual(layout.auxiliaryWordOffset,
    layout.primaryWordOffset + OCTREE_LOSASSO_ADAPTIVE_PHI_RENDERER_ROW_WORDS * capacity);
  assert.ok(layout.liveWordCount <= layout.allocatedWordCount);
});

test("factor-one renderer source prefers adaptive phi over the legacy coarse directory", () => {
  const getter = Object.getOwnPropertyDescriptor(
    WebGPUOctreeProjection.prototype, "coarseLevelSetSource")?.get?.toString() ?? "";
  const compactGetter = compact(getter);
  assert.match(compactGetter,
    /constcoarse=power\?\?\(this\.coarseOnlySurfaceTracking\?adaptive\?/,
    "factor one must use adaptive one-way renderer authority while legacy storage remains allocated");
  assert.match(compactGetter, /constgradients=adaptive\?(?:undefined|void0):/,
    "adaptive leaf-indexed rows must not advertise legacy pressure-row gradients");
  assert.match(compactGetter,
    /constgeneration=power\?.*?:this\.coarseOnlySurfaceTracking\?adaptive\?this\.adaptiveSurfaceGeneration:0:legacyLosasso/s,
    "the renderer generation must come from the same adaptive authority as its directory");
  assert.match(compactGetter, /if\(!coarse\|\|generation<1.*?\)return(?:undefined|void0)/s,
    "a missing adaptive generation must fail closed instead of falling through to legacy phi");
});

test("adaptive phi publisher parallelizes live rows and clears the renderer capacity tail", () => {
  const shader = compact(octreeLosassoAdaptivePhiEvidenceWGSL);
  assert.match(shader,
    /fnprepareTopologyEvidence\(@builtin\(global_invocation_id\)gid:vec3u\).*if\(item<arrayLength\(&renderer\)\)\{renderer\[item\]=0u;\}/s,
    "renderer tail clearing must be parallel rather than serialized in one invocation");
  assert.match(shader,
    /fnpublishTopologyEvidenceRows\(@builtin\(global_invocation_id\)gid:vec3u\).*letrow=gid\.x/s,
    "each live graph leaf must publish its renderer row independently");
  assert.match(shader, /fnfinishTopologyEvidence\(\).*for\(varrow=0u;row<rows;row\+=1u\)/s,
    "the collision-sensitive hash insertion must remain canonical and serial");
  assert.match(shader, /letaux=8u\+8u\*rows\+8u\*row/);
  assert.doesNotMatch(shader, /letaux=8u\+8u\*p\.limits\.y\+8u\*row/);
  assert.match(shader, /rendererRowCapacity=\(rendererWords-8u\)\/16u/);
  assert.match(shader,
    /fncanonicalPublishedZero\(v:f32\)->f32\{letbits=bitcast<u32>\(v\);returnbitcast<f32>\(select\(bits,0u,\(bits&0x7fffffffu\)==0u\)\);\}[\s\S]*letcentre=canonicalPublishedZero\(rawCentre\);mn=canonicalPublishedZero\(mn\);mx=canonicalPublishedZero\(mx\)/,
    "derived primary rows must publish one canonical zero bit while authoritative nodal corners retain their exact values");
});

test("adaptive renderer evaluates one continuous nodal field in leaf coordinates", () => {
  const sample = compact(makeOctreePowerCoarseLevelSetSampleWGSL());
  assert.match(sample,
    /letfraction=clamp\(\(grid-vec3f\(origin\)\)\/f32\(entry\.size\),vec3f\(0\.\),vec3f\(1\.\)\)/,
    "every leaf must interpolate in the same finest-lattice coordinate system");
  assert.match(sample,
    /letvalue=powerCoarseAdaptiveCorner\(aux,corner\).*terms\[corner\]=\(wx\*wz\)\*wy\*value/s,
    "adaptive sampling must trilinearly interpolate the eight published nodal corners");
  assert.match(sample,
    /fnpowerCoarseContainingLookup\(grid:vec3f\).*letonPlane=abs\(grid-round\(grid\)\).*for\(varmask=0u;mask<8u;mask\+=1u\)/s,
    "shared faces must resolve through the same boundary-aware finest-grid convention");
  assert.match(sample,
    /letq=vec3u\(floor\(probe\)\);if\(any\(q>=powerCoarseSamples\.dimensions\)\)\{continue;\}/,
    "high-boundary nodal probes must not alias a wrapped linear cell before the inward mask is tried");
});

test("cold authored boxes bootstrap directly from the finest node lattice", () => {
  const shader = compact(octreeLosassoAdaptivePhiWGSL);
  assert.match(shader,
    /fnbootstrapNodalLattice\(.*letq=nodePosition\(i\);letd=p\.dims\.xyz\+vec3u\(1u\);letat=q\.x\+d\.x\*\(q\.y\+d\.y\*q\.z\).*letv=distanceA\[at\]/s,
    "node-lattice bootstrap must gather the exact authored node without interpolation");
  const encode = compact((WebGPUOctreeLosassoAdaptivePhi.prototype as unknown as {
    encodeCandidateSelected: (...args: unknown[]) => void;
  }).encodeCandidateSelected.toString());
  assert.match(encode,
    /nodal-lattice-cpu.*product\*\(value\+1\).*bootstrapNodalLattice/s,
    "the retained cold retry must preserve the node-lattice layout and entry point");
  const schedule = compact(octreeLosassoAdaptivePhiScheduleWGSL);
  assert.match(schedule,
    /letwarm=acceptedGraph\[0\]!=0u.*letwn=select\(0u,groups\(candidateGraph\[2\]\),warm&&candidateValid\).*letcn=select\(groups\(candidateGraph\[2\]\),0u,warm\|\|!candidateValid\)/s,
    "once accepted adaptive phi is live, warm handoff must publish zero cold-bootstrap work");
});

test("warm candidate scheduling preserves the accepted phi bank", () => {
  const shader = compact(octreeLosassoAdaptivePhiScheduleWGSL);
  const warmBranch = shader.match(/else\{atomicStore\(&state\[1\],acceptedGraph\[0\]\)(.*?)\}atomicStore\(&state\[8\]/s);
  assert.ok(warmBranch, "warm schedule branch must remain identifiable");
  assert.doesNotMatch(warmBranch[1]!, /atomicStore\(&state\[6\],0u\)/);
});

test("accepted identity transport retains the exact current phi bank", () => {
  const shader = compact(octreeLosassoAdaptivePhiWGSL);
  assert.match(shader, /fnprepareAdvance\(\).*for\(vari=0u;i<=12u;i\+=1u\).*atomicStore\(&control\[17\],1u\)/s,
    "every advance must clear stale reference-delta receipt state");
  assert.match(shader,
    /fncaptureTransportReceipt\(\).*letretain=maximumDelta==0u&&atomicLoad\(&receipts\[12\]\)==0u&&p\.inflowVelocityStrength\.w==0\..*atomicStore\(&control\[17\],select\(1u,0u,retain\)\)/s);
  assert.match(shader,
    /fnfinalizeAccepted\(\).*if\(valid\)\{if\(redistanceEnabled\(\)\)\{atomicStore\(&control\[6\],1u-currentBank\(\)\);\}atomicAdd\(&control\[2\],1u\);\}/s,
    "a valid retained no-op advances generation without flipping banks");
  const encode = compact(WebGPUOctreeLosassoAdaptivePhi.prototype.encodeAcceptedAdvance.toString());
  assert.match(encode,
    /prepareAdvance.*applyReferenceVolumeDelta.*markTransportReach.*markTransportInflow.*markTransportConstrained.*publishTransportIndependent.*publishTransportConstrained.*this\.backtrace.*this\.transport.*this\.correction.*captureTransportReceipt.*projectTransportedBand.*capturePreRedistanceVolumes/s,
    "the corrected hanging-node projection must run after the exactly-once transport receipt");
});

test("constraint reductions are bit invariant under D4 master permutations", () => {
  const masters = [[1, 0.18437592685222626], [3, -0.07187500596046448],
    [2, 0.00937500037252903], [2, -0.003124999813735485]] as const;
  const results = permutations(masters).map(orderedConstraintOracle);
  const bits = results.map((value) => new Uint32Array(new Float32Array([value]).buffer)[0]!);
  assert.equal(new Set(bits).size, 1);

  const shader = compact(octreeLosassoAdaptivePhiWGSL);
  for (const entry of ["constrainedValue", "projectAccepted", "projectTransported"]) {
    assert.match(shader, new RegExp(`fn${entry}\\(.*?orderedConstraintSum\\(`, "s"),
      `${entry} must not sum graph master order directly`);
  }
  const redistance = compact(octreeLosassoAdaptivePhiRedistanceProjectAWGSL
    + octreeLosassoAdaptivePhiRedistanceProjectBWGSL);
  for (const entry of ["projectDistanceA", "projectDistanceB"]) {
    assert.match(redistance, new RegExp(`fn${entry}\\(.*?orderedFour\\(`, "s"),
      `${entry} must not sum graph master order directly`);
  }
  assert.match(redistance,
    /if\(!k0\|\|!k1\)\{distance\[i\]=min\(abs\(distance\[i\]\),select\(d1,d0,k0\)\)/,
    "two-master constraints must propagate a reached master instead of averaging the far sentinel");
  assert.match(redistance,
    /select\(FAR,d0,k0\).*select\(FAR,d3,k3\).*distance\[i\]=min\(abs\(distance\[i\]\),reached\)/s,
    "four-master constraints must propagate reached masters until canonical interpolation is defined");
  const handoff = compact(octreeLosassoAdaptivePhiHandoffWGSL);
  assert.match(handoff, /fnprojectCandidate\(.*orderedConstraintSum\(/s);
  assert.match(handoff, /fnsample\(.*orderedSum8\(/s);
});

test("accepted transport publishes a compact physical-reach band", () => {
  const shader = compact(octreeLosassoAdaptivePhiWGSL);
  const prepare = compact(octreeLosassoAdaptivePhiWorklistPrepareWGSL);
  const reach = compact(octreeLosassoAdaptivePhiWorklistReachWGSL);
  const inflow = compact(octreeLosassoAdaptivePhiWorklistInflowWGSL);
  const independent = compact(octreeLosassoAdaptivePhiWorklistIndependentWGSL);
  const constraintMark = compact(octreeLosassoAdaptivePhiWorklistConstraintMarkWGSL);
  assert.match(shader,
    /fnprepareAdvance\(\).*velocityReceipt\[0\]!=graph\[5\].*velocityReceipt\[7\]!=1u/s);
  assert.match(prepare,
    /fnprepareTransportBand\(\).*atomicStore\(&arena\[11\],velocityReceipt\[1\]\)/s);
  assert.match(prepare,
    /fnprepareRedistanceBand\(\).*atomicStore\(&arena\[11\],bitcast<u32>\(2\.\*p\.originCell\.w\)\).*atomicStore\(&arena\[12\],1u\)/s);
  assert.match(reach,
    /fnmarkTransportReach\(.*bank=select\(current,1u-current,atomicLoad\(&arena\[12\]\)==1u\).*abs\(phi\[i\]\[bank\]\)<reach/s);
  assert.match(inflow,
    /fnmarkTransportInflow\(.*axial>=-reach.*dot\(radial,radial\)<=radius\*radius.*bandMask\[i\]\|=1u/s);
  const redistance = compact(octreeLosassoAdaptivePhiRedistanceInitializeWGSL
    + octreeLosassoAdaptivePhiRedistanceFinishWGSL);
  assert.match(reach,
    /scheduled=scheduled\|\|abs\(value\)<reach[\s\S]*scheduled=scheduled\|\|frozen/,
    "redistance must compact a strict phi core plus one non-cascading donor halo");
  assert.match(independent, /fnpublishTransportIndependent\(.*worklist\[rank\]=i\|select\(0u,0x80000000u/s);
  assert.match(independent,
    /if\(\(marked&1u\)==0u\)\{if\(atomicLoad\(&transportControl\[12\]\)==0u\)\{letcurrent=bank\(\);phi\[i\]\[1u-current\]=phi\[i\]\[current\]/,
    "far independent nodes must be retained bitwise");
  assert.match(constraintMark,
    /constraints\[master\]\.header\.y!=0u.*atomicOr\(&control\[12\],ERR_CONSTRAINT\)/s,
    "constraint scheduling must reject non-independent masters before reading masks");
  assert.match(reach + constraintMark + redistance,
    /minimumPhi<=0\.&&maximumPhi>=0\.[\s\S]*atomicOr\(&bandMask\[master\],marked\)[\s\S]*bitcast<u32>\(abs\(signed\)\)\|FROZEN/,
    "an exact zero and every other mixed-leaf corner must retain transported interface authority");
  const backtrace = compact(octreeLosassoAdaptivePhiBacktraceWGSL);
  assert.match(backtrace,
    /fnbacktraceIndependent\(.*v0=packedVelocity\(i\).*sampleVelocity\(boundedMid,incidentLeaves\[8u\*i\+midpointOctant\].*incidentLeaves\[8u\*i\+departureOctant\].*departures\[2u\*i\]=vec4f/s,
    "required characteristics must use compiled incident leaves for midpoint and departure support");
  const reverseBacktrace = compact(octreeLosassoAdaptivePhiReverseBacktraceWGSL);
  assert.match(reverseBacktrace,
    /fnreverseBacktraceIndependent\(.*mid=q\+snapDisplacement.*reverse=q\+snapDisplacement.*departures\[2u\*i\+1u\]=vec4f/s);
  const transport = compact(octreeLosassoAdaptivePhiTransportWGSL);
  assert.match(transport, /fntransportIndependent\(.*i=worklist\[work\].*departures\[2u\*i\].*samplePhi/s);
  const predictorSnapshot = compact(octreeLosassoAdaptivePhiPredictorSnapshotWGSL);
  assert.match(predictorSnapshot,
    /fnsnapshotProjectedPredictor\(.*value=phi\[i\]\[1u-\(atomicLoad\(&control\[6\]\)&1u\)\].*predictorScratch\[i\]=value/s);
  const correction = compact(octreeLosassoAdaptivePhiCorrectionWGSL);
  assert.match(correction, /fndonorBounds\(leaf:u32,b:u32\)->vec3f\{if\(leaf>=leafCount\(\)\)\{returnvec3f\(0\);\}/,
    "an invalid backward donor leaf must fail closed before storage access");
  assert.match(correction,
    /fncorrectTransportIndependent\(.*backward=departures\[2u\*i\].*reverse=departures\[2u\*i\+1u\].*reversed=samplePredictor\(reverse\.xyz,bitcast<u32>\(reverse\.w\)\).*clamp\(predictor\+\.5\*\(prior-reversed\.value\),bounds\.x,bounds\.y\).*quantizePhi\(corrected\)/s);
  for (const atomicSource of [reach, backtrace, reverseBacktrace, transport,
    predictorSnapshot, correction]) {
    assert.doesNotMatch(atomicSource, /var<storage,read>[^;]*array<atomic</,
      "WebGPU atomic storage declarations must be read_write even when kernels only load them");
  }
  const project = compact(octreeLosassoAdaptivePhiWorklistProjectWGSL);
  assert.match(project, /fnprojectTransportedBand\(.*i=worklist\[capacity\(\)\+work\]/s);
  const receipt = unpackAdaptivePhiReceipt(new Uint32Array(OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS)
    .map((_, index) => index));
  assert.deepEqual(receipt.transportBand, {
    scheduledIndependentNodes: 32, retainedIndependentNodes: 33,
    scheduledConstrainedNodes: 34, retainedConstrainedNodes: 35,
  });
  assert.deepEqual(receipt.volumeTransaction, {
    epoch: 36, generation: 37, validatedNodes: 38, constrainedNodes: 39,
    coveredLeaves: 40,
    maximumAbsoluteLeafDrift_m3: new Float32Array(new Uint32Array([41]).buffer)[0],
    totalAbsoluteLeafDrift_m3: new Float32Array(new Uint32Array([42]).buffer)[0],
    valid: false,
  });
  assert.deepEqual(receipt.candidateVolumeTransaction, {
    epoch: 44, generation: 45, validatedNodes: 46, constrainedNodes: 47,
    coveredLeaves: 48,
    maximumAbsoluteLeafDrift_m3: new Float32Array(new Uint32Array([49]).buffer)[0],
    totalAbsoluteLeafDrift_m3: new Float32Array(new Uint32Array([50]).buffer)[0],
    valid: false,
  });
  assert.deepEqual(receipt.redistanceBand, {
    activeIndependentNodes: 52,
    activeConstrainedNodes: 53,
  });
});

test("precomputed inflow cylinder preserves the authored source predicate", () => {
  const dimensions = [24, 18, 20] as const; const cell = .05;
  const outlet = [.08, .11, -.04] as const; const velocity = [2, -1, 3] as const;
  const speed = Math.hypot(...velocity); const direction = velocity.map((v) => v / speed);
  const centre = [outlet[0] + .5 * dimensions[0] * cell, outlet[1],
    outlet[2] + .5 * dimensions[2] * cell];
  const radius = .12; const reach = .18;
  const dot = (a: readonly number[], b: readonly number[]) =>
    a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
  for (const x of [-.4, -.15, 0, .2, .45]) for (const y of [-.3, 0, .17, .4]) {
    for (const z of [-.35, -.08, .12, .38]) {
      const world = [centre[0] + x, centre[1] + y, centre[2] + z];
      const relative = world.map((v, axis) => v - centre[axis]!);
      const axial = dot(relative, direction);
      const radial = relative.map((v, axis) => v - axial * direction[axis]!);
      const oldSdf = Math.max(Math.hypot(...radial) - radius,
        Math.max(-axial, axial - 2 * cell));
      const expanded = radius + reach;
      const squaredSupport = axial >= -reach && axial <= 2 * cell + reach
        && dot(radial, radial) <= expanded * expanded;
      assert.equal(squaredSupport, oldSdf <= reach);
    }
  }
  const authoring = WebGPUOctreeLosassoAdaptivePhi.toString();
  assert.match(authoring, /Math\.hypot\(inflow\.velocity_m_s\.x/);
  assert.match(authoring, /centreX.*\.5\s*\*\s*this\.options\.dimensions\[0\]/s);
});
