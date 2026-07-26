import assert from "node:assert/strict";
import test from "node:test";
import {
  FINE_LEVELSET_TRANSPORT_SUMMARY_ITEMS_PER_WORKGROUP,
  FINE_LEVELSET_TRANSPORT_CLASS_BINDINGS,
  planFineLevelSetGPUTransport,
  planFineLevelSetGPUTransportPasses,
  structuredFineLevelSetTransportWGSL,
  WebGPUFineLevelSetTransport,
} from "../lib/webgpu-octree-fine-levelset-transport";

function compact(source: string) { return source.replace(/\s+/g, ""); }

test("direct fine transport has one page-bounded structured authority path", () => {
  const plan = planFineLevelSetGPUTransport(16_777_216, 4_096, 262_144);
  assert.deepEqual(plan, {
    queryCapacity: 16_777_216,
    velocityChunkCapacity: 16_777_216,
    chunkCount: 1,
    topologyDeltaBytes: (8 + 2 * 262_144) * 4,
    pageStatusBytes: 262_144 * 12,
    worksetBytes: 128 + 262_144 * 4,
    controlBytes: 64,
    allocatedBytes: 256 + 262_144 * 12 + 128 + 262_144 * 4
      + (8 + 2 * 262_144) * 4 + 64 + 2 * 512,
  });
  assert.deepEqual(planFineLevelSetGPUTransportPasses(plan, 8), {
    chunkCount: 1, segmentCount: 64, passesPerSegment: 0, passesPerChunk: 1, encodedPasses: 10,
  });
  assert.equal(FINE_LEVELSET_TRANSPORT_SUMMARY_ITEMS_PER_WORKGROUP, 4_096);
});

test("fine transport consumes the shared positive-air tag and vector authority", () => {
  const construction = compact(WebGPUFineLevelSetTransport.toString());
  const shader = compact(structuredFineLevelSetTransportWGSL);
  assert.match(construction,
    /air\.layout\.rowCapacity!==structured\.plan\.rowCapacity.*air\.arena\.size<air\.layout\.totalBytes/s);
  assert.match(shader, /fntaggedVelocity\(tag:u32\)->vec4f/);
  assert.match(shader, /\(tag&SUPPORT_TAG\)==0u.*canonicalVelocity\(tag\)/s);
  assert.match(shader, /supportVectorOffset\/4u\+support/);
  assert.doesNotMatch(shader, /@binding\(19\).*cpt|cpt\[/,
    "the retired row-only extension must not bypass the support publication");
});

test("recurring transport publishes two validity classes before atomic commit", () => {
  const encode = compact(WebGPUFineLevelSetTransport.prototype.encode.toString());
  const construction = compact(WebGPUFineLevelSetTransport.toString());
  for (const label of [
    "PlanGPU-residentfinetransportsubsteps",
    "Classifydirectstructuredfinetransportblocks",
    "Publishdirectstructuredfinetransportworksets",
    "Publishstructuredfinetransportstatus",
    "Commitstructuredfinephiandphasedelta",
    "Compactstructuredfinetopologydelta",
  ]) assert.match(encode, new RegExp(label));
  assert.match(encode,
    /classNames=\["common","rare"\].*label:`Advectfinephi\$\{classNames\[index\]\}`/,
    "common and validity-aware rare trajectories must receive attributable passes");
  assert.match(encode,
    /classifyPipeline.*publishWorksetsPipeline.*fence\("structuredfinetransportworksetspublished"\).*transportPipelines.*summarizePipeline.*commitPipeline.*deltaPipeline/,
    "dependent storage dispatches share a pass; only the storage-to-indirect copy closes it");
  for (const obsoleteFence of [
    "directfineownerrowspublished",
    "structuredfinetransportclassesmarked",
    "structuredfinetrajectoriescomplete",
    "structuredfinetransportaccepted",
    "structuredfinephicommitted",
  ]) assert.doesNotMatch(encode, new RegExp(`fence\\(\"${obsoleteFence}\\"\\)`),
    `${obsoleteFence} must not split a storage-only dependency chain`);
  assert.doesNotMatch(encode, /ownerPipeline|ownerGroup|Publishdirectfinesampleownerrows/,
    "per-trajectory owner selection must not retain a page-anchor publication dispatch");
  assert.match(encode, /classify\.dispatchWorkgroupsIndirect\(this\.indirectDispatch,160\)/);
  assert.match(encode, /transport\.dispatchWorkgroupsIndirect\(this\.indirectDispatch,\(4\+7\*FINE_LEVELSET_TRANSPORT_WORKSET_CLASSES\[index\]\+4\)\*4\)/);
  assert.match(compact(structuredFineLevelSetTransportWGSL),
    /state\[base\+4u\]=counts\[cls\]/,
    "one transport workgroup owns one page; the 64 lanes cover its samples rather than 64 pages");
  assert.doesNotMatch(compact(structuredFineLevelSetTransportWGSL),
    /state\[base\+4u\]=u32\(ceil\(f32\(counts\[cls\]\)\/64\.\)\)/,
    "the class page count must not be divided by the within-page workgroup width");
  assert.match(encode, /commit\.dispatchWorkgroupsIndirect\(this\.indirectDispatch,160\)/);
  assert.doesNotMatch(encode, /dispatchWorkgroupsIndirect\(this\.governor/,
    "the writable governor must never also be consumed as INDIRECT in a compute scope");
  assert.doesNotMatch(encode, /faceBand|powerFace|velocityPrepass|fallback|legacy|createBindGroup/i);
  assert.match(construction,
    /this\.transportGroups=this\.transportPipelines\.map\(\(pipeline,index\)=>group\(pipeline,FINE_LEVELSET_TRANSPORT_CLASS_BINDINGS\[index\]\)\)/);
  assert.match(construction, /this\.classifyGroup=group\(this\.classifyPipeline,\[0,1,2,3,4,13\]\)/,
    "classification must remain independent of the per-trajectory air-owner authority");
  FINE_LEVELSET_TRANSPORT_CLASS_BINDINGS.forEach((bindings, index) => {
    assert.ok(bindings.filter((binding) => binding !== 0).length <= 10,
      `class ${index} exceeds Dawn's portable storage-buffer ceiling`);
    assert.ok(bindings.includes(20), `class ${index} does not bind air support`);
  });
  assert.equal(FINE_LEVELSET_TRANSPORT_CLASS_BINDINGS[1].length - 1, 9);
  assert.equal(FINE_LEVELSET_TRANSPORT_CLASS_BINDINGS.length, 2,
    "per-trajectory owner selection eliminates page-anchor regular/transition dispatch multiplication");
  assert.ok(FINE_LEVELSET_TRANSPORT_CLASS_BINDINGS.every((bindings) => !bindings.includes(11)),
    "dense air-owner publication must eliminate the transport row-geometry binding");
});

test("WGSL fails closed unless generation, epoch, and accepted A\/B bank agree", () => {
  const shader = compact(structuredFineLevelSetTransportWGSL);
  assert.match(shader, /fnactiveBank\(\)->u32\{returnaccepted\[4\]&1u/);
  assert.match(shader,
    /fnstructuredValid\(\)->bool\{returnarrayLength\(&accepted\)>=6u&&accepted\[0\]==0u&&accepted\[2\]>0u&&accepted\[3\]!=0u&&accepted\[4\]<=1u/);
  assert.match(shader,
    /fnairPublicationValid\(\)->bool.*boundary\[4\]!=accepted\[3\].*airWord\(at\+2u\)==accepted\[3\].*airWord\(at\+3u\)==activeBank\(\).*airWord\(at\+5u\)==accepted\[2\].*airWord\(at\+13u\)==SUPPORT_VALID.*airWord\(at\+14u\)==SUPPORT_VERSION.*airWord\(at\+15u\)==p\.generation/s,
    "transport must consume a committed support transaction for its exact fine generation");
  assert.doesNotMatch(shader, /airWord\(at\+12u\)>=p\.maxBacktrace/,
    "face-graph hops must not be compared numerically with fine-cell displacement");
  const encode = compact(WebGPUFineLevelSetTransport.prototype.encode.toString());
  assert.match(encode,
    /maximumBacktraceFineCells>plan\.fineFactor.*Finetransportdisplacementboundexceedsitsconfiguredsupportdepth/s,
    "the host API must reject a backtrace bound larger than its single configured fine-factor maximum");
  assert.match(shader,
    /letvalid=structuredValid\(\)&&airPublicationValid\(\).*scheduleValid=valid&&governorInvalid\[0\]==0u&&required<=64u&&displacement<=p\.maxBacktrace.*activeSteps=select\(0u,required,scheduleValid\)/s,
    "invalid velocity or an over-bound characteristic must publish zero work");
  assert.match(shader,
    /state\[32\]=select\(0u,accepted\[2\],scheduleValid\).*state\[33\]=select\(0u,activeBank\(\),scheduleValid\).*state\[34\]=select\(0u,accepted\[3\],scheduleValid\)/s,
    "the plan must snapshot row count, bank, and accepted epoch");
  assert.match(shader,
    /letvalid=state\[0\]==0u&&accepted\[2\]==state\[32\]&&activeBank\(\)==state\[33\]&&accepted\[3\]==state\[34\]/,
    "class worksets must be empty when the accepted A/B publication changes");
  assert.match(shader, /fnairOwner\(cell:u32\)->AirOwner/);
  assert.match(shader, /fnairOwnerAtPosition\(x:vec3f\)->AirOwner/);
  assert.doesNotMatch(shader, /publicationRow|rowGeometry|@binding\(11\)/,
    "fine transport must consume the coalesced dense owner directory without a row-directory chase");
  assert.match(shader, /fncanonicalVelocity\(row:u32\)->vec4f/);
  assert.match(shader,
    /maximum=max\(maximum,length\(v\.xyz\)\).*ceil\(d\.maximumSpeed\*p\.dt\/p\.h\)/s,
    "transport displacement telemetry must measure the sampled speed times dt over fine-cell width");
  assert.doesNotMatch(shader, /rowDirectory|powerFaceControl|faceBandControl|velocitySampleControl|fallback|legacy/i);
});

test("sparse-page classification ignores unowned sample slots but rejects corrupt owned phi", () => {
  const shader = compact(structuredFineLevelSetTransportWGSL);
  const start = shader.indexOf("fnclassifyStructuredFineTransportBlocks");
  const end = shader.indexOf("@compute@workgroup_size(1)fnpublishStructuredFineTransportWorksets", start);
  assert.ok(start >= 0 && end > start, "transport classifier must have a bounded WGSL region");
  const classify = shader.slice(start, end);
  assert.match(classify, /letvalid=id!=INVALID&&classifyInvalid\[0\]==0u/,
    "page classification must remain independent of an arbitrary page-anchor owner");
  assert.doesNotMatch(classify, /ownerRequired|!ownerRequired\|\|owner\.tag!=INVALID/,
    "the actual owner at each trajectory point, not the page anchor, gates velocity support");
  assert.match(classify,
    /letsampleValid=\(flags\[index\]&VALID\)!=0u;if\(sampleValid\)\{letvalue=phi\[index\];invalid\|=select\(1u,0u,finite\(value\)\)/,
    "only owned fine samples require finite phi");
  assert.match(classify, /else\{rare=1u;\}/,
    "a sparse page with unowned slots must use the validity-aware rare kernel");
  assert.doesNotMatch(classify,
    /\(flags\[index\]&VALID\)!=0u&&finite\(phi\[index\]\)&&row<state\[32\]/,
    "an unowned sparse slot must not reject its whole resident page");
});

test("outer redistance support is copied without velocity and failures retain bounded page diagnostics", () => {
  const shader = compact(structuredFineLevelSetTransportWGSL);
  assert.match(shader, /fninTransportBand\(value:f32\)->bool\{returnfinite\(value\)&&abs\(value\)<=f32\(p\.bandCells\)\*p\.h;\}/,
    "the host-published transport band must define the characteristic domain");
  for (const name of ["runRegularCommon", "runRegularRare", "runTransitionCommon", "runTransitionRare"]) {
    const start = shader.indexOf(`fn${name}`);
    const end = shader.indexOf("fn", start + 2);
    const run = shader.slice(start, end);
    assert.match(run, /if\(!inTransportBand\(old\)\)\{nextPhi\[index\]=old;continue;\}/,
      `${name} must preserve outer support without requesting velocity`);
    assert.match(run, /firstBad=min\(firstBad,select\(INVALID,local,s\.bad!=0u\)\)/,
      `${name} must retain the first true in-band miss`);
  }
  assert.match(shader,
    /state\[base\+2u\]=\(reduceDisplacement\[0\]&65535u\)\|\(min\(reduceFirstBad\[0\],65535u\)<<16u\)/,
    "the existing per-page status word must pack displacement and first failure without a new buffer");
  assert.match(shader,
    /if\(work==INVALID\)\{control\.nonfinite\+=1u;control\.invalidStatus\+=1u;continue;\}/,
    "classification rejection must fail closed without decoding the INVALID marker as two 65535 counters");
  assert.match(shader, /letpacked=state\[base\+2u\];[\s\S]*packed&65535u/,
    "summary displacement must ignore the packed first-failure half-word");
});

test("every trajectory reselects regular or transition interpolation m times and gathers phi once", () => {
  const shader = compact(structuredFineLevelSetTransportWGSL);
  const regularCommon = shader.slice(shader.indexOf("fnregularCommonDeparture"), shader.indexOf("fnregularRareDeparture"));
  const regularRare = shader.slice(shader.indexOf("fnregularRareDeparture"), shader.indexOf("fntransitionCommonDeparture"));
  const transitionCommon = shader.slice(shader.indexOf("fntransitionCommonDeparture"), shader.indexOf("fntransitionRareDeparture"));
  const transitionRare = shader.slice(shader.indexOf("fntransitionRareDeparture"), shader.indexOf("fnfinishSample"));
  for (const path of [regularCommon, regularRare, transitionCommon, transitionRare]) {
    assert.match(path, /for\(varstage=0u;stage<state\[1\];stage\+=1u\)/,
      "each class must execute exactly the GPU-selected Section 5 substeps");
    assert.doesNotMatch(path, /stage<64u|if\(stage<state\[1\]\)/,
      "the validated runtime count must not retain a masked 64-iteration loop");
    assert.doesNotMatch(path, /sampleFine\(/, "departure integration must not gather phi per segment");
  }
  assert.match(regularCommon, /letv=transitionSample\(x\)/);
  assert.match(regularRare, /letv=transitionSample\(x\).*extended\+=select\(0u,1u,air\)/s);
  assert.match(shader,
    /fntransitionSample\(x:vec3f\)->vec4f.*letsampleX=velocityDomainPoint\(x\).*if\(caseId==0u\)\{letregular=regularSampleExact\(sampleX,owner\);if\(regular\.exact!=0u\)\{returnregular\.value;\}\}.*letweights=tetraWeights/s,
    "each Euler step fuses its eight exact cube-owner proofs with the trilinear gather and otherwise retains case-zero tetrahedra");
  assert.doesNotMatch(shader, /regularCubeExact|fnregularSample\(/,
    "the regular path must not repeat owner-directory lookups in a proof pass and a gather pass");
  assert.match(transitionCommon, /letv=transitionSample\(x\)/);
  assert.match(transitionRare, /letv=transitionSample\(x\).*extended\+=select\(0u,1u,air\)/s);
  assert.equal([...shader.matchAll(/sampleFine\(/g)].length, 2,
    "WGSL must contain one sampleFine declaration and one final-trajectory invocation");
  assert.match(shader, /fnfinishSample.*if\(d\.good!=0u\)\{sampled=sampleFine\(x\);\}/s);
});

test("missing sparse phi support remains a fail-closed sentinel", () => {
  const shader = compact(structuredFineLevelSetTransportWGSL);
  assert.match(shader, /fnfinite\(v:f32\)->bool\{returnv==v&&abs\(v\)<3\.402823e38;\}/,
    "FLT_MAX must not pass the scalar validity predicate used by finishSample");
  assert.doesNotMatch(shader, /abs\(v\)<=3\.402823e38/,
    "the missing-stencil sentinel must never be committed as transported phi");
  assert.match(shader,
    /fnsampleFine.*returnselect\(3\.402823e38,.*weight>0\.999\).*fnfinishSample.*acceptedSample=d\.good!=0u&&finite\(sampled\).*if\(acceptedSample\)\{nextPhi\[index\]=sampled;\}else\{nextPhi\[index\]=old;\}/s,
    "an incomplete trilinear stencil must preserve the previous finite phi and reject publication telemetry");
});

test("transition interpolation rejects missing positive-weight support", () => {
  const shader = compact(structuredFineLevelSetTransportWGSL);
  assert.match(shader,
    /fnselectorVelocity\(owner:AirOwner,selectorIndex:u32,selector:vec4f\)->vec4f.*if\(all\(selectorCenter>=lower-vec3f\(tolerance\)\)&&all\(selectorCenter<=upper\+vec3f\(tolerance\)\)\)\{letother=airOwnerAtPosition\(selectorCenter\);if\(other\.tag==INVALID.*returnvec4f\(0,0,0,-1\).*returntaggedVelocity\(other\.tag\);\}returntaggedVelocity\(owner\.tag\)/s,
    "an absent in-domain selector must carry invalidity instead of a zero vector");
  assert.match(shader,
    /varw=max\(weights,vec4f\(0\)\).*w\/=total.*if\(w\.x>0\.\).*if\(w\.y>0\.\).*if\(v\.w<0\.\|\|!finite3\(v\.xyz\)\)\{returninvalid;\}/s,
    "only zero-weight vertices may be skipped; every positive contributor remains strict");
  assert.match(shader,
    /fnregularSampleExact.*letowner=airOwnerAtPosition\(samplePoint\).*owner\.tag==INVALID\|\|owner\.size!=anchor\.size.*returnRegularAttempt\(vec4f\(0,0,0,-1\),0u\).*if\(w==0\.\)\{continue;\}letvalue=taggedVelocity\(owner\.tag\)/s,
    "regular interpolation must distinguish a non-cube topology fallback from invalid selected-cube velocity");
  assert.match(shader, /fnselectorGeometryValid.*length\(selector\.xyz\)>=1e-7\|\|\(all\(selector\.xyz==vec3f\(0\.\)\)&&selector\.w==1\.\)/s,
    "only the exact zero/unit selector may alias the anchor owner");
  assert.match(shader,
    /p\.tetraVertexCount>\(words-p\.tetraVertexOffset\)\/4u.*words-thAt<3u.*first>words-p\.tetraOffset.*count>words-p\.tetraOffset-first/s,
    "tetra header, payload, and neighbor-selector table must validate before indexed catalog reads");
  assert.doesNotMatch(shader, /anchor=vec4f\(bitsf\(p\.tetraVertexOffset\)/,
    "the current-cell anchor is implicit and must not be read from the neighbor-selector table");
  assert.match(shader, /fntetraWeights.*!finite\(d\)\|\|abs\(d\)<1e-10/,
    "a non-finite or degenerate tetrahedron determinant must fail closed");
  assert.doesNotMatch(shader, /returnselect\(vec3f\(0\),rowV\(other\),other!=INVALID\)/,
    "fine transport must not retain the retired zero substitute");
});

test("open-domain characteristics sample cube and transition velocity on the same boundary point", () => {
  const shader = compact(structuredFineLevelSetTransportWGSL);
  assert.match(shader,
    /fnvelocityDomainPoint\(x:vec3f\)->vec3f\{returnclamp\(x,p\.origin\+vec3f\(\.5\*p\.physical\),p\.origin\+vec3f\(p\.dimensions\)\*p\.physical-vec3f\(\.5\*p\.physical\)\);\}/,
    "velocity continuation must project a departed characteristic to the nearest octree cell centre");
  assert.match(shader,
    /fnregularSampleExact.*letclamped=velocityDomainPoint\(x\)/s,
    "regular cube interpolation must use the shared boundary continuation");
  assert.match(shader,
    /fntransitionSample.*letsampleX=velocityDomainPoint\(x\).*airOwnerAtPosition\(sampleX\).*regularSampleExact\(sampleX,owner\).*powerTransform\(\(sampleX-center\)\/extent,transform\)/s,
    "power-tetra interpolation must locate and interpolate the same projected boundary point");
  assert.match(shader,
    /fnfinishSample.*if\(!insideDomain\(x\)\)\{outside=1u.*sampleFine\(x\)/s,
    "boundary velocity continuation must not erase final departure and phi-support diagnostics");
});

test("fine phi commit publishes old and new interface membership atomically", () => {
  const shader = compact(structuredFineLevelSetTransportWGSL);
  assert.match(shader,
    /fncommitStructuredFineTransport.*oldInterface.*newInterface.*membership=pageChanged\[0\]!=0u\|\|before\|\|after.*phi\[index\]=nextPhi\[index\]/s);
  assert.match(shader,
    /fnpublishStructuredFineDelta.*delta\[0\]=count.*delta\[1\]=p\.generation.*delta\[2\]=control\.committed/s);
  assert.doesNotMatch(shader, /atomic/,
    "page-owned reductions and deterministic compaction must not revive an append race");
});

test("WGSL declares exactly the ten direct specialized production entry points", () => {
  const entries = [...structuredFineLevelSetTransportWGSL.matchAll(
    /@compute\s*@workgroup_size\([^)]*\)\s*fn\s+([A-Za-z_]\w*)/g,
  )].map((match) => match[1]);
  assert.deepEqual(entries, [
    "planStructuredFineTransportSubsteps",
    "classifyStructuredFineTransportBlocks",
    "publishStructuredFineTransportWorksets",
    "transportRegularCommonPhi",
    "transportTransitionCommonPhi",
    "transportRegularRarePhi",
    "transportTransitionRarePhi",
    "summarizeStructuredFineTransport",
    "commitStructuredFineTransport",
    "publishStructuredFineDelta",
  ]);
});
