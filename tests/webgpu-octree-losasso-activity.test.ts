import assert from "node:assert/strict";
import test from "node:test";
import { octreeLosassoFineTransportWGSL }
  from "../lib/webgpu-octree-losasso-fine-transport.wgsl";
import { octreeLosassoExtensionBandWGSL }
  from "../lib/webgpu-octree-losasso-extension-band.wgsl";
import { WebGPUOctreeLosassoExtensionBand }
  from "../lib/webgpu-octree-losasso-extension-band";
import {
  OCTREE_LOSASSO_FINE_TRANSPORT_DISPATCH_BYTES,
  planOctreeLosassoFineTransport,
  WebGPUOctreeLosassoFineTransport,
} from "../lib/webgpu-octree-losasso-fine-transport";

const compact = (source: string) => source.replace(/\s+/g, "");

test("Losasso fine transport compacts awake pages before expensive transport", () => {
  const shader = compact(octreeLosassoFineTransportWGSL);
  const encode = compact(WebGPUOctreeLosassoFineTransport.prototype.encode.toString());
  assert.equal(OCTREE_LOSASSO_FINE_TRANSPORT_DISPATCH_BYTES, 48);
  assert.match(shader,
    /fnclassifyLosassoFineActivity.*pageAwake\(metadata\[4u\*page\+3u\]\).*atomicAdd\(&control\[14\],1u\).*delta\[8u\+2u\*p\.pageCapacity\+rank\]=work/s);
  assert.match(shader,
    /fnfinalizeLosassoFineActivity.*liveDispatch\[4\].*active.*acceptedStep/s);
  assert.match(shader,
    /flags&\(PAGE_INTERFACE\|PAGE_ACTIVITY_MOVING\|PAGE_WAKE_HALO\|PAGE_DIRTY\)/,
    "interface pages must not sleep merely because the previous trace had zero displacement");
  assert.match(encode,
    /classifyActivity.*dispatchWorkgroups\(Math\.ceil\(this\.plan\.pageCapacity\/64\)\).*finalizeActivity.*fence\("Losassofinetransportactivedispatchpublication"\).*advect.*dispatchWorkgroupsIndirect\(this\.liveDispatch,16\).*commit.*dispatchWorkgroupsIndirect\(this\.liveDispatch,16\)/s);
});

test("Losasso sleep keeps full live-order delta publication", () => {
  const shader = compact(octreeLosassoFineTransportWGSL);
  const encode = compact(WebGPUOctreeLosassoFineTransport.prototype.encode.toString());
  assert.match(shader,
    /fnpublishLosassoFineDelta.*letwork=group\.x.*delta\[8u\+p\.pageCapacity\+work\]=key/s);
  assert.match(shader,
    /if\(!awake\).*PAGE_ACTIVITY_VALID.*delta\[8u\+page\]=select\(INVALID,key,\(flags&PAGE_INTERFACE\)!=0u\)/s);
  assert.match(shader,
    /letaccepted=acceptedStep\(\)&&\(live==0u\|\|activePages==0u\|\|atomicLoad\(&control\[2\]\)>0u\)/,
    "an all-asleep generation must still publish an accepted identity receipt");
  assert.match(encode,
    /publishDelta.*dispatchWorkgroupsIndirect\(this\.liveDispatch,32\)/,
    "delta publication must retain the sorted live worklist domain");
});

test("Losasso fine transport plan accounts for activity dispatches", () => {
  const plan = planOctreeLosassoFineTransport({ plan: {
    maximumResidentBricks: 4_096,
    samplesPerBrick: 64,
  } } as never);
  assert.equal(plan.encodedDispatchCount, 7);
  assert.equal(plan.allocatedBytes,
    160 + 64 + 48 + (8 + 3 * 4_096) * 4);
});

test("Losasso extension publication owns one fine page per invocation", () => {
  const shader = compact(octreeLosassoExtensionBandWGSL);
  const encode = compact(WebGPUOctreeLosassoExtensionBand.prototype.encodePublication.toString());
  assert.match(shader,
    /fnpublishLosassoAirBandFaces.*letwork=linearInvocation\(group,lane\).*for\(varlocal=0u;local<facesPerBrick;local\+=1u\)/s);
  assert.match(encode,
    /run\("publishLosassoAirBandFaces",Math\.ceil\(finePlan\.maximumResidentBricks\/64\)\)/);
  assert.doesNotMatch(encode, /macFacesPerBrick\*finePlan\.maximumResidentBricks/);
  assert.match(encode,
    /prepareLosassoBandDispatch.*runIndirect\("dilateLosassoAirBand5"\).*prepareLosassoBandDispatch.*runIndirect\("dilateLosassoAirBand6"\).*prepareLosassoBandDispatch.*runIndirect\("dilateLosassoAirBand7"\).*prepareLosassoBandDispatch.*runIndirect\("buildLosassoExtensionAdjacency"\)/s);
  assert.match(shader,
    /fnprepareLosassoBandDispatch\(\).*count=min\(atomicLoad\(&control\[2\]\),p\.band\.x\).*liveFaceDispatch\[0\]=\(count\+63u\)\/64u/s);
});
