import assert from "node:assert/strict";
import test from "node:test";
import { UniformMixedFramePlan } from "../lib/methods/uniform/uniform-mixed-frame-plan";

function fixture(surfaceOnly=true){
 const events:string[]=[];
 const pass={setBindGroup(){},setPipeline(){},dispatchWorkgroups(){},end(){}};
 const encoder={clearBuffer(_buffer:unknown,offset:number){events.push(`clear:${offset}`);},beginComputePass(d:GPUComputePassDescriptor){events.push(d.label!);return pass;}} as unknown as GPUCommandEncoder;
 const plan=Object.create(UniformMixedFramePlan.prototype) as UniformMixedFramePlan;
 const ownership={revision:3,capacity:{tiles:8,tileDimensions:[2,2,2]},bindGroup:{},dispatchX:65535,
  dispatchAllCounted(){events.push("owners");}};
 Object.assign(plan,{surfaceOnly,ownership,supportRevision:-1,supportSettings:[],group:{group:{}},extendedGroup:{group:{}},params:{},
  device:{queue:{writeBuffer(){events.push("policy");}}},pipelines:new Map(["seed","dilate0","dilate1","dilate2","localSpeed","spread0","spread1","spread2","prefix0","prefix1","prefix2","certify","speed"].map(name=>[name,{}]))});
 return {plan,ownership,events,encoder};
}

test("speed refresh requires a matching prior surface plan and ownership revision",()=>{
 const {plan,ownership,events,encoder}=fixture();
 plan.encodeSpeed(encoder);
 assert.deepEqual(events,["policy","clear:128","clear:0","Uniform shared frame plan","owners"]);
 events.length=0;plan.encodeSpeed(encoder);
 assert.deepEqual(events,["clear:128","Uniform shared frame speed","owners"]);
 ownership.revision++;events.length=0;plan.encodeSpeed(encoder);
 assert.ok(events.includes("clear:0"),"a changed layout rebuilds support");
 events.length=0;plan.encodeSpeed(encoder,{fineReach:3,shellReach:1,twoLevel:true,shellOnly:true});
 assert.ok(events.includes("clear:0"),"a changed reach rebuilds support");
 events.length=0;plan.encodeSpeed(encoder,{fineReach:3,shellReach:1,twoLevel:true,shellOnly:true});
 assert.deepEqual(events,["clear:128","Uniform shared frame speed","owners"]);
});

test("volume-derived support never takes the surface-only reuse path",()=>{
 const {plan,events,encoder}=fixture(false);plan.encode(encoder);events.length=0;plan.encodeSpeed(encoder);
 assert.ok(events.includes("clear:0"));assert.ok(events.includes("Uniform shared frame plan"));
});
