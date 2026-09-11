import assert from "node:assert/strict";
import test from "node:test";
import { CM12ResidentGeneration } from "../lib/sparse-world/internal/adaptive-volume-adapter";
import type { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident";

function deferred<T = void>() {
 let resolve!: (value: T | PromiseLike<T>) => void;
 let reject!: (reason?: unknown) => void;
 const promise = new Promise<T>((yes,no) => {resolve=yes;reject=no;});
 return {promise,resolve,reject};
}
const settled = () => new Promise<void>(resolve => setImmediate(resolve));
function resident(bytes=64, ready:Promise<void>=Promise.resolve()) {
 let destroyed=0, leases=0, releases=0;
 const value={allocatedBytes:bytes,waitForSimulationPipelines:()=>ready,destroy(){destroyed++;},
  acquireGenerationSourceTopologyLease(){leases++;let released=false;return()=>{if(!released){released=true;releases++;}};}} as unknown as WebGPUSparseCM12Resident;
 return {value,get destroyed(){return destroyed;},get leases(){return leases;},get releases(){return releases;}};
}
function device(fence:Promise<void>=Promise.resolve()) {
 return {queue:{onSubmittedWorkDone:()=>fence}} as unknown as GPUDevice;
}

test("resident construction remains advanceable; only final publication is pending", async () => {
 const accepted=resident(), next=resident(128);
 const owner=new CM12ResidentGeneration(device(),accepted.value);
 const built=deferred(), committed=deferred(), enteredCommit=deferred();
 let disposed=0;
 const work=owner.prepare(async current=>{
  assert.equal(current,accepted.value);
  assert.equal(accepted.leases,0,"owner must not hold topology through background construction");
  await built.promise;
  return {resident:next.value,commit:async()=>{enteredCommit.resolve();await committed.promise;},
   disposePreparation(){disposed++;}};
 });
 assert.equal(owner.pending,false);
 assert.equal(owner.current,accepted.value);
 assert.throws(()=>owner.prepare(async()=>undefined),/unavailable/);
 built.resolve(); await enteredCommit.promise;
 assert.equal(owner.pending,true);
 assert.equal(owner.current,accepted.value);
 committed.resolve(); await work; await settled();
 assert.equal(owner.pending,false);
 assert.equal(owner.current,next.value);
 assert.equal(owner.publications,1);
 assert.ok(owner.maximumPublicationMs>=0);
 assert.equal(accepted.releases,0);
 assert.equal(disposed,1);
 assert.equal(accepted.destroyed,1);
 assert.equal(next.destroyed,0);
 owner.destroy(); await settled(); assert.equal(next.destroyed,1);
});

test("old read leases and the queue fence defer retirement after publication", async () => {
 const accepted=resident(64), next=resident(128);
 const fence=deferred(), read=deferred<number>();
 const owner=new CM12ResidentGeneration(device(fence.promise),accepted.value);
 const lease=owner.read(async captured=>{
  assert.equal(captured,accepted.value); return read.promise;
 });
 await owner.prepare(async()=>({resident:next.value,commit:async()=>{},disposePreparation(){}}));
 assert.equal(owner.current,next.value);
 assert.equal(owner.allocatedBytes,192);
 assert.equal(accepted.destroyed,0);
 read.resolve(7); assert.equal(await lease,7); await settled();
 assert.equal(accepted.destroyed,0,"queue work still holds the retired resident");
 fence.resolve(); await settled();
 assert.equal(accepted.destroyed,1);
 assert.equal(owner.allocatedBytes,128);
 owner.destroy(); await settled();
});

test("an edit aborts preparation and disposes its candidate before commit", async () => {
 const accepted=resident(), next=resident();
 const owner=new CM12ResidentGeneration(device(),accepted.value);
 const built=deferred(); let signal!:AbortSignal; let commits=0,disposed=0;
 const work=owner.prepare(async(_,inputSignal)=>{
  signal=inputSignal; await built.promise;
  return {resident:next.value,commit:async()=>{commits++;},disposePreparation(){disposed++;}};
 });
 const rejected=assert.rejects(work,/scene changed/);
 owner.changed(); assert.equal(signal.aborted,true);
 assert.equal(accepted.leases,0);
 built.resolve(); await rejected;
 assert.equal(commits,0); assert.equal(disposed,1); assert.equal(next.destroyed,1);
 assert.equal(owner.current,accepted.value); assert.equal(owner.publications,0);
 assert.equal(owner.pending,false);
 await owner.prepare(async()=>undefined);
 owner.destroy(); await settled(); assert.equal(accepted.destroyed,1);
});

test("a failed publication retains accepted state and releases preparation resources", async () => {
 const accepted=resident(), next=resident(); let disposed=0;
 const owner=new CM12ResidentGeneration(device(),accepted.value);
 await assert.rejects(owner.prepare(async()=>({resident:next.value,
  commit:async()=>{throw new Error("transfer failed");},disposePreparation(){disposed++;}})),/transfer failed/);
 assert.equal(owner.current,accepted.value); assert.equal(owner.pending,false);
 assert.equal(owner.publications,0); assert.equal(accepted.destroyed,0);
 assert.equal(next.destroyed,1); assert.equal(disposed,1);
 owner.destroy(); await settled();
});

test("destroy aborts the builder, cancels adoption, and retires both residents once", async () => {
 const accepted=resident(), next=resident(); const built=deferred();
 const owner=new CM12ResidentGeneration(device(),accepted.value);
 let signal!:AbortSignal; let disposed=0;
 const work=owner.prepare(async(_,inputSignal)=>{
  signal=inputSignal; await built.promise;
  return {resident:next.value,commit:async()=>assert.fail("destroyed owner committed"),
   disposePreparation(){disposed++;}};
 });
 const rejected=assert.rejects(work,/scene changed/);
 owner.destroy(); owner.destroy(); assert.equal(signal.aborted,true);
 assert.equal(accepted.leases,0);
 built.resolve(); await rejected; await settled();
 assert.equal(accepted.destroyed,1); assert.equal(next.destroyed,1); assert.equal(disposed,1);
 assert.equal(owner.pending,false);
 assert.throws(()=>owner.prepare(async()=>undefined),/unavailable/);
});

test("an edit during the short commit cannot publish a stale generation", async () => {
 const accepted=resident(), next=resident(); const commit=deferred(), entered=deferred();
 const owner=new CM12ResidentGeneration(device(),accepted.value);
 let disposed=0;
 const work=owner.prepare(async()=>({resident:next.value,commit:async()=>{entered.resolve();await commit.promise;},
  disposePreparation(){disposed++;}}));
 const rejected=assert.rejects(work,/scene changed during resident publication/);
 await entered.promise; assert.equal(owner.pending,true);
 owner.changed(); commit.resolve(); await rejected;
 assert.equal(owner.current,accepted.value); assert.equal(owner.publications,0);
 assert.equal(next.destroyed,1); assert.equal(disposed,1); assert.equal(owner.pending,false);
 owner.destroy(); await settled();
});
