import assert from 'node:assert/strict';
import test from 'node:test';
import {GPUInitializationTaskRunner, settleGPUInitialization, type GPUInitializationSnapshot, type GPUInitializationTask} from '../lib/core/gpu-initialization';

test('failed initialization drains sibling continuations before resource cleanup', async () => {
 let release!:()=>void;
 const gate=new Promise<void>(resolve=>{release=resolve;});
 const events:string[]=[];
 const failure=new Error('allocation failed');
 const pending=settleGPUInitialization([
  Promise.reject(failure),
  gate.then(()=>{events.push('sibling finished');}),
 ]).catch(error=>{events.push('cleanup');assert.equal(error,failure);});
 await Promise.resolve();await Promise.resolve();
 assert.deepEqual(events,[]);
 release();await pending;
 assert.deepEqual(events,['sibling finished','cleanup']);
});

test('cancelling the final startup task cannot publish a completed build', async () => {
 const controller=new AbortController();
 const runner=new GPUInitializationTaskRunner(()=>{},controller.signal);
 await assert.rejects(runner.run([{id:'last',phase:'attach',label:'Last',run:()=>controller.abort()}]),{name:'AbortError'});
 assert.equal(runner.completedCount,0);
});

test('parallel startup reports finished tasks while a slow compile is pending', async () => {
 const snapshots:GPUInitializationSnapshot[]=[];
 const runner=new GPUInitializationTaskRunner(s=>snapshots.push(s),new AbortController().signal);
 let release!:()=>void, started!:()=>void;
 const barrier=new Promise<void>(resolve=>{release=resolve;});
 const ready=new Promise<void>(resolve=>{started=resolve;});
 const tasks:GPUInitializationTask[]=[
  {id:'fast',label:'Fast',phase:'solver-pipelines',run:async()=>{}},
  {id:'slow',label:'Slow',phase:'solver-pipelines',workUnits:3,run:async(_signal,report)=>{
   report?.('Slow partial',1);started();await barrier;
  }},
  {id:'dependent',label:'Dependent',phase:'attach',dependencies:['fast','slow'],run:()=>{
   assert.equal(runner.completedCount,2);
  }},
 ];
 const running=runner.run(tasks);await ready;await Promise.resolve();
 assert.equal(runner.completedCount,1);
 assert.equal(snapshots.at(-1)?.taskId,'slow');
 assert.equal(snapshots.at(-1)?.completed,2);
 release();await running;
 assert.equal(snapshots.at(-1)?.completed,5);
 for(let i=1;i<snapshots.length;i++)assert.ok(snapshots[i]!.completed>=snapshots[i-1]!.completed);
});
