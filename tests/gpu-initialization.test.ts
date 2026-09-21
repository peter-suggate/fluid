import assert from 'node:assert/strict';
import test from 'node:test';
import {GPUInitializationTaskRunner, type GPUInitializationSnapshot, type GPUInitializationTask} from '../lib/core/gpu-initialization';

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
