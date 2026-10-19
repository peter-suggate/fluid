import assert from 'node:assert/strict';
import test from 'node:test';
import {isolateComputePassEncoders} from '../lib/harness/webgpu-pass-encoder-isolation';

test('counter isolation inserts a disjoint blit before each compute pass',()=>{
 const commands:unknown[][]=[];const scratch={source:{} as GPUBuffer,destination:{} as GPUBuffer,destroy(){}};
 const raw={copyBufferToBuffer:(...args:unknown[])=>commands.push(['copy',...args]),beginComputePass:(...args:unknown[])=>{commands.push(['begin',...args]);return 'pass';}} as unknown as GPUCommandEncoder;
 const encoder=isolateComputePassEncoders(raw,scratch),descriptor={label:'sample'};
 assert.equal(encoder.beginComputePass(descriptor),'pass');
 assert.deepEqual(commands,[['copy',scratch.source,0,scratch.destination,0,4],['begin',descriptor]]);
});
