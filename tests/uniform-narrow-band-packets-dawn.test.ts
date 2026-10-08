import assert from "node:assert/strict";
import test from "node:test";
import { NarrowBandParticlePackets } from "../tools/narrow-band-particle-packets";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("Particle tile packets cover every index once, bound work to 64, and refresh topology",async()=>{
  await withUniformDevice("particle tile packets",async device=>{
    const dims=[32,16,8],tiles=64,capacity=1024;
    const buffer=(size:number)=>device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    const particles=[buffer(48*capacity),buffer(48*capacity)],state=buffer(48),workspace=buffer(4*(16+capacity+4*(capacity/64+tiles)));
    const topology=buffer(5*tiles*4),support=buffer(4*tiles*4);
    const packets=new NarrowBandParticlePackets(device,dims,particles,state,workspace,topology,support);
    const tileOf=(q:readonly number[])=>Math.floor(Math.max(0,Math.min(31.999,q[0]!))/4)+8*(Math.floor(Math.max(0,Math.min(15.999,q[1]!))/4)+4*Math.floor(Math.max(0,Math.min(7.999,q[2]!))/4));
    try {
      await packets.initialize();
      for(const [epoch,count] of [0,1,65,1024,7,0].entries()){
        const parity=epoch%2,input=new Float32Array(12*capacity),expected=new Map<number,number[]>();
        for(let i=0;i<count;i++){
          // A large crowded tile, scattered nonempty tiles and domain-edge
          // positions. Invalid positions must still reach advection once so
          // its existing collision/outflow rules decide their fate.
          const q=i<700?[4.25,4.5,1.75]:[i%32+0.25,(i*7)%16+0.5,(i*3)%8+0.75];
          if(i===0)q[0]=-0.25;if(i===1)q[0]=32;
          input.set(q,12*i);const t=tileOf(q);const ids=expected.get(t)??[];ids.push(i);expected.set(t,ids);
        }
        const top=new Uint32Array(5*tiles),sup=new Uint32Array(4*tiles);
        for(let t=0;t<tiles;t++){top[2*tiles+2*t]=(t+epoch)%3===0?1<<27:4<<27;sup[3*tiles+t]=(t+epoch)%2;}
        device.queue.writeBuffer(particles[parity]!,0,input);device.queue.writeBuffer(state,0,new Uint32Array([count]));
        device.queue.writeBuffer(topology,0,top);device.queue.writeBuffer(support,0,sup);
        const encoder=device.createCommandEncoder();packets.encode(encoder,parity);device.queue.submit([encoder.finish()]);
        const raw=await readMixedBuffer(device,workspace),data=new Uint32Array(raw.buffer,raw.byteOffset,raw.length);
        const seen=new Set<number>();let expectedPackets=0;for(const ids of expected.values())expectedPackets+=Math.ceil(ids.length/64);
        assert.equal(data[0],expectedPackets);assert.equal(data[4],expectedPackets);assert.equal(data[5],count?1:0);assert.equal(data[6],1);
        assert.equal(data[8],Math.ceil(count/64));assert.equal(data[9],count?1:0);assert.equal(data[10],1);
        for(let job=0;job<data[0]!;job++){
          const at=packets.packetBase+4*job,first=data[at]!,end=data[at+1]!,tile=data[at+2]!,regular=data[at+3]!;
          assert.ok(end>first&&end-first<=64);assert.equal(regular,Number((sup[3*tiles+tile]!&1)!==0&&(top[2*tiles+2*tile]!>>>27)===1));
          for(let j=first;j<end;j++){
            const id=data[16+j]!;assert.ok(id<count);assert.ok(!seen.has(id),"no duplicated particle");seen.add(id);
            assert.equal(tileOf(Array.from(input.subarray(12*id,12*id+3))),tile,"every lane belongs to its descriptor's tile");
          }
        }
        assert.equal(seen.size,count,"no omitted particle, including empty-after-full epochs");
      }
    } finally {packets.destroy();for(const b of [...particles,state,workspace,topology,support])b.destroy();}
  });
});
