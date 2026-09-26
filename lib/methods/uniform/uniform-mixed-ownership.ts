import type { UniformMixedLayout } from "./uniform-mixed-layout";

/** Shared owner/worklist ABI for transport, face operations and pressure.
 * Two fixed-size buffers support live region edits without pipeline rebuilds.
 */
export class UniformMixedOwnership {
  readonly bindLayout: GPUBindGroupLayout;
  readonly bindGroup: GPUBindGroup;
  readonly allocatedBytes: number;
  readonly dispatchX: number;
  private readonly topology: GPUBuffer;
  /** Stable read-only view for consumers of the accepted ownership generation. */
  readonly presentation: GPUBufferBinding;
  private readonly counts: GPUBuffer;
  /** Per tile seed and three separable support planes, rebuilt at frame entry. */
  readonly support: GPUBuffer;
  readonly certifiedDispatch:GPUBuffer;
  private seamCounts=[0,0,0];

  private frameHeld=false;
  private currentLayout: UniformMixedLayout;
  get layout(): UniformMixedLayout { return this.currentLayout; }

  constructor(private readonly device: GPUDevice, layout: UniformMixedLayout) {
    this.currentLayout = layout;
    this.dispatchX = device.limits.maxComputeWorkgroupsPerDimension;
    this.bindLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    this.topology = device.createBuffer({ label: "Uniform mixed owners and tier worklists", size: layout.metadataBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE });
    this.presentation={buffer:this.topology,size:this.topology.size};
    this.counts = device.createBuffer({ label: "Uniform mixed work counts", size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM });
    this.support = device.createBuffer({label:"Uniform shared frame support and certified work",size:(layout.tiles.length*8+20)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    this.certifiedDispatch=device.createBuffer({label:"Uniform certified fine dispatch",size:32,usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
    this.update(layout);
    this.allocatedBytes = this.topology.size + this.counts.size + this.support.size+this.certifiedDispatch.size;
    this.bindGroup = device.createBindGroup({ layout: this.bindLayout, entries: [
      { binding: 0, resource: { buffer: this.topology } }, { binding: 1, resource: { buffer: this.counts } },
      { binding: 2, resource: { buffer: this.support } },
    ] });
  }

  /** Call between submitted frames, after remapping fields from the previous
   * ownership. Queue writes are ordered after earlier submitted GPU work. */
  update(layout: UniformMixedLayout): void {
    if(this.frameHeld)throw new Error("Ownership is immutable during an active frame");
    const prior=this.currentLayout.lattice;
    if(layout.metadataBytes!==this.topology.size || layout.lattice.dimensions.some((n,a)=>n!==prior.dimensions[a])
      || layout.lattice.cellSize_m.some((n,a)=>n!==prior.cellSize_m[a])
      || (["x","y","z"] as const).some(a=>layout.lattice.origin_m[a]!==prior.origin_m[a]))
      throw new Error("Live ownership edits cannot change the simulation lattice");
    const words=new Uint32Array(layout.metadataBytes/4);let offset=0;
    for(const part of [layout.tiles,layout.fineTiles,layout.transitionTiles,layout.coarseTiles,layout.stencils]){words.set(part,offset);offset+=part.length;}
    this.device.queue.writeBuffer(this.topology,0,words);
    // A uniform loop bound prevents explosive Metal sampler unrolling.
    this.device.queue.writeBuffer(this.counts,0,new Uint32Array([layout.fineTiles.length,layout.transitionTiles.length,layout.coarseTiles.length,8]));
    // Standalone stages conservatively visit everything until a frame census.
    this.device.queue.writeBuffer(this.support,0,new Uint32Array(layout.tiles.length*4).fill(3));
    // Chessboard distance to a non-fine tile, fixed for this ownership
    // generation. A frame can certify an entire characteristic's footprint
    // with one distance test instead of rediscovering ownership at every tap.
    const n=layout.tiles.length,d=layout.tileDimensions,distance=new Uint32Array(n).fill(0xffffffff),queue:number[]=[];
    for(let t=0;t<n;t++)if((layout.tiles[t]!&0x80000000)===0){distance[t]=0;queue.push(t);}
    for(let at=0;at<queue.length;at++){
      const t=queue[at]!,p=[t%d[0],Math.floor(t/d[0])%d[1],Math.floor(t/(d[0]*d[1]))];
      for(let z=-1;z<=1;z++)for(let y=-1;y<=1;y++)for(let x=-1;x<=1;x++){
        const q=[p[0]!+x,p[1]!+y,p[2]!+z];if(q.some((v,a)=>v<0||v>=d[a]!))continue;
        const key=q[0]!+d[0]*(q[1]!+d[1]*q[2]!);
        if(distance[key]!>distance[t]!+1){distance[key]=distance[t]!+1;queue.push(key);}
      }
    }
    const header=new Uint32Array(16);header[2]=layout.fineTiles.length;
    header.set([Math.min(layout.fineTiles.length,this.dispatchX),Math.ceil(layout.fineTiles.length/this.dispatchX),1],8);
    this.device.queue.writeBuffer(this.support,n*16,header);
    this.device.queue.writeBuffer(this.certifiedDispatch,0,header.subarray(4,12));
    this.device.queue.writeBuffer(this.support,(4*n+16)*4,distance);
    this.device.queue.writeBuffer(this.support,(6*n+16)*4,layout.fineTiles);
    const seamLists=[layout.fineTiles,layout.transitionTiles,layout.coarseTiles].map(tiles=>
      [...tiles].filter(tile=>(layout.stencils[2*tile]!>>>27)!==(layout.stencils[2*tile+1]!>>>27)));
    this.seamCounts=seamLists.map(list=>list.length);
    this.device.queue.writeBuffer(this.support,(7*n+16)*4,new Uint32Array([...this.seamCounts,0,...seamLists.flat()]));
    this.currentLayout=layout;
  }

  /** Hold ownership and its stencil/support storage through every asynchronous
   * pressure receipt. Edits can only publish after the frame finishes. */
  acquireFrame():()=>void {
    if(this.frameHeld)throw new Error("Ownership already belongs to an active frame");
    this.frameHeld=true;let released=false;
    return ()=>{if(!released){released=true;this.frameHeld=false;}};
  }

  /** One workgroup covers 64 h owners, eight 2h tiles, or 64 4h tiles. */
  dispatch(pass: GPUComputePassEncoder, pipelines: readonly GPUComputePipeline[], indirect?: GPUBuffer): void {
    for (const tier of [0, 1, 2] as const) this.dispatchTier(pass, pipelines[tier]!, tier, indirect);
  }
  /** Fine work is split by a conservative whole-characteristic certificate.
   * Both lists use the same state, ownership generation and numerical stage. */
  dispatchCertified(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[],regular:GPUComputePipeline):void{
    pass.setPipeline(regular);pass.dispatchWorkgroupsIndirect(this.certifiedDispatch,0);
    pass.setPipeline(pipelines[0]!);pass.dispatchWorkgroupsIndirect(this.certifiedDispatch,16);
    this.dispatchTier(pass,pipelines[1]!,1);this.dispatchTier(pass,pipelines[2]!,2);
  }

  /** Frozen interface work is shared by all pressure sweeps. */
  dispatchSeams(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[]):void{
    for(const tier of [0,1,2] as const){
      const groups=Math.ceil(this.seamCounts[tier]!/(1<<(tier*3)));
      if(!groups)continue;
      pass.setPipeline(pipelines[tier]!);
      pass.dispatchWorkgroups(Math.min(groups,this.dispatchX),Math.ceil(groups/this.dispatchX));
    }
  }

  dispatchRegular(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[]):void{
    const counts=[this.layout.fineTiles.length,this.layout.transitionTiles.length,this.layout.coarseTiles.length];
    for(const tier of [0,1,2] as const)if(counts[tier]!>this.seamCounts[tier]!)this.dispatchTier(pass,pipelines[tier]!,tier);
  }

  /** Face kernels with large shared samplers can compile once for all widths.
   * The same tier worklists are packed into a single owner launch. */
  dispatchAll(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline): void {
    const groups = Math.ceil(this.layout.cellCount / 64);
    if (!groups) return;
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(Math.min(groups, this.dispatchX), Math.ceil(groups / this.dispatchX));
  }

  /** Pressure colours visit one tier; do not launch idle work for the others. */
  dispatchTier(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, tier: 0 | 1 | 2, indirect?: GPUBuffer): void {
    const count = tier === 0 ? this.layout.fineTiles.length * 64
      : tier === 1 ? this.layout.transitionTiles.length * 8 : this.layout.coarseTiles.length;
    if (!count) return;
    const groups = Math.ceil(count / 64);
    pass.setPipeline(pipeline);
    if(indirect)pass.dispatchWorkgroupsIndirect(indirect,tier*12);
    else pass.dispatchWorkgroups(Math.min(groups, this.dispatchX), Math.ceil(groups / this.dispatchX));
  }

  destroy(): void { this.topology.destroy(); this.counts.destroy(); this.support.destroy();this.certifiedDispatch.destroy(); }
}
