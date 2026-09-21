import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { UNIFORM_PAGE_DOMAIN_BASE, type UniformPageDomain } from "./uniform-page-domain";

/** The accepted GPU catalogue is the authority for both dispatch and the pages
 * overlay. No cached host page count participates in publication or launch. */
export class UniformPageDomainPublication {
  readonly dispatch: GPUBuffer;
  readonly view: GPUBuffer;
  private pipeline?: GPUComputePipeline;
  private group?: GPUBindGroup;
  constructor(private readonly device: GPUDevice, private readonly domain: UniformPageDomain,
    private readonly accepted: GPUBuffer) {
    this.dispatch=device.createBuffer({label:"Accepted uniform page dispatch",size:32,
      usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
    this.view=device.createBuffer({label:"Accepted uniform page overlay",size:(8+2*domain.capacity)*4,
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  }
  async initialize(signal?: AbortSignal): Promise<void> {
    const compiler=gpuCompilationManagerFor(this.device);
    const module=compiler.createShaderModule({label:"Publish accepted uniform page domain",code:`
@group(0) @binding(0) var<storage,read_write> accepted:array<u32>;
@group(0) @binding(1) var<storage,read_write> view:array<u32>;
const B:u32=${UNIFORM_PAGE_DOMAIN_BASE}u;
const CAP:u32=${this.domain.capacity}u;
const EDGE:u32=${this.domain.edge}u;
@compute @workgroup_size(1) fn publish(){
 let count=accepted[B+8u];
 // Validation of coordinates/uniqueness belongs to generation acceptance.
 // A corrupt count must never turn into an out-of-bounds indirect traversal.
 let valid=count<=CAP;let n=select(0u,count,valid);
 accepted[B]=n*(EDGE/4u);accepted[B+1u]=EDGE/4u;accepted[B+2u]=EDGE/4u;
 let v=(EDGE+4u)/4u;
 accepted[B+4u]=n*v;accepted[B+5u]=v;accepted[B+6u]=v;
 accepted[B+11u]=0u;
 let grid=(vec3u(accepted[B+12u],accepted[B+13u],accepted[B+14u])+vec3u(EDGE-1u))/EDGE;
 view[0]=EDGE;view[1]=grid.x;view[2]=grid.y;view[3]=grid.z;view[4]=n;
 for(var i=0u;i<n;i++){
  let slot=accepted[B+16u+16u*CAP+i];
  let at=B+16u+16u*slot;
  let q=vec3u(accepted[at],accepted[at+1u],accepted[at+2u]);
  let linear=q.x+grid.x*(q.y+grid.y*q.z);
  view[8u+linear]=1u;view[8u+CAP+i]=linear;
 }
}`});
    this.pipeline=await compiler.compileComputePipeline({label:"Publish accepted uniform page domain",
      layout:"auto",compute:{module,entryPoint:"publish"}},{priority:"visible",signal});
    this.group=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.accepted}},{binding:1,resource:{buffer:this.view}},
    ]});
  }
  encode(encoder: GPUCommandEncoder): void {
    if(!this.pipeline||!this.group)throw new Error("Page domain publication is not initialized");
    // Only compact page metadata is cleared; field arenas are untouched.
    encoder.clearBuffer(this.view);
    const pass=encoder.beginComputePass({label:"Publish accepted uniform page domain"});
    pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.group);pass.dispatchWorkgroups(1);pass.end();
    encoder.copyBufferToBuffer(this.accepted,UNIFORM_PAGE_DOMAIN_BASE*4,this.dispatch,0,32);
  }
  destroy(): void {this.dispatch.destroy();this.view.destroy();}
}
