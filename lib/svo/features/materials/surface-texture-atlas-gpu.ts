import type { SurfaceTextureAtlas } from "../../../core/imported-assets/surface-texture-atlas";
/** A single sampled binding fits alongside World water reflections on the
 * minimum 16-texture WebGPU device. No source geometry reaches this resource. */
export class SurfaceTextureAtlasGpu {
  private texture: GPUTexture;
  private publication?: SurfaceTextureAtlas;
  view: GPUTextureView;
  constructor(private device: GPUDevice) {
    this.texture=device.createTexture({label:"Empty voxel surface textures",size:[1,1,1],format:"rgba8unorm-srgb",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    this.view=this.texture.createView({dimension:"2d-array"});
  }
  publish(atlas?:SurfaceTextureAtlas):void {
    if(atlas===this.publication) return;
    const width=atlas?.width??1,height=atlas?.height??1,layers=Math.max(1,atlas?.layers??1),mipLevelCount=atlas?.mipLevels??1;
    let bytes=0; for(let mip=0;mip<mipLevelCount;mip++)bytes+=Math.max(1,width>>mip)*Math.max(1,height>>mip)*4*layers;
    if(width>this.device.limits.maxTextureDimension2D||height>this.device.limits.maxTextureDimension2D||layers>this.device.limits.maxTextureArrayLayers||bytes>256*1024*1024)
      throw new RangeError("Voxel surface textures exceed the device or 256 MiB atlas budget");
    const next=this.device.createTexture({label:"Voxel surface image attributes",size:[width,height,layers],mipLevelCount,format:"rgba8unorm-srgb",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    let layerBase=0;
    for(const mapping of atlas?.mappings??[]) {
      const h=mapping.header;let offset=0;
      for(let layer=0;layer<h.textureLayerCount;layer++) for(let mip=0;mip<h.mipLevelCount;mip++) {
        const w=Math.max(1,h.textureWidth>>mip),height=Math.max(1,h.textureHeight>>mip),size=w*height*4;
        this.device.queue.writeTexture({texture:next,mipLevel:mip,origin:[0,0,layerBase+layer]},mapping.textureBytes.subarray(offset,offset+size),{bytesPerRow:w*4,rowsPerImage:height},[w,height,1]);
        offset+=size;
      }
      layerBase+=h.textureLayerCount;
    }
    const old=this.texture;this.texture=next;this.view=next.createView({dimension:"2d-array"});this.publication=atlas;
    void this.device.queue.onSubmittedWorkDone().then(()=>old.destroy(),()=>old.destroy());
  }
  destroy():void { this.texture.destroy(); }
}
