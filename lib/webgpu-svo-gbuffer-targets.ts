import {
  SVO_GBUFFER_BYTES_PER_PIXEL,
  SVO_GBUFFER_COLOR_ATTACHMENT_COUNT,
  SVO_GBUFFER_COLOR_BYTES_PER_SAMPLE,
  SVO_GBUFFER_LAYOUT,
} from "./svo-gbuffer";

export const SVO_GBUFFER_RENDER_TARGET_CONTRACT = Object.freeze({
  colorAttachmentCount: SVO_GBUFFER_COLOR_ATTACHMENT_COUNT,
  colorBytesPerSample: SVO_GBUFFER_COLOR_BYTES_PER_SAMPLE,
  bytesPerPixelIncludingDepth: SVO_GBUFFER_BYTES_PER_PIXEL,
  externalRadianceDepthFormat: SVO_GBUFFER_LAYOUT.radianceDepth.format,
  packedSurfaceFormat: SVO_GBUFFER_LAYOUT.packedSurface.format,
  identityMediaFormat: SVO_GBUFFER_LAYOUT.identityMedia.format,
  hardwareDepthFormat: SVO_GBUFFER_LAYOUT.hardwareDepth.format,
  depthClearValue: 0,
  depthCompare: "greater" as GPUCompareFunction,
} as const);

export interface SparseVoxelGBufferTextures {
  readonly width: number;
  readonly height: number;
  /** Location 0 remains the water compositor's existing HDR input. */
  readonly radianceDepthOwnership: "external-water-compositor-target";
  readonly packedSurface: GPUTexture;
  readonly identityMedia: GPUTexture;
  readonly hardwareDepth: GPUTexture;
}

export interface SparseVoxelGBufferViews {
  readonly packedSurface: GPUTextureView;
  readonly identityMedia: GPUTextureView;
  readonly hardwareDepth: GPUTextureView;
}

/**
 * Owns the auxiliary compact G-buffer attachments. Location 0 intentionally
 * remains external so the established water compositor samples the exact HDR
 * result without a copy or compatibility conversion.
 */
export class SparseVoxelGBufferTargetArena {
  private packedSurface?: GPUTexture;
  private identityMedia?: GPUTexture;
  private hardwareDepth?: GPUTexture;
  private packedSurfaceView?: GPUTextureView;
  private identityMediaView?: GPUTextureView;
  private hardwareDepthView?: GPUTextureView;
  private width = 0;
  private height = 0;

  constructor(private readonly device: GPUDevice) {}

  ensureSize(width: number, height: number): boolean {
    if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1) {
      throw new RangeError("Sparse voxel G-buffer dimensions must be positive safe integers");
    }
    if (this.packedSurface && this.identityMedia && this.hardwareDepth && this.width === width && this.height === height) return false;
    this.releaseTextures();
    const colorUsage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
    this.packedSurface = this.device.createTexture({
      label: "Sparse voxel G-buffer packed surface",
      size: [width, height],
      format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.packedSurfaceFormat,
      usage: colorUsage,
    });
    this.identityMedia = this.device.createTexture({
      label: "Sparse voxel G-buffer identity and media",
      size: [width, height],
      format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.identityMediaFormat,
      usage: colorUsage,
    });
    this.hardwareDepth = this.device.createTexture({
      label: "Sparse voxel G-buffer reversed-Z depth",
      size: [width, height],
      format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.packedSurfaceView = this.packedSurface.createView();
    this.identityMediaView = this.identityMedia.createView();
    this.hardwareDepthView = this.hardwareDepth.createView();
    this.width = width;
    this.height = height;
    return true;
  }

  get textures(): SparseVoxelGBufferTextures | undefined {
    if (!this.packedSurface || !this.identityMedia || !this.hardwareDepth) return undefined;
    return {
      width: this.width,
      height: this.height,
      radianceDepthOwnership: "external-water-compositor-target",
      packedSurface: this.packedSurface,
      identityMedia: this.identityMedia,
      hardwareDepth: this.hardwareDepth,
    };
  }

  get views(): SparseVoxelGBufferViews | undefined {
    if (!this.packedSurfaceView || !this.identityMediaView || !this.hardwareDepthView) return undefined;
    return {
      packedSurface: this.packedSurfaceView,
      identityMedia: this.identityMediaView,
      hardwareDepth: this.hardwareDepthView,
    };
  }

  destroy(): void {
    this.releaseTextures();
  }

  private releaseTextures(): void {
    this.packedSurface?.destroy();
    this.identityMedia?.destroy();
    this.hardwareDepth?.destroy();
    this.packedSurface = undefined;
    this.identityMedia = undefined;
    this.hardwareDepth = undefined;
    this.packedSurfaceView = undefined;
    this.identityMediaView = undefined;
    this.hardwareDepthView = undefined;
    this.width = 0;
    this.height = 0;
  }
}
