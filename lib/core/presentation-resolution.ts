/** Water retains 136 bytes per render pixel, plus 16 with world reflections. Limit optional extra
 * supersampling to eight megapixels; never shrink an existing native target.
 * The returned scale, rather than the requested UI value, is reported by the
 * renderer and used for every pass and pointer-coordinate transform. */
export function boundedPresentationScale(width: number, height: number, requested: number, textureLimit: number): number {
  const nativePixels = Math.max(1, width * height);
  const extraPixelLimit = Math.max(nativePixels, 8 * 1024 * 1024);
  return Math.min(Number.isFinite(requested) ? Math.max(.35, requested) : 1,
    2, Math.sqrt(extraPixelLimit / nativePixels), textureLimit / width, textureLimit / height);
}
