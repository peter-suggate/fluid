/** View-uniform equality for static, body-free dry receiver visibility.
 *
 * Water shadows must be compiled out before using this key. These four words
 * then belong solely to simulation presentation: clock, pressure-cell overlay
 * scale, last fluid step and fluid speed. Neither opaque primary geometry nor
 * the raster AO / light-visibility producers read them. Keep every other word,
 * including camera motion/LOD, projection, diagnostics and terrain publication.
 * Animated scene geometry and materials are invalidated by their independent
 * scene/source/mesh/parameter revisions; this comparison never replaces those.
 */
export function rasterVisibilityViewsEqual(previous:Float32Array,current:Float32Array):boolean {
  if(previous.length!==current.length)return false;
  for(let i=0;i<current.length;i++){
    if(i===2||i===19||i===29||i===30)continue;
    if(!Object.is(previous[i],current[i]))return false;
  }
  return true;
}
