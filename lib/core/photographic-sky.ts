import capture from "./imported-assets/metadata/kloofendal-sky.json";

const width = capture.width;
const height = capture.height;
export const SKY_CAPTURE_BYTES = width * height / 2;
export const SKY_CAPTURE_FIELD = `skyBlocks:array<vec4u,${SKY_CAPTURE_BYTES / 16}>,`;
let hemisphere: Uint32Array<ArrayBuffer> | undefined;

/** Native little-endian endpoint pairs and two-bit selectors, one word each.
 * Uploaded once per device buffer. This section follows its mutable lighting header. */
export function skyCaptureWords(): Uint32Array<ArrayBuffer> {
  if (!hemisphere) {
    const binary = atob(capture.blocks);
    const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
    hemisphere = new Uint32Array(bytes.buffer);
  }
  return hemisphere;
}

type RGB = [number, number, number];
const mean = capture.meanLinear as RGB;

function skyTexel(x: number, y: number): RGB {
  x &= width - 1;
  y = Math.max(0, Math.min(height - 1, y));
  const block = (x >> 2) + (width / 4) * (y >> 2), words = skyCaptureWords();
  const ends = words[2 * block]!, indices = words[2 * block + 1]!;
  const t = ((indices >>> (2 * ((x & 3) + 4 * (y & 3)))) & 3) / 3;
  return [11, 5, 0].map((shift, channel) => {
    const mask = channel === 1 ? 63 : 31;
    const a = ((ends >>> shift) & mask) / mask, b = ((ends >>> (shift + 16)) & mask) / mask;
    return 4 * (a + t * (b-a)) ** 2;
  }) as RGB;
}

export function capturedSkyRadiance(direction: readonly number[], key: readonly number[], roughness: number): RGB {
  const x = (Math.atan2(direction[2]!, direction[0]!) - Math.atan2(key[2]!, key[0]!)) * width / (2*Math.PI);
  const t = Math.asin(Math.max(0, Math.min(1, direction[1]!))) * 2 / Math.PI;
  const y = height * (1.8 * t) / (1 + .8 * t) - .5;
  const i = Math.floor(x), j = Math.floor(y), u = x-i, v = y-j;
  const a = skyTexel(i,j), b = skyTexel(i+1,j), c = skyTexel(i,j+1), d = skyTexel(i+1,j+1);
  return mean.map((average, k) => {
    const value = (a[k]!*(1-u)+b[k]!*u)*(1-v)+(c[k]!*(1-u)+d[k]!*u)*v;
    return value + (average-value)*roughness*roughness;
  }) as RGB;
}

/** Uses the same meter-independent direction and filtered linear RGB in glass,
 * liquid and sky. The authored directional light supplies the solar disc. */
export function capturedSkyWGSL(uniform: string): string {
  return /* wgsl */ `
fn captureEndpoint(word:u32)->vec3f {
  return vec3f(f32((word>>11u)&31u)/31.0, f32((word>>5u)&63u)/63.0, f32(word&31u)/31.0);
}
fn captureTexel(p:vec2i)->vec3f {
  let q=vec2u(u32(p.x)&${width-1}u,u32(clamp(p.y,0,${height-1})));
  let block=(q.x>>2u)+${width/4}u*(q.y>>2u);
  let pair=${uniform}.skyBlocks[block>>1u];
  let lane=2u*(block&1u);
  let ends=pair[lane];
  let index=(pair[lane+1u]>>(2u*((q.x&3u)+4u*(q.y&3u))))&3u;
  let rgb=mix(captureEndpoint(ends),captureEndpoint(ends>>16u),f32(index)/3.0);
  return 4.0*rgb*rgb;
}
fn capturedSky(direction:vec3f,key:vec3f,roughness:f32)->vec3f {
  let azimuth=select(0.0,atan2(direction.z,direction.x),dot(direction.xz,direction.xz)>0.0);
  let solarAzimuth=select(0.0,atan2(key.z,key.x),dot(key.xz,key.xz)>0.0);
  let t=asin(clamp(direction.y,0.0,1.0)) * ${2/Math.PI};
  let uv=vec2f((azimuth-solarAzimuth)*${width/(2*Math.PI)},
    ${height}.0*(1.8*t)/(1.0+.8*t)-.5);
  let p=vec2i(floor(uv));let f=fract(uv);
  let rgb=mix(mix(captureTexel(p),captureTexel(p+vec2i(1,0)),f.x),
              mix(captureTexel(p+vec2i(0,1)),captureTexel(p+1),f.x),f.y);
  return mix(rgb,vec3f(${mean.join(",")}),roughness*roughness);
}`;
}
