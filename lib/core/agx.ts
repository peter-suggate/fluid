/** AgX base contrast in linear Rec.709, matching the Blender preview's view
 * transform. Matrices and polynomial follow Three.js / Filament's AgX fit:
 * https://github.com/mrdoob/three.js/blob/dev/src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js
 * Three.js: Copyright © 2010-2026 three.js authors, MIT license.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions: The above copyright notice and this
 * permission notice shall be included in all copies or substantial portions of
 * the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
 * EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
type Matrix = readonly (readonly number[])[];
const to2020:Matrix=[[.6274,.3293,.0433],[.0691,.9195,.0113],[.0164,.0880,.8956]];
const inset:Matrix=[[.8566271533,.0951212405,.0482516061],[.1373189729,.7612419906,.1014390365],[.1118982130,.0767994186,.8113023684]];
const outset:Matrix=[[1.1271005818,-.1106066431,-.0164939387],[-.1413297635,1.1578237022,-.0164939387],[-.1413297635,-.1106066431,1.2519364066]];
const to709:Matrix=[[1.6605,-.5876,-.0728],[-.1246,1.1329,-.0083],[-.0182,-.1006,1.1187]];
const multiply=(m:Matrix,v:readonly number[])=>m.map(row=>row.reduce((n,c,i)=>n+c*v[i]!,0));
const clamp=(x:number)=>Math.min(1,Math.max(0,x));
export function agxToneMap(exposed:readonly number[]):[number,number,number]{
  const log=multiply(inset,multiply(to2020,exposed)).map(v=>clamp((Math.log2(Math.max(1e-10,v))+12.47393)/16.499999));
  const contrast=log.map(x=>((((((15.5*x-40.14)*x+31.96)*x-6.868)*x+.4298)*x+.1191)*x-.00232));
  return multiply(to709,multiply(outset,contrast).map(v=>Math.max(0,v)**2.2)).map(clamp) as [number,number,number];
}
const matrix=(m:Matrix)=>`mat3x3f(${[0,1,2].map(c=>`vec3f(${m.map(row=>row[c]).join(",")})`).join(",")})`;
export const agxWGSL=/* wgsl */`
fn agxToneMap(exposed:vec3f)->vec3f{
  let inset=${matrix(inset)}*(${matrix(to2020)}*exposed);
  let x=clamp((log2(max(inset,vec3f(1e-10)))+vec3f(12.47393))/16.499999,vec3f(0),vec3f(1));
  let contrast=((((((15.5*x-vec3f(40.14))*x+vec3f(31.96))*x-vec3f(6.868))*x+vec3f(.4298))*x+vec3f(.1191))*x-vec3f(.00232));
  let linear=pow(max(${matrix(outset)}*contrast,vec3f(0)),vec3f(2.2));
  return clamp(${matrix(to709)}*linear,vec3f(0),vec3f(1));
}
`;
