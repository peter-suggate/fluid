/** Conservative cloth-cloth rejection; the caller supplies current round geometry and DAT bounds. */
export const CLOTH_SEPARATION_CULL_WGSL = /* wgsl */`
// Along a fixed axis, each primitive's support gap is concave in the shared path
// parameter. Its endpoint minimum therefore bounds the whole swept interval.
// Current predictions, caps and path strays are always read; no old force or distance is reused.
fn separatedClothPair(ids: vec4u, nA: u32) -> bool {
  var x: array<vec3f,4>; var w: array<vec3f,4>;
  var centreA = vec3f(0.0); var centreB = vec3f(0.0);
  let origin = xd(ids[0]); var magnitude = 0.0;
  var capA = 0.0; var capB = 0.0; var strayA = 0.0; var strayB = 0.0;
  for (var k=0u; k<4u; k++) {
    let p = xd(ids[k]); x[k] = p-origin; w[k] = predOf(ids[k]);
    magnitude = max(magnitude, absMax(p)+absMax(w[k]));
    if (k<nA) { centreA += x[k]; capA=max(capA,envOf(ids[k])); strayA=max(strayA,pathStray(ids[k])); }
    else { centreB += x[k]; capB=max(capB,envOf(ids[k])); strayB=max(strayB,pathStray(ids[k])); }
  }
  let axis = centreA/f32(nA)-centreB/f32(4u-nA);
  let len = length(axis); if (!(len>1e-12)) { return false; }
  let n = axis/len;
  var a0=3.4e38; var a1=3.4e38; var b0=-3.4e38; var b1=-3.4e38;
  for (var k=0u; k<4u; k++) {
    let p0=dot(n,x[k]); let p1=dot(n,x[k]+w[k]-w[0]);
    if(k<nA) { a0=min(a0,p0); a1=min(a1,p1); } else { b0=max(b0,p0); b1=max(b1,p1); }
  }
  // Extra rounding headroom covers normalized support arithmetic in addition to DAT's guard.
  let guard=8.0*datRound(magnitude);
  return a0-b0>P.r+guard && min(a0-b0,a1-b1)>datCaps(capA,capB)+strayA+strayB+guard;
}
`;
