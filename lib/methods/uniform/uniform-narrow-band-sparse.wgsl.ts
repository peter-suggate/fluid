import type { NarrowBandSurfaceSamples } from "./uniform-narrow-band-sites";
/** A full h cell seeds eight samples. Requiring a threshold of *other*
 * samples in the 3x3x3 h cells around a sample's cell (about a two-cell
 * radius's volume) is deliberately much less dense than a one-layer sheet
 * at 8; 16 or 32 also retire small clumps of spray. Cells
 * are h cells, so the criterion follows anisotropic resolution.
 * The continuous isolation age travels with the complete particle record.
 * Never use reconstructed phi as a veto: an orphan can create its own zero.
 * The threshold is the runtime sparseNeighbours (params.live.y); 0 disables.
 * samples (surfaceSamples, uniform-narrow-band-sites.ts): past eight a cell
 * holds S units in samples of 1 (surface layer) to S/8 (deep band) units, so
 * the count of neighbours that means the same liquid is the threshold times
 * S/(8m) for a sample of m units; nbSparseAge takes the sample's mass. At 8
 * the text the solver always compiled. */
export const narrowBandSparseWGSLFor=(samples:NarrowBandSurfaceSamples=8)=>/* wgsl */`
fn nbSparseAge(i:u32,p:vec3f,age:f32${samples===8?"":",mass:f32"})->f32{
 let lifetime=params.activity.y;let threshold=${samples===8?"params.live.y":`select(0u,max(1u,u32(round(f32(params.live.y)*${(samples/8).toFixed(1)}/mass))),params.live.y!=0u)`};
 if(lifetime<=0.0||threshold==0u){return 0.0;}
 let cell=vec3i(floor(p));let own=nbCellRun(cell);
 // Every point in the same h cell is less than two cells away. Exclude self.
 var neighbours=own.y-own.x-1u;
 if(neighbours>=threshold){return 0.0;}
 // The other samples of the 3x3x3 cells around the sample's cell, by
 // population alone (no positions): x rows of the 3x3 (y,z) neighbourhood,
 // its own row first, each split where it crosses a tile (a piece within
 // one tile is one run). The block's 27 cells are about the two-cell
 // sphere's volume (33.5); counting positions within the sphere took half
 // of update for the samples near the threshold. Reads use the immutable
 // position bank's bins; the following compaction changes them.
 var block=0u;
 for(var k=0u;k<9u;k++){
  let r=(k+4u)%9u;let c=vec3i(cell.x-1,cell.y+i32(r%3u)-1,cell.z+i32(r/3u)-1);
  let stop=min(cell.x+1,c.x|3);
  var run=nbRun(c,vec3i(stop,c.y,c.z));block+=run.y-run.x;
  if(stop<cell.x+1){run=nbRun(vec3i(stop+1,c.y,c.z),vec3i(cell.x+1,c.y,c.z));block+=run.y-run.x;}
  // The block holds the sample itself.
  if(block>threshold){return 0.0;}
 }
 return min(lifetime,age+max(params.hDt.w,0.0));
}
fn nbSparseExpired(age:f32)->bool{return params.activity.y>0.0&&params.live.y!=0u&&age>=params.activity.y;}
`;
export const narrowBandSparseWGSL=narrowBandSparseWGSLFor();
