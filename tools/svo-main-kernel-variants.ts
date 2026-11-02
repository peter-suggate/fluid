/** Benchmark-only candidates. No source or representation changes in a frame. */
export interface MainKernelVariant {
  name: string;
  section: number;
  hypothesis: string;
  entries?: readonly string[];
  transform?: (code: string) => string;
  omit?: readonly string[];
  rangeShape?: readonly [number, number];
  tiling?: boolean;
  inlineEdges?: boolean;
  receipts?: "off" | "sync";
  fusedTileClassification?: boolean;
}
function replace(code: string, from: string, to: string): string {
  if (!code.includes(from)) throw new Error(`Experiment source no longer matches: ${from.slice(0, 100)}`);
  return code.replaceAll(from, to);
}
const range = ["rasterAoRangeMain"];
const edges = ["rasterLightEdgesMain"];
const exact = ["rasterExactEdgesMain"];
const position = "var<workgroup> rasterAoPositions:array<vec4f,256>;";
const normal = "var<workgroup> rasterAoNormals:array<vec3f,256>;";
const neighbors = "let neighbours=array<u32,3>(lane^1u,lane^16u,lane^17u);";
export const mainKernelVariants: MainKernelVariant[] = [
  { name: "layout-aos", section: 1, hypothesis: "Interleave the position and normal used by each AO neighbor instead of two separate shared arrays.", entries: range,
    transform: code => replace(replace(code, position, "struct StudyAoReceiver{position:vec4f,normal:vec3f}\nvar<workgroup> studyAoReceivers:array<StudyAoReceiver,256>;"), normal, "")
      .replaceAll("rasterAoPositions[lane]", "studyAoReceivers[lane].position").replaceAll("rasterAoNormals[lane]", "studyAoReceivers[lane].normal")
      .replaceAll("rasterAoPositions[neighbours[i]]", "studyAoReceivers[neighbours[i]].position").replaceAll("rasterAoNormals[neighbours[i]]", "studyAoReceivers[neighbours[i]].normal") },
  { name: "layout-quad-tiled", section: 1, hypothesis: "Store each reused 2x2 receiver quad in four adjacent shared slots.", entries: range,
    transform: code => replace(code, position, `${position}\nfn studyAoSlot(lane:u32)->u32{return (lane&1u)|((lane&16u)>>3u)|((lane&14u)<<1u)|(lane&224u);}`)
      .replaceAll("rasterAoPositions[lane]", "rasterAoPositions[studyAoSlot(lane)]").replaceAll("rasterAoNormals[lane]", "rasterAoNormals[studyAoSlot(lane)]")
      .replaceAll("rasterAoPositions[neighbours[i]]", "rasterAoPositions[studyAoSlot(neighbours[i])]").replaceAll("rasterAoNormals[neighbours[i]]", "rasterAoNormals[studyAoSlot(neighbours[i])]") },
  { name: "bandwidth-late-shadow", section: 2, hypothesis: "Skip the four-byte shadow texture read for rejected zero-weight reconstruction taps.", entries: edges,
    transform: code => replace(code, "total+=vec2f(v.x,textureLoad(rasterShadowVisibility,c,0).x)*w;", "if(w>0.0){total+=vec2f(v.x,textureLoad(rasterShadowVisibility,c,0).x)*w;}") },
  { name: "bandwidth-representative", section: 2, hypothesis: "Resolve representative pixels from AO/shadow/identity before loading the 16-byte geometry record.", entries: edges,
    transform: code => {
      const start = code.indexOf("fn rasterLightEdgeAt("); const end = code.indexOf("fn rasterInlineEdgeAt(", start);
      const body = code.slice(start, end);
      const load = "  let g=rasterVisibilityGeometryAt(coord);if(!(g.w<DRY_MISS)||dot(g.xyz,g.xyz)<.1){return;}\n";
      const modified = replace(body, load, "").replace("  rasterPixel=vec2f(coord)+.5;", load + "  rasterPixel=vec2f(coord)+.5;");
      return replace(code, body, modified);
    } },
  { name: "active-tiles", section: 3, hypothesis: "Compact occupied lighting tiles and dispatch indirectly; include classification and argument production.", tiling: true },
  { name: "divergence-row-coherent", section: 4, hypothesis: "Adjacent lanes process adjacent full-resolution pixels; four tile quadrants replace per-lane 2x2 serial work.", entries: edges,
    transform: code => replace(code, `if(all(id<textureDimensions(rasterContact))){
    let stride=i32(rasterVisibilityStride());let origin=vec2i(id)*stride;
    for(var y=0;y<stride;y++){for(var x=0;x<stride;x++){rasterLightEdgeAt(origin+vec2i(x,y));}}
  }`, `if(all(id-local.xy<textureDimensions(rasterContact))){
    let origin=vec2i(id-local.xy)*2+vec2i(local.xy);
    for(var y=0;y<2;y++){for(var x=0;x<2;x++){rasterLightEdgeAt(origin+vec2i(x,y)*8);}}
  }`) },
  { name: "fuse-exact-contact", section: 5, hypothesis: "Fuse the exact receiver AO producer into its visibility consumer, removing a dispatch and queue AO write/read.", entries: exact, omit: ["Exact receiver contact AO"],
    transform: code => replace(code, "resolved.x=bitcast<f32>(atomicLoad(&rasterExactRequests[5u+index*2u]));", "resolved.x=min(rasterHorizonVisibility(rasterPixel,position,normal),rasterCoarseVisibility(position,normal));") },
  { name: "fuse-range-classification", section: 5, hypothesis: "Build occupied-tile requests inside AO range production, reusing its representative geometry load; retain the global finalizer.", entries: range, tiling: true, fusedTileClassification: true,
    transform: code => {
      code = replace(code, position, position + "\nvar<workgroup> studyOccupiedTiles:array<atomic<u32>,4>;");
      code = replace(code, "  var position=vec3f(0);var normal=vec3f(0);", `  let studyTile=(lane%16u)/8u+2u*((lane/16u)/8u);
  let studyOrigin=id.xy*2u;
  for(var y=0u;y<2u;y++){for(var x=0u;x<2u;x++){
    let c=studyOrigin+vec2u(x,y);if(any(c>=vec2u(dims))){continue;}
    var geometry=g;if(x!=1u||y!=1u){geometry=rasterVisibilityGeometryAt(vec2i(c));}
    if(geometry.w<DRY_MISS&&dot(geometry.xyz,geometry.xyz)>=.1){atomicStore(&studyOccupiedTiles[studyTile],1u);}
  }}
  var position=vec3f(0);var normal=vec3f(0);`);
      return replace(code, "    atomicMax(&rasterAoRangeOutput[0],atomicLoad(&rasterAoRangeMinimum));", `    let columns=(u32(dims.x)+15u)/16u;
    for(var tile=0u;tile<4u;tile++){
      if(atomicLoad(&studyOccupiedTiles[tile])!=0u){
        let index=atomicAdd(&rasterTileRequestsOutput[0],1u);
        let c=id.xy/8u+vec2u(tile%2u,tile/2u);
        atomicStore(&rasterTileRequestsOutput[4u+index],c.y*columns+c.x);
      }
    }
    atomicMax(&rasterAoRangeOutput[0],atomicLoad(&rasterAoRangeMinimum));`);
    } },
  { name: "split-inline-control", section: 6, hypothesis: "Compare production common/exception kernels against inline exact work, including queue construction and all consumers.", inlineEdges: true },
  { name: "atomics-subgroup-range", section: 7, hypothesis: "Subgroup-reduce AO min/max so only one lane per subgroup updates each workgroup atomic.", entries: range,
    transform: code => {
      code = replace(code, "  var importance=0.0;", "  var importance=0.0;var studyMin=0u;var studyMax=0u;");
      code = replace(code, "atomicMax(&rasterAoRangeMinimum,~bits);atomicMax(&rasterAoRangeMaximum,bits);", "studyMin=~bits;studyMax=bits;");
      code = replace(code, "  if(inBounds){textureStore(rasterAoImportanceOutput", `  let reducedMin=subgroupMax(studyMin);let reducedMax=subgroupMax(studyMax);
  if(subgroupElect()){atomicMax(&rasterAoRangeMinimum,reducedMin);atomicMax(&rasterAoRangeMaximum,reducedMax);}
  if(inBounds){textureStore(rasterAoImportanceOutput`);
      return code.includes("enable subgroups;") ? code : "enable subgroups;\n" + code;
    } },
  { name: "atomics-global-per-receiver", section: 7, hypothesis: "Compare the shipping per-tile reservation with one global queue reservation per exceptional receiver.", entries: edges,
    transform: code => {
      const start = code.indexOf("fn rasterBeginExactTile(");
      const end = code.indexOf("\n}", code.indexOf("fn rasterFlushExactTile(", start)) + 2;
      if (start < 0 || end < start) throw new Error("Exact queue source no longer matches");
      return code.slice(0, start) + `fn rasterBeginExactTile(lane:u32){}
fn rasterAppendExactTile(coordinate:u32){
  let index=atomicAdd(&rasterExactRequests[0],1u);
  atomicStore(&rasterExactRequests[4u+index*2u],coordinate);
}
fn rasterFlushExactTile(lane:u32){}` + code.slice(end);
    } },
  { name: "reuse-reload-neighbors", section: 8, hypothesis: "Compare the shared receiver tile against three extra geometry fetches and reconstruction, removing its shared arrays and barrier.", entries: range,
    transform: code => {
      code = replace(replace(code, position, ""), normal, "");
      code = replace(code, "  if(lane==0u){atomicStore(&rasterAoRangeMinimum,0u);atomicStore(&rasterAoRangeMaximum,0u);}", "");
      code = replace(code, `  rasterAoPositions[lane]=vec4f(position,select(0.0,1.0,valid));
  rasterAoNormals[lane]=normal;
  workgroupBarrier();`, "");
      code = replace(code, "let other=rasterAoPositions[neighbours[i]];", `let otherId=vec2i(id.xy)-vec2i(i32(lane%16u),i32(lane/16u))+vec2i(i32(neighbours[i]%16u),i32(neighbours[i]/16u));
      let otherCoord=min(otherId*stride+vec2i(stride/2),dims-1);
      let otherGeometry=drySplitGeometryAt(otherCoord);
      let otherValid=all(otherId<(dims+stride-1)/stride)&&otherGeometry.w>0.0&&otherGeometry.w<DRY_MISS&&dot(otherGeometry.xyz,otherGeometry.xyz)>=.1;
      let other=vec4f(rasterWorldAt(vec2f(otherCoord)+.5,otherGeometry.w),select(0.0,1.0,otherValid));`);
      return replace(code, "rasterAoNormals[neighbours[i]]", "normalize(otherGeometry.xyz)");
    } },
  ...([[8, 8], [32, 8], [8, 32]] as const).map(([x, y]): MainKernelVariant => ({
    name: `occupancy-${x}x${y}`, section: 9, hypothesis: `AO range workgroup ${x}x${y}; preserve the exact same 2x2 neighborhood.`, entries: range, rangeShape: [x, y],
    transform: code => replace(replace(replace(replace(code, "@compute @workgroup_size(16,16) fn rasterAoRangeMain", `@compute @workgroup_size(${x},${y}) fn rasterAoRangeMain`), neighbors,
      `let neighbours=array<u32,3>(lane^1u,lane^${x}u,lane^${x + 1}u);`), position, `var<workgroup> rasterAoPositions:array<vec4f,${x * y}>;`), normal, `var<workgroup> rasterAoNormals:array<vec3f,${x * y}>;`),
  })),
  ...[32,128,256].map((size): MainKernelVariant => ({
    name: `occupancy-exact-${size}`, section: 9, hypothesis: `Dispatch exact AO and visibility with ${size} lanes, including GPU argument generation.`,
    entries: ["rasterExactContactMain", "rasterExactEdgesMain", "rasterExactFinalize"],
    transform: code => replace(replace(replace(replace(code,
      "@compute @workgroup_size(64) fn rasterExactContactMain", `@compute @workgroup_size(${size}) fn rasterExactContactMain`),
      "@compute @workgroup_size(64) fn rasterExactEdgesMain", `@compute @workgroup_size(${size}) fn rasterExactEdgesMain`),
      "(group.x+group.y*65535u)*64u+lane", `(group.x+group.y*65535u)*${size}u+lane`),
      "(atomicLoad(&rasterExactRequests[0])+63u)/64u", `(atomicLoad(&rasterExactRequests[0])+${size-1}u)/${size}u`),
  })),
  { name: "coordination-no-receipts", section: 10, hypothesis: "Suppress ready-state mesh telemetry polling while GPU revisions and culling continue; compare against the production asynchronous 250ms cadence.", receipts: "off" },
  { name: "coordination-sync-receipt", section: 10, hypothesis: "Measure the cost of making a mesh diagnostic receipt a current-frame CPU dependency, with identical rendered work.", receipts: "sync" },
];

const coherentRows = mainKernelVariants.find(v => v.name === "divergence-row-coherent")!;
const exact128 = mainKernelVariants.find(v => v.name === "occupancy-exact-128")!;
mainKernelVariants.push({
  name: "combined-row-exact-128", section: 9,
  hypothesis: "Combine contiguous edge scheduling with 128-lane exact consumers after measuring each independently.",
  entries: [...coherentRows.entries!, ...exact128.entries!],
  transform: code => exact128.transform!(coherentRows.transform!(code)),
});
