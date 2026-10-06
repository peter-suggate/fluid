/** Compare the current tile closure with exact owner membership on frozen
 * source-free transport inputs. This is a work-count ceiling, not a timing
 * result: building and traversing finer lists also costs GPU work. */
export function uniformTransportWorkCensus(dims:readonly number[],tiles:Uint32Array,volume:Float32Array,departures:Float32Array){
 const [nx,ny,nz]=dims as [number,number,number],tx=nx/4,ty=ny/4;
 const tileOf=(x:number,y:number,z:number)=>(x>>2)+tx*((y>>2)+ty*(z>>2));
 const ownerOf=(x:number,y:number,z:number)=>{const word=tiles[tileOf(x,y,z)]!;return (word&0x3fffffff)+((word>>>31)?(x&3)+4*((y&3)+4*(z&3)):0);};
 const ownerTiles:number[]=[],ownerWidths:number[]=[],neighbors:number[][]=[],seed:number[]=[];
 const tileNeighbors=Array.from({length:tiles.length},()=>new Set<number>()),tileSeed=new Uint8Array(tiles.length);
 for(let tile=0;tile<tiles.length;tile++){
  const word=tiles[tile]!,width=(word>>>31)?1:4,base=word&0x3fffffff,side=4/width;
  const origin=[4*(tile%tx),4*(Math.floor(tile/tx)%ty),4*Math.floor(tile/(tx*ty))];
  // The current implementation takes the union bounding box, including self.
  const tileLow=[origin[0]!/4,origin[1]!/4,origin[2]!/4],tileHigh=[...tileLow];
  for(let lane=0;lane<side**3;lane++){
   const o=base+lane,x=origin[0]!+width*(lane%side),y=origin[1]!+width*(Math.floor(lane/side)%side),z=origin[2]!+width*Math.floor(lane/(side*side));
   const at=x+nx*(y+ny*z),low=[0,1,2].map(a=>departures[4*at+a]!-.5*width),high=low.map(v=>v+width);
   if(!low.every(Number.isFinite))throw new Error("Nonfinite frozen departure");
   const donors=new Set([o]);
   for(let zz=Math.max(0,Math.floor(low[2]!));zz<=Math.min(nz-1,Math.ceil(high[2]!)-1);zz++)for(let yy=Math.max(0,Math.floor(low[1]!));yy<=Math.min(ny-1,Math.ceil(high[1]!)-1);yy++)for(let xx=Math.max(0,Math.floor(low[0]!));xx<=Math.min(nx-1,Math.ceil(high[0]!)-1);xx++)donors.add(ownerOf(xx,yy,zz));
   neighbors[o]=[...donors];ownerTiles[o]=tile;ownerWidths[o]=width;
   seed[o]=Number(volume[at]!==0);if(seed[o])tileSeed[tile]=1;
   for(let a=0;a<3;a++){
    tileLow[a]=Math.min(tileLow[a]!,Math.max(0,Math.min(dims[a]!/4-1,Math.floor(low[a]!/4))));
    tileHigh[a]=Math.max(tileHigh[a]!,Math.max(0,Math.min(dims[a]!/4-1,Math.floor((Math.ceil(high[a]!)-1)/4))));
   }
  }
  for(let z=tileLow[2]!;z<=tileHigh[2]!;z++)for(let y=tileLow[1]!;y<=tileHigh[1]!;y++)for(let x=tileLow[0]!;x<=tileHigh[0]!;x++)tileNeighbors[tile]!.add(x+tx*(y+ty*z));
 }
 const closure=(graph:readonly (readonly number[])[],seeds:Uint8Array)=>{
  const gather=(from:Uint8Array)=>Uint8Array.from(graph,row=>Number(row.some(i=>from[i])));
  const scatter=(from:Uint8Array)=>{const into=new Uint8Array(graph.length);for(let i=0;i<graph.length;i++)if(from[i])for(const j of graph[i]!)into[j]=1;return into;};
  const R1=gather(seeds),D2=scatter(R1),Q2=gather(D2),D1=scatter(Q2),Q1=gather(D1),Q0=gather(Q1),donors=scatter(Q0);
  return {Q0,Q1,Q2,R1,donors,D1,D2};
 };
 const exact=closure(neighbors,Uint8Array.from(seed)),boxed=closure(tileNeighbors.map(s=>[...s]),tileSeed);
 const counts=Object.fromEntries(Object.entries(exact).map(([name,membership])=>{
  const old=boxed[name as keyof typeof boxed];let current=0,necessary=0;
  for(let o=0;o<ownerTiles.length;o++){current+=old[ownerTiles[o]!]!;necessary+=membership[o]!;}
  return [name,{tileClosureOwners:current,ownerClosureOwners:necessary,avoidableOwners:current-necessary,avoidableFraction:current?1-necessary/current:0}];
 }));
 return {scope:"Frozen source-free V and departure fields; exact owner graph versus current tile-box closure. Work counts only; no speedup claim.",owners:ownerTiles.length,fineOwners:ownerWidths.filter(w=>w===1).length,counts};
}
