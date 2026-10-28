/** The storage ABI of a whole 32h page represented by eight 16h cells.
 * Page words carry a slot, never a 4h block index. Slot zero is reserved;
 * the all-ones SOLID word is outside the encoding's range. */
export const BULK16_PAGE_TAG=0x80000000;
export const BULK16_PAGE_SLOT_MASK=0x3fffffff;
export const BULK16_PAGE_WORDS=1024;
export const BULK16_PAGE={
 page:0,flags:3,generation:4,open:5,
 pressure:8,velocity:16,velocityNext:52,pressureNext:88,
 rhs:96,diagonal:104,couplings:112,seamHeads:160,
} as const;

export function bulk16PageWord(slot:number):number{
 if(!Number.isSafeInteger(slot)||slot<1||slot>BULK16_PAGE_SLOT_MASK)throw new Error("Invalid BULK16 slot");
 return (BULK16_PAGE_TAG|slot)>>>0;
}
export function bulk16PageSlot(word:number):number{
 return ((word>>>30)===2&&(word&BULK16_PAGE_SLOT_MASK)!==0)?word&BULK16_PAGE_SLOT_MASK:0;
}
/** MAC face of the page's 2x2x2 cells. The normal coordinate is 0..2;
 * tangential coordinates are 0..1. Twelve faces per component, 36 total. */
export function bulk16FaceIndex(axis:number,normal:number,u:number,v:number):number{
 if(!Number.isInteger(axis)||axis<0||axis>2||!Number.isInteger(normal)||normal<0||normal>2||
  !Number.isInteger(u)||u<0||u>1||!Number.isInteger(v)||v<0||v>1)throw new Error("Invalid BULK16 face");
 return 12*axis+4*normal+u+2*v;
}
export const bulk16FaceIndexWGSL=/* wgsl */`
fn bpFace(axis:u32,normal:u32,u:u32,v:u32)->u32{return 12u*axis+4u*normal+u+2u*v;}
fn bpSlot(word:u32)->u32{return select(0u,word&0x3fffffffu,(word>>30u)==2u);}
`;

/** Restrict a page's 4h MAC faces (three component-major 9x8x8 arrays,
 * normal coordinate fastest) by area, and its 8^3 pressures by volume. */
export function restrictBulk16Page(faces:ArrayLike<number>,pressure:ArrayLike<number>):{faces:Float64Array;pressure:Float64Array}{
 if(faces.length!==3*9*8*8||pressure.length!==512)throw new Error("A 4h page has 1728 faces and 512 pressures");
 const coarse=new Float64Array(36),p=new Float64Array(8);
 for(let a=0;a<3;a++)for(let n=0;n<3;n++)for(let v=0;v<2;v++)for(let u=0;u<2;u++){
  let sum=0;
  for(let j=0;j<4;j++)for(let i=0;i<4;i++)sum+=faces[a*576+4*n+9*(4*u+i+8*(4*v+j))]!;
  coarse[bulk16FaceIndex(a,n,u,v)]=sum/16;
 }
 for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++)p[(x>>2)+2*((y>>2)+2*(z>>2))]+=pressure[x+8*(y+8*z)]!/64;
 return {faces:coarse,pressure:p};
}

/** Conservative normal-linear prolongation: each 16h face's sixteen 4h
 * subfaces retain its mean exactly. Pressure is the cell mean plus the
 * supplied physical hydrostatic gradient, sampled at each 4h centre. */
export function prolongBulk16Page(faces:ArrayLike<number>,pressure:ArrayLike<number>,gradient:readonly [number,number,number]=[0,0,0]):{faces:Float64Array;pressure:Float64Array}{
 if(faces.length!==36||pressure.length!==8)throw new Error("A BULK16 page has 36 faces and 8 pressures");
 const fine=new Float64Array(1728),p=new Float64Array(512);
 for(let a=0;a<3;a++)for(let v=0;v<8;v++)for(let u=0;u<8;u++)for(let n=0;n<9;n++){
  const region=Math.min(1,n>>2),fraction=(n-4*region)/4;
  const lo=faces[bulk16FaceIndex(a,region,u>>2,v>>2)]!,hi=faces[bulk16FaceIndex(a,region+1,u>>2,v>>2)]!;
  fine[a*576+n+9*(u+8*v)]=(1-fraction)*lo+fraction*hi;
 }
 for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
  p[x+8*(y+8*z)]=pressure[(x>>2)+2*((y>>2)+2*(z>>2))]!+
   4*(gradient[0]*((x&3)-1.5)+gradient[1]*((y&3)-1.5)+gradient[2]*((z&3)-1.5));
 }
 return {faces:fine,pressure:p};
}

/** Slot records plus free, live and quarantine lists; every capacity is
 * fixed at construction. No allocation is proportional to 4h rows. */
export function bulk16PageLayout(capacity:number){
 if(!Number.isSafeInteger(capacity)||capacity<1||capacity>BULK16_PAGE_SLOT_MASK)throw new Error("Invalid BULK16 capacity");
 const free=32,live=free+capacity,pending=live+capacity,records=Math.ceil((pending+capacity)/64)*64;
 const words=records+BULK16_PAGE_WORDS*(capacity+1);
 if(!Number.isSafeInteger(4*words))throw new Error("BULK16 storage is too large");
 return {capacity,free,live,pending,records,words,bytes:4*words};
}

/** Shared read-only binding for simulation and pressure ownership. Slot
 * identity is in the directory; no reader may use it as a pool block. */
export function bulk16ReadWGSL(layout:ReturnType<typeof bulk16PageLayout>,group:number,binding:number):string{return /* wgsl */`
@group(${group}) @binding(${binding}) var<storage,read> umBulkPages:array<u32>;
fn umBulkRead(slot:u32,word:u32)->u32{return umBulkPages[${layout.records}u+${BULK16_PAGE_WORDS}u*slot+word];}
fn umBulkPressure(slot:u32,cell:vec3u)->f32{return bitcast<f32>(umBulkRead(slot,${BULK16_PAGE.pressure}u+cell.x+2u*(cell.y+2u*cell.z)));}
fn umBulkVelocityFace(slot:u32,axis:u32,normal:u32,u:u32,v:u32)->f32{return bitcast<f32>(umBulkRead(slot,${BULK16_PAGE.velocity}u+12u*axis+4u*normal+u+2u*v));}
`;}
