/** A workgroup's tile and its six face neighbours, resolved once by its
 * lanes instead of per face: lanes 0..5 call utcLoadSide(t, lane) for tile
 * key t, then the workgroup takes a barrier before any read. Side s =
 * 2 axis + (positive): the neighbour's key (UM_TILES where no page stores
 * it), width (0 there) and page kind (UM_RESIDENT where stored, else the
 * table's AIR or SOLID); utcOrigin is t's first cell. */
export const uniformTileContextSidesWGSL = /* wgsl */ `
struct UTCSide {key:u32,width:u32,kind:u32}
var<workgroup> utcSides:array<UTCSide,6>;
var<workgroup> utcOrigin:vec3i;
fn utcLoadSide(t:u32,lane:u32){
 let c=vec3i(umTileCoord(t));var d=vec3i(0);d[lane/2u]=select(-1,1,(lane&1u)!=0u);
 let k=umTileKey(c+d);var side=UTCSide(UM_TILES,0u,umTileKind(c+d));
 if(umKeyStored(k)){side=UTCSide(k,umTileWidth(k),UM_RESIDENT);}
 utcSides[lane]=side;
 if(lane==0u){utcOrigin=4*c;}
}
`;
