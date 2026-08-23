import {
  SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_INTERIOR_WORDS,
  SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM,
  SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM_WORDS,
  type SparseCM12BrickTileFaceProgramLayout,
} from "./sparse-cm12-brick-tile-face-program";

const identifier = (value: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError("BFP1 WGSL arena name must be an identifier");
  }
  return value;
};

/** Accessors for the compact interior-tile and explicit seam-port streams. */
export function createSparseCM12BrickTileFaceProgramWGSL(options: Readonly<{
  layout: SparseCM12BrickTileFaceProgramLayout;
  arenaName?: string;
  baseWords?: number;
}>): string {
  const arena = identifier(options.arenaName ?? "bfp1Program");
  const base = options.baseWords ?? 0, l = options.layout;
  if (!Number.isSafeInteger(base) || base < 0
    || base + l.totalWords > 0xffff_ffff) {
    throw new RangeError("BFP1 WGSL base is outside the u32 arena");
  }
  return /* wgsl */ `
const BFP1_INTERIOR_TILE_COUNT:u32=${l.interiorTileCount}u;
const BFP1_SEAM_PORT_COUNT:u32=${l.seamPortCount}u;
const BFP1_SEAM_PACKET_COUNT:u32=${l.seamPacketCount}u;
const BFP1_INTERIOR_BASE:u32=${base + l.interiorBaseWords}u;
const BFP1_SEAM_BASE:u32=${base + l.seamBaseWords}u;
fn bfp1Load(at:u32)->u32{return ${arena}[at];}
fn bfp1InteriorTile(ordinal:u32)->u32{
  if(ordinal>=BFP1_INTERIOR_TILE_COUNT){return BTI1_INVALID;}
  return bfp1Load(BFP1_INTERIOR_BASE
    +${SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_INTERIOR_WORDS}u*ordinal);
}
fn bfp1SeamPort(ordinal:u32)->vec3u{
  if(ordinal>=BFP1_SEAM_PORT_COUNT){return vec3u(BTI1_INVALID);}
  let at=BFP1_SEAM_BASE+${SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM_WORDS}u*ordinal;
  return vec3u(bfp1Load(at+${SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM.stableTile}u),
    bfp1Load(at+${SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM.address}u),
    bfp1Load(at+${SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM.row}u));
}
`;
}
