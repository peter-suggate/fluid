import {
  SPARSE_CM12_LOGICAL_OWNER_BRICK_LIMIT,
  SPARSE_CM12_LOGICAL_OWNER_FLAG,
  SPARSE_CM12_LOGICAL_OWNER_HEADER,
  SPARSE_CM12_LOGICAL_OWNER_HEADER_WORDS,
  SPARSE_CM12_LOGICAL_OWNER_INVALID,
  SPARSE_CM12_LOGICAL_OWNER_MAGIC,
  SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS,
  SPARSE_CM12_LOGICAL_OWNER_VERSION,
  type SparseCM12LogicalOwnerDirectoryLayout,
} from "./sparse-cm12-logical-owner-directory";

export interface SparseCM12LogicalOwnerDirectoryWGSLOptions {
  readonly layout: SparseCM12LogicalOwnerDirectoryLayout;
  /** Existing immutable `array<u32>` directory binding. */
  readonly directoryName?: string;
  readonly brickActiveFunction?: string;
  readonly acceptedBrickResolutionFunction?: string;
  readonly templateBrickCellRangeFunction?: string;
  readonly cellResolutionFunction?: string;
  readonly cellOpenVolumeFunction?: string;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

/**
 * Binding-free LOD1 accessors. Mutable activity, rung, and range authority is
 * deliberately supplied by functions in the embedding shader: the directory
 * stays immutable while those values continue to follow the accepted GPU
 * topology generation.
 */
export function createSparseCM12LogicalOwnerDirectoryWGSL(
  options: SparseCM12LogicalOwnerDirectoryWGSLOptions,
): string {
  const { layout } = options;
  const directory = identifier(options.directoryName ?? "logicalOwnerDirectory",
    "directoryName");
  const brickActive = identifier(options.brickActiveFunction ?? "brickActive",
    "brickActiveFunction");
  const acceptedResolution = identifier(
    options.acceptedBrickResolutionFunction ?? "acceptedBrickResolution",
    "acceptedBrickResolutionFunction",
  );
  const cellRange = identifier(
    options.templateBrickCellRangeFunction ?? "templateBrickCellRange",
    "templateBrickCellRangeFunction",
  );
  const cellResolution = identifier(options.cellResolutionFunction ?? "cellResolution",
    "cellResolutionFunction");
  const cellOpenVolume = identifier(options.cellOpenVolumeFunction ?? "cellOpenVolume",
    "cellOpenVolumeFunction");
  const h = (word: number) => `${word}u`;
  const dimensions = layout.logicalBrickDimensions;
  return /* wgsl */ `
const cm12LogicalOwnerMagic:u32=0x${SPARSE_CM12_LOGICAL_OWNER_MAGIC.toString(16)}u;
const cm12LogicalOwnerVersion:u32=${SPARSE_CM12_LOGICAL_OWNER_VERSION}u;
const cm12LogicalOwnerInvalid:u32=0x${SPARSE_CM12_LOGICAL_OWNER_INVALID.toString(16)}u;
const cm12LogicalOwnerBrickLimit:u32=${SPARSE_CM12_LOGICAL_OWNER_BRICK_LIMIT}u;
const cm12LogicalOwnerBrickFine:u32=${layout.brickFineResolution}u;
const cm12LogicalOwnerPresentationPage:u32=${layout.presentationPageResolution}u;
const cm12LogicalOwnerDimensions:vec3u=vec3u(${dimensions[0]}u,${dimensions[1]}u,${dimensions[2]}u);
const cm12LogicalOwnerCount:u32=${layout.logicalBrickCount}u;
const cm12LogicalOwnerResidentCount:u32=${layout.residentBrickCount}u;
const cm12LogicalOwnerMaximumSpanLog:u32=${layout.maximumSpanLog}u;
const cm12LogicalOwnerAtlasGeneration:u32=${layout.atlasGeneration}u;
const cm12LogicalOwnerRecordBase:u32=${layout.recordBaseWords}u;
const cm12LogicalOwnerTotalWords:u32=${layout.totalWords}u;

struct CM12LogicalOwnerRecord {
  brick:u32,
  spanBricks:u32,
  originKey:u32,
}

fn cm12LogicalOwnerInvalidRecord()->CM12LogicalOwnerRecord{
  return CM12LogicalOwnerRecord(cm12LogicalOwnerInvalid,0u,cm12LogicalOwnerInvalid);
}

fn cm12LogicalOwnerHeaderValid()->bool{
  let flags=${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.flags)}];
  return arrayLength(&${directory})>=cm12LogicalOwnerTotalWords
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.magic)}]==cm12LogicalOwnerMagic
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.version)}]==cm12LogicalOwnerVersion
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.headerWords)}]
      ==${SPARSE_CM12_LOGICAL_OWNER_HEADER_WORDS}u
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.recordWords)}]
      ==${SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS}u
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.brickFineResolution)}]
      ==cm12LogicalOwnerBrickFine
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.presentationPageResolution)}]
      ==cm12LogicalOwnerPresentationPage
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.logicalBricksX)}]
      ==cm12LogicalOwnerDimensions.x
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.logicalBricksY)}]
      ==cm12LogicalOwnerDimensions.y
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.logicalBricksZ)}]
      ==cm12LogicalOwnerDimensions.z
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.logicalBrickCount)}]
      ==cm12LogicalOwnerCount
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.residentBrickCount)}]
      ==cm12LogicalOwnerResidentCount
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.maximumSpanLog)}]
      ==cm12LogicalOwnerMaximumSpanLog
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.atlasGeneration)}]
      ==cm12LogicalOwnerAtlasGeneration
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.recordBase)}]
      ==cm12LogicalOwnerRecordBase
    &&${directory}[${h(SPARSE_CM12_LOGICAL_OWNER_HEADER.totalWords)}]
      ==cm12LogicalOwnerTotalWords
    &&(flags&${SPARSE_CM12_LOGICAL_OWNER_FLAG.complete}u)!=0u
    &&(flags&${SPARSE_CM12_LOGICAL_OWNER_FLAG.validated}u)!=0u;
}

fn cm12LogicalOwnerCoordinate(key:u32)->vec3u{
  let xy=cm12LogicalOwnerDimensions.x*cm12LogicalOwnerDimensions.y;
  let z=key/xy;let remainder=key-z*xy;let y=remainder/cm12LogicalOwnerDimensions.x;
  return vec3u(remainder-y*cm12LogicalOwnerDimensions.x,y,z);
}

fn cm12LogicalOwnerKey(coordinate:vec3u)->u32{
  return coordinate.x+cm12LogicalOwnerDimensions.x
    *(coordinate.y+cm12LogicalOwnerDimensions.y*coordinate.z);
}

fn cm12LogicalOwnerRecordAtKey(key:u32)->CM12LogicalOwnerRecord{
  if(!cm12LogicalOwnerHeaderValid()||key>=cm12LogicalOwnerCount){
    return cm12LogicalOwnerInvalidRecord();
  }
  let at=cm12LogicalOwnerRecordBase+${SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS}u*key;
  let packed=${directory}[at];let originKey=${directory}[at+1u];
  if(packed==cm12LogicalOwnerInvalid||originKey==cm12LogicalOwnerInvalid){
    return cm12LogicalOwnerInvalidRecord();
  }
  let brick=packed>>5u;let spanLog=packed&31u;
  if(brick>=cm12LogicalOwnerResidentCount||brick>=cm12LogicalOwnerBrickLimit
    ||spanLog>cm12LogicalOwnerMaximumSpanLog||spanLog>30u
    ||originKey>=cm12LogicalOwnerCount){
    return cm12LogicalOwnerInvalidRecord();
  }
  let span=1u<<spanLog;let origin=cm12LogicalOwnerCoordinate(originKey);
  let query=cm12LogicalOwnerCoordinate(key);
  if(any(origin%vec3u(span)!=vec3u(0u))||any(query<origin)
    ||any(query-origin>=vec3u(span))){
    return cm12LogicalOwnerInvalidRecord();
  }
  return CM12LogicalOwnerRecord(brick,span,originKey);
}

fn cm12LogicalOwnerRecordAtCoordinate(coordinate:vec3u)->CM12LogicalOwnerRecord{
  if(any(coordinate>=cm12LogicalOwnerDimensions)){
    return cm12LogicalOwnerInvalidRecord();
  }
  return cm12LogicalOwnerRecordAtKey(cm12LogicalOwnerKey(coordinate));
}

fn cm12LogicalOwnerBrickAtKey(key:u32)->u32{
  return cm12LogicalOwnerRecordAtKey(key).brick;
}

fn cm12LogicalOwnerSpanAtKey(key:u32)->u32{
  return cm12LogicalOwnerRecordAtKey(key).spanBricks;
}

fn cm12LogicalOwnerOriginKeyAtKey(key:u32)->u32{
  return cm12LogicalOwnerRecordAtKey(key).originKey;
}

fn cm12LogicalOwnerActiveAtKey(key:u32)->bool{
  let owner=cm12LogicalOwnerRecordAtKey(key);
  return owner.brick!=cm12LogicalOwnerInvalid&&${brickActive}(owner.brick);
}

fn cm12LogicalOwnerRungAtKey(key:u32)->u32{
  let owner=cm12LogicalOwnerRecordAtKey(key);
  if(owner.brick==cm12LogicalOwnerInvalid){return 0u;}
  return ${acceptedResolution}(owner.brick);
}

fn cm12LogicalOwnerCellRangeAtKey(key:u32)->vec2u{
  let owner=cm12LogicalOwnerRecordAtKey(key);
  if(owner.brick==cm12LogicalOwnerInvalid){return vec2u(cm12LogicalOwnerInvalid,0u);}
  let resolution=${acceptedResolution}(owner.brick);
  if(resolution==0u||(resolution&(resolution-1u))!=0u
    ||resolution>cm12LogicalOwnerBrickFine
    ||cm12LogicalOwnerBrickFine%resolution!=0u){
    return vec2u(cm12LogicalOwnerInvalid,0u);
  }
  return ${cellRange}(owner.brick,resolution);
}

// Exact replacement for the old compactOwnerCellAt directory portion. It
// intentionally does not apply the active/open predicate; that is the second
// helper below, matching the old compactOwnerCellAt -> ownerCellAt split.
fn cm12LogicalOwnerCellAtFine(q:vec3i,finestDimensions:vec3u)->vec3u{
  if(any(q<vec3i(0))||any(q>=vec3i(finestDimensions))){return vec3u(cm12LogicalOwnerInvalid);}
  let uq=vec3u(q);let logical=uq/cm12LogicalOwnerBrickFine;
  let owner=cm12LogicalOwnerRecordAtCoordinate(logical);
  if(owner.brick==cm12LogicalOwnerInvalid){return vec3u(cm12LogicalOwnerInvalid);}
  let origin=cm12LogicalOwnerCoordinate(owner.originKey);
  let resolution=${acceptedResolution}(owner.brick);
  if(resolution==0u||(resolution&(resolution-1u))!=0u
    ||resolution>cm12LogicalOwnerBrickFine
    ||cm12LogicalOwnerBrickFine%resolution!=0u
    ||owner.spanBricks>0xffffffffu/cm12LogicalOwnerBrickFine){
    return vec3u(cm12LogicalOwnerInvalid);
  }
  let range=${cellRange}(owner.brick,resolution);
  let spanFine=cm12LogicalOwnerBrickFine*owner.spanBricks;
  let scale=spanFine/resolution;
  if(scale==0u||any(origin>=(finestDimensions+vec3u(cm12LogicalOwnerBrickFine-1u))
    /cm12LogicalOwnerBrickFine)){
    return vec3u(cm12LogicalOwnerInvalid);
  }
  let originFine=origin*cm12LogicalOwnerBrickFine;
  let local=(uq-originFine)/scale;
  let valid=min(finestDimensions-originFine+vec3u(scale-1u),vec3u(spanFine))/scale;
  let offset=local.x+valid.x*(local.y+valid.y*local.z);
  if(range.x==cm12LogicalOwnerInvalid||offset>=range.y
    ||range.x>cm12LogicalOwnerInvalid-offset){
    return vec3u(cm12LogicalOwnerInvalid);
  }
  return vec3u(range.x+offset,owner.brick,resolution);
}

fn cm12LogicalActiveCellAtFine(q:vec3i,finestDimensions:vec3u)->u32{
  let owner=cm12LogicalOwnerCellAtFine(q,finestDimensions);
  if(owner.x==cm12LogicalOwnerInvalid||!${brickActive}(owner.y)
    ||${cellResolution}(owner.x)!=owner.z||${cellOpenVolume}(owner.x)<=1e-8){
    return cm12LogicalOwnerInvalid;
  }
  return owner.x;
}
`;
}
