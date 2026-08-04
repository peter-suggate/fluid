import { OCTREE_LOSASSO_FACE_WORDS } from "./octree-losasso-operator";

/** GPU-gated copy from the isolated candidate operator bank into fixed live buffers. */
export const webgpuOctreeLosassoAuthorityCommitWGSL = /* wgsl */ `
const FACE_WORDS:u32=${OCTREE_LOSASSO_FACE_WORDS}u;
struct Params{rows:u32,faces:u32,incidences:u32,adjacencies:u32,directory:u32,pad:vec3u}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>candidateControl:array<u32>;
@group(0)@binding(2)var<storage,read>ownerCandidate:array<u32>;
@group(0)@binding(3)var<storage,read>frontier:array<u32>;
@group(0)@binding(4)var<storage,read>candidateOffsets:array<u32>;
@group(0)@binding(5)var<storage,read_write>acceptedOffsets:array<u32>;
@group(0)@binding(6)var<storage,read>candidateIncidences:array<u32>;
@group(0)@binding(7)var<storage,read_write>acceptedIncidences:array<u32>;
@group(0)@binding(8)var<storage,read>candidateFaces:array<u32>;
@group(0)@binding(9)var<storage,read_write>acceptedFaces:array<u32>;
@group(0)@binding(10)var<storage,read>candidateRowDispatch:array<u32>;
@group(0)@binding(11)var<storage,read_write>acceptedRowDispatch:array<u32>;
@group(0)@binding(12)var<storage,read>candidateFaceDispatch:array<u32>;
@group(0)@binding(13)var<storage,read_write>acceptedFaceDispatch:array<u32>;
@group(0)@binding(14)var<storage,read>candidateRhs:array<f32>;
@group(0)@binding(15)var<storage,read_write>acceptedRhs:array<f32>;
@group(0)@binding(16)var<storage,read>candidateMetrics:array<vec4u>;
@group(0)@binding(17)var<storage,read_write>acceptedMetrics:array<vec4u>;
@group(0)@binding(18)var<storage,read>candidateAdjacencyOffsets:array<u32>;
@group(0)@binding(19)var<storage,read_write>acceptedAdjacencyOffsets:array<u32>;
@group(0)@binding(20)var<storage,read>candidateAdjacency:array<u32>;
@group(0)@binding(21)var<storage,read_write>acceptedAdjacency:array<u32>;
@group(0)@binding(22)var<storage,read>candidateExtended:array<f32>;
@group(0)@binding(23)var<storage,read_write>acceptedExtended:array<f32>;
@group(0)@binding(24)var<storage,read>candidateGeometry:array<vec4u>;
@group(0)@binding(25)var<storage,read_write>acceptedGeometry:array<vec4u>;
@group(0)@binding(26)var<storage,read>candidateDirectory:array<vec2u>;
@group(0)@binding(27)var<storage,read_write>acceptedDirectory:array<vec2u>;
@group(0)@binding(28)var<storage,read>candidateDiagonal:array<f32>;
@group(0)@binding(29)var<storage,read_write>acceptedDiagonal:array<f32>;
@group(0)@binding(30)var<storage,read>candidateSolver:array<u32>;
@group(0)@binding(31)var<storage,read_write>acceptedSolver:array<u32>;
@group(0)@binding(32)var<storage,read_write>acceptedControl:array<u32>;

fn ready()->bool{
 if(arrayLength(&candidateControl)<8u||arrayLength(&ownerCandidate)<29u
    ||arrayLength(&frontier)<10u){return false;}
 let selector=frontier[7u];if(selector>1u){return false;}
 let generation=ownerCandidate[24u];let count=frontier[selector];
 return generation!=0u&&ownerCandidate[28u]==0x80000000u&&ownerCandidate[23u]==0u
   &&frontier[6u]==1u&&frontier[8u]==generation&&frontier[9u]==0u
   &&count<=p.rows&&candidateControl[0u]==generation&&candidateControl[1u]==count
   &&candidateControl[2u]<=p.faces&&candidateControl[3u]==1u&&candidateControl[4u]==0u;
}
@compute @workgroup_size(64)fn commitLosassoAuthorityRows(@builtin(global_invocation_id)g:vec3u){
 if(!ready()){return;}let row=g.x;let count=candidateControl[1u];
 if(row<=count){acceptedOffsets[row]=candidateOffsets[row];}
 if(row<count){acceptedRhs[row]=candidateRhs[row];acceptedDiagonal[row]=candidateDiagonal[row];}
}
@compute @workgroup_size(64)fn commitLosassoAuthorityIncidences(@builtin(global_invocation_id)g:vec3u){
 if(!ready()){return;}let cursor=g.x;let count=candidateOffsets[candidateControl[1u]];
 if(cursor<count&&cursor<p.incidences){acceptedIncidences[cursor]=candidateIncidences[cursor];}
}
@compute @workgroup_size(64)fn commitLosassoAuthorityFaces(@builtin(global_invocation_id)g:vec3u){
 if(!ready()){return;}let face=g.x;let count=candidateControl[2u];if(face>=count||face>=p.faces){return;}
 let base=FACE_WORDS*face;for(var word=0u;word<FACE_WORDS;word+=1u){acceptedFaces[base+word]=candidateFaces[base+word];}
 acceptedGeometry[face]=candidateGeometry[face];
}
@compute @workgroup_size(64)fn commitLosassoAuthorityFaceAux(@builtin(global_invocation_id)g:vec3u){
 if(!ready()){return;}let cursor=g.x;let count=candidateControl[2u];
 if(cursor<count&&cursor<p.faces){acceptedMetrics[cursor]=candidateMetrics[cursor];}
 if(cursor<=count&&cursor<=p.faces){acceptedAdjacencyOffsets[cursor]=candidateAdjacencyOffsets[cursor];}
 if(cursor<p.adjacencies){acceptedAdjacency[cursor]=candidateAdjacency[cursor];}
}
@compute @workgroup_size(64)fn commitLosassoAuthorityVelocity(@builtin(global_invocation_id)g:vec3u){
 if(!ready()){return;}let face=g.x;if(face<candidateControl[2u]&&face<p.faces){acceptedExtended[face]=candidateExtended[face];}
}
@compute @workgroup_size(64)fn commitLosassoAuthorityDirectory(@builtin(global_invocation_id)g:vec3u){
 if(!ready()){return;}let slot=g.x;if(slot<p.directory){acceptedDirectory[slot]=candidateDirectory[slot];}
}
@compute @workgroup_size(1)fn finishLosassoAuthorityCommit(){
 if(!ready()){return;}
 for(var word=0u;word<3u;word+=1u){acceptedRowDispatch[word]=candidateRowDispatch[word];acceptedFaceDispatch[word]=candidateFaceDispatch[word];}
 for(var word=0u;word<7u;word+=1u){acceptedSolver[word]=candidateSolver[word];}
 // Publish the accepted epoch last. Every consumer gates its arrays through this control.
 for(var word=0u;word<8u;word+=1u){acceptedControl[word]=candidateControl[word];}
}
`;
