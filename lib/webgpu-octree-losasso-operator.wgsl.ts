/** Reduced-binding first-order Losasso matrix apply. */
export const octreeLosassoOperatorWGSL = /* wgsl */ `
const INVALID_ROW: u32 = 0xffffffffu;

struct Face {
  negativeRow: u32,
  positiveRow: u32,
  axis: u32,
  reserved: u32,
  area: f32,
  inverseDistance: f32,
  openFraction: f32,
  normalVelocity: f32,
};

@group(0) @binding(0) var<storage, read> authority: array<u32>;
@group(0) @binding(1) var<storage, read> rowFaceOffsets: array<u32>;
@group(0) @binding(2) var<storage, read> rowFaces: array<u32>;
@group(0) @binding(3) var<storage, read> faces: array<Face>;
@group(0) @binding(4) var<storage, read> inputVector: array<f32>;
@group(0) @binding(5) var<storage, read_write> outputVector: array<f32>;
@group(0) @binding(6) var<storage, read> solveControl: array<u32>;

fn addExact(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;
  if(magnitude==0u){return;}let rawExponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;
  let significand=select(fraction,0x800000u|fraction,rawExponent!=0u);let shift=select(3u,rawExponent+2u,rawExponent!=0u);
  let firstLimb=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);
  for(var digit=0u;digit<4u;digit+=1u){let limb=firstLimb+digit;let byte=i32((shifted>>(digit*8u))&0xffu);
    if(byte!=0&&limb<36u){(*limbs)[limb]+=sign*byte;}}}
fn floorDiv256(value:i32)->vec2i{var carry=value/256;var digit=value-carry*256;if(digit<0){digit+=256;carry-=1;}return vec2i(carry,digit);}
fn exactValue(source:ptr<function,array<i32,36>>)->f32{var limbs=(*source);
  for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256(limbs[limb]);limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}
  let negative=limbs[35]<0;if(negative){for(var limb=0u;limb<36u;limb+=1u){limbs[limb]=-limbs[limb];}
    for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256(limbs[limb]);limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}}
  var magnitude=0.;for(var limb=0u;limb<36u;limb+=1u){magnitude+=ldexp(f32(limbs[limb]),-152+i32(limb*8u));}
  return select(magnitude,-magnitude,negative);}

@compute @workgroup_size(64)
fn applyLosassoOperator(@builtin(global_invocation_id) invocation: vec3u) {
  let row = invocation.x;
  if (authority[3] != 1u || solveControl[0] != 0u || solveControl[1] != 0u
      || row >= authority[1]) {
    return;
  }
  let centre = inputVector[row];
  if (row + 1u >= arrayLength(&rowFaceOffsets)) {
    outputVector[row] = 3.402823e38;
    return;
  }
  let begin = rowFaceOffsets[row];
  let end = rowFaceOffsets[row + 1u];
  if (begin > end || end > arrayLength(&rowFaces) || end - begin > 24u) {
    outputVector[row] = 3.402823e38;
    return;
  }
  var limbs:array<i32,36>;var finiteTerms=true;
  for (var cursor = begin; cursor < end; cursor += 1u) {
    let face = faces[rowFaces[cursor]];
    let coefficient = (face.openFraction * face.area) * face.inverseDistance;
    var difference = centre;
    if (face.negativeRow == row) {
      if (face.positiveRow != INVALID_ROW) {
        difference = centre - inputVector[face.positiveRow];
      }
    } else {
      difference = centre - inputVector[face.negativeRow];
    }
    // Face-id order is not geometric or D4-canonical. The signed radix-256
    // limbs make this fold permutation-independent before its single decode.
    let term=coefficient*difference;if(term!=term||abs(term)>=3.402823e38){finiteTerms=false;}else{addExact(&limbs,term);}
  }
  outputVector[row] = select(3.402823e38,exactValue(&limbs),finiteTerms);
}
`;
