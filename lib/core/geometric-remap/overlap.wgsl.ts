/** Raw intersection volumes. Deliberately do not use the production helper's
 * min(whole, clipped) guards: these receipts must expose geometry errors.
 * One tetrahedron, six box planes and one affine liquid halfspace fit within
 * this indexed clipper's fixed storage bounds (at most eleven faces).
 */
export const remapOverlapWGSL = /* wgsl */ `
// Indexed storage avoids copying the production clipper's 256 duplicated
// vertices by value at every clip. Metal rejects that composition for stack
// exhaustion. A convex tetrahedron + seven cuts has <=11 faces and <=18
// vertices; 24 vertices, 12 faces and 12 indices per face leave explicit room.
struct Poly {
  vertices:array<vec4f,24>, indices:array<u32,144>, counts:array<u32,12>,
  vertexCount:u32, faceCount:u32, fault:u32,
}
fn signedDistance(v:vec4f,normal:vec3f,bound:f32,liquid:bool)->f32{
  return select(bound-dot(normal,v.xyz),v.w,liquid);
}
fn addVertex(poly:ptr<function,Poly>,v:vec4f,tolerance:f32)->u32{
  for(var i=0u;i<(*poly).vertexCount;i++){
    if(distance((*poly).vertices[i].xyz,v.xyz)<=tolerance){return i;}
  }
  let i=(*poly).vertexCount;
  if(i>=24u){(*poly).fault=1u;return 0u;}
  (*poly).vertices[i]=v;(*poly).vertexCount++;return i;
}
fn clip(poly:ptr<function,Poly>,normal:vec3f,bound:f32,liquid:bool,tolerance:f32){
  var out:Poly;out.fault=(*poly).fault;
  var cap:array<u32,24>;var capCount=0u;
  for(var f=0u;f<(*poly).faceCount;f++){
    let count=(*poly).counts[f];if(count<3u){continue;}
    var polygon:array<u32,12>;var written=0u;
    var previous=(*poly).vertices[(*poly).indices[12u*f+count-1u]];
    var a=signedDistance(previous,normal,bound,liquid);
    for(var k=0u;k<count;k++){
      let current=(*poly).vertices[(*poly).indices[12u*f+k]];
      let b=signedDistance(current,normal,bound,liquid);
      if((a>=0.0)!=(b>=0.0)){
        let point=mix(previous,current,a/(a-b));
        let id=addVertex(&out,point,tolerance);
        if(written<12u){polygon[written]=id;written++;}else{out.fault=1u;}
        var found=false;
        for(var j=0u;j<capCount;j++){found=found||cap[j]==id;}
        if(!found){if(capCount<24u){cap[capCount]=id;capCount++;}else{out.fault=1u;}}
      }
      if(b>=0.0){
        let id=addVertex(&out,current,tolerance);
        if(written<12u){polygon[written]=id;written++;}else{out.fault=1u;}
      }
      previous=current;a=b;
    }
    if(written>=3u){
      if(out.faceCount>=12u){out.fault=1u;continue;}
      out.counts[out.faceCount]=written;
      for(var k=0u;k<written;k++){out.indices[12u*out.faceCount+k]=polygon[k];}
      out.faceCount++;
    }
  }
  if(capCount>=3u){
    if(capCount>12u||out.faceCount>=12u){out.fault=1u;}
    else{
      var center=vec3f(0.0);
      for(var k=0u;k<capCount;k++){center+=out.vertices[cap[k]].xyz;}
      center/=f32(capCount);
      let axis=out.vertices[cap[0]].xyz-center;
      var perpendicular=vec3f(0.0);var best=0.0;
      for(var k=1u;k<capCount;k++){
        let candidate=cross(axis,out.vertices[cap[k]].xyz-center);
        let magnitude=dot(candidate,candidate);
        if(magnitude>best){best=magnitude;perpendicular=candidate;}
      }
      if(best>0.0){
        let u=normalize(axis);let v=normalize(cross(perpendicular,u));
        var angles:array<f32,24>;
        for(var k=0u;k<capCount;k++){
          let delta=out.vertices[cap[k]].xyz-center;
          angles[k]=atan2(dot(delta,v),dot(delta,u));
        }
        for(var k=1u;k<capCount;k++){
          let id=cap[k];let angle=angles[k];var j=k;
          loop{
            if(j==0u){break;}if(angles[j-1u]<=angle){break;}
            cap[j]=cap[j-1u];angles[j]=angles[j-1u];j--;
          }
          cap[j]=id;angles[j]=angle;
        }
        out.counts[out.faceCount]=capCount;
        for(var k=0u;k<capCount;k++){out.indices[12u*out.faceCount+k]=cap[k];}
        out.faceCount++;
      }
    }
  }
  *poly=out;
}
fn volume(poly:ptr<function,Poly>)->f32{
  if((*poly).vertexCount==0u){return 0.0;}
  var center=vec3f(0.0);
  for(var k=0u;k<(*poly).vertexCount;k++){center+=(*poly).vertices[k].xyz;}
  center/=f32((*poly).vertexCount);var result=0.0;
  for(var f=0u;f<(*poly).faceCount;f++){
    let a=(*poly).vertices[(*poly).indices[12u*f]].xyz-center;
    for(var k=1u;k+1u<(*poly).counts[f];k++){
      let b=(*poly).vertices[(*poly).indices[12u*f+k]].xyz-center;
      let c=(*poly).vertices[(*poly).indices[12u*f+k+1u]].xyz-center;
      result+=abs(dot(a,cross(b,c)));
    }
  }
  return result/6.0;
}
struct Input { vertices:array<vec4f,4> }
@group(0) @binding(0) var<storage,read> inputs:array<Input>;
@group(0) @binding(1) var<storage,read_write> transfers:array<vec2f>;
@group(0) @binding(2) var<storage,read> ranges:array<vec2u>;
@group(0) @binding(3) var<storage,read_write> received:array<vec2f>;

@compute @workgroup_size(32)
fn intersect(@builtin(global_invocation_id) gid:vec3u){
  let index=gid.x;if(index>=arrayLength(&inputs)){return;}
  let data=inputs[index];
  var poly:Poly;poly.faceCount=4u;poly.vertexCount=4u;
  for(var k=0u;k<4u;k++){poly.vertices[k]=data.vertices[k];}
  let faces=array<vec3u,4>(vec3u(0,1,2),vec3u(0,3,1),vec3u(0,2,3),vec3u(1,3,2));
  for(var f=0u;f<4u;f++){
    poly.counts[f]=3u;
    for(var k=0u;k<3u;k++){
      poly.indices[f*12u+k]=faces[f][k];
    }
  }
  var longest=0.0;
  for(var a=0u;a<4u;a++){for(var b=a+1u;b<4u;b++){
    longest=max(longest,length(data.vertices[a].xyz-data.vertices[b].xyz));
  }}
  let tolerance=1e-6*longest;
  for(var a=0u;a<3u;a++){
    var axis=vec3f(0.0);axis[a]=1.0;
    clip(&poly,axis,1.0,false,tolerance);
    clip(&poly,-axis,0.0,false,tolerance);
  }
  let capacity=volume(&poly);
  clip(&poly,vec3f(0.0),0.0,true,tolerance);
  let liquid=volume(&poly);
  transfers[index]=select(vec2f(capacity,liquid),vec2f(-1.0),poly.fault!=0u);
}

// Deterministic receiver ownership; each raw record is read once. No atomics,
// receiver normalization, clamping, excess redistribution or intermediate V.
@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid:vec3u){
  let cell=gid.x;if(cell>=arrayLength(&ranges)){return;}
  let range=ranges[cell];var sum=vec2f(0.0);var correction=vec2f(0.0);
  for(var i=range.x;i<range.y;i++){
    let value=transfers[i]-correction;
    let next=sum+value;
    correction=(next-sum)-value;sum=next;
  }
  received[cell]=sum;
}
`;
