#include <metal_stdlib>
using namespace metal;

struct Params {
    float4 dimsDt;
    float4 cellGravity;
    float4 containerDensity;
    float4 cameraPosition;
    float4 cameraTarget;
    float4 viewport;
    float4 physical;
    float4 boundary;
};

struct BodyGPU { float4 positionShape; float4 dimensions; float4 orientation; float4 linearVelocity; float4 angularVelocity; float4 inverseMassRestitutionFriction; float4 inverseInertia; };

inline float3 qrotate(float4 q,float3 v){float3 uv=cross(q.yzw,v),uuv=cross(q.yzw,uv);return v+2*(q.x*uv+uuv);}
inline float3 qinverse(float4 q,float3 v){return qrotate(float4(q.x,-q.yzw),v);}
inline float4 qmultiply(float4 a,float4 b){return float4(a.x*b.x-dot(a.yzw,b.yzw),a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));}
inline float bodySDF(BodyGPU body,float3 world){
    float3 q=qinverse(body.orientation,world-body.positionShape.xyz),d=body.dimensions.xyz;int shape=int(round(body.positionShape.w));
    if(shape==0)return length(q)-d.x;
    if(shape==1){float3 e=abs(q)-d*.5f;return length(max(e,0.0f))+min(max(e.x,max(e.y,e.z)),0.0f);}
    if(shape==2){float y=clamp(q.y,-d.y*.5f,d.y*.5f);return length(float3(q.x,q.y-y,q.z))-d.x;}
    float2 e=float2(length(q.xz)-d.x,abs(q.y)-d.y*.5f);return length(max(e,0.0f))+min(max(e.x,e.y),0.0f);
}
inline float positiveCube(float x){x=max(x,0.0f);return x*x*x;}
inline float positiveSquare(float x){x=max(x,0.0f);return x*x;}
inline float fraction1(float alpha,float a){return a<1e-8f?select(0.0f,1.0f,alpha>=0):clamp(alpha/a,0.0f,1.0f);}
inline float fraction2(float alpha,float a,float b){if(a<1e-8f)return fraction1(alpha,b);if(b<1e-8f)return fraction1(alpha,a);return clamp((positiveSquare(alpha)-positiveSquare(alpha-a)-positiveSquare(alpha-b)+positiveSquare(alpha-a-b))/(2*a*b),0.0f,1.0f);}
inline float fraction3(float alpha,float a,float b,float c){if(a<1e-8f)return fraction2(alpha,b,c);if(b<1e-8f)return fraction2(alpha,a,c);if(c<1e-8f)return fraction2(alpha,a,b);float value=positiveCube(alpha)-positiveCube(alpha-a)-positiveCube(alpha-b)-positiveCube(alpha-c)+positiveCube(alpha-a-b)+positiveCube(alpha-a-c)+positiveCube(alpha-b-c)-positiveCube(alpha-a-b-c);return clamp(value/(6*a*b*c),0.0f,1.0f);}
inline float solidFraction3(float sdf,float3 normal,float3 h){float3 signedA=normal*h;float alpha=dot(signedA,float3(.5f))-sdf;float3 a=abs(signedA);if(signedA.x<0)alpha+=a.x;if(signedA.y<0)alpha+=a.y;if(signedA.z<0)alpha+=a.z;return fraction3(alpha,a.x,a.y,a.z);}
inline float solidFraction2(float sdf,float2 normal,float2 h){float2 signedA=normal*h;float alpha=dot(signedA,float2(.5f))-sdf;float2 a=abs(signedA);if(signedA.x<0)alpha+=a.x;if(signedA.y<0)alpha+=a.y;return fraction2(alpha,a.x,a.y);}

inline int3 dimensions(constant Params &p) { return int3(p.dimsDt.xyz); }
inline int index3(int3 q, int3 d) { return q.x + d.x * (q.y + d.y * q.z); }
inline bool valid(int3 q, int3 d) { return all(q >= 0) && all(q < d); }
inline float volumeAt(device const float *v, int3 q, int3 d) { return valid(q, d) ? v[index3(q, d)] : 0.0f; }
inline float3 velocityAt(device const float4 *v, int3 q, int3 d) { return valid(q, d) ? v[index3(q, d)].xyz : float3(0.0f); }
inline float4 solidAt(device const float4*s,int3 q,int3 d){return valid(q,d)?s[index3(q,d)]:float4(1,-1,1,0);}
inline float3 aperturesAt(device const float4*s,int3 q,int3 d){return valid(q,d)?s[index3(q,d)].xyz:float3(0);}

kernel void buildSolidGeometry(device const BodyGPU*bodies[[buffer(0)]],device float4*solidCell[[buffer(1)]],device float4*solidFace[[buffer(2)]],constant Params&p[[buffer(3)]],uint3 gid[[thread_position_in_grid]]){
    int3 d=dimensions(p),id=int3(gid);if(!valid(id,d))return;int i=index3(id,d);float3 h=p.cellGravity.xyz;float3 world=float3(-p.containerDensity.x*.5f+(id.x+.5f)*h.x,(id.y+.5f)*h.y,-p.containerDensity.z*.5f+(id.z+.5f)*h.z);
    float best=1e20f;int owner=-1;for(int b=0;b<int(p.boundary.z);b++){float value=bodySDF(bodies[b],world);if(value<best){best=value;owner=b;}}
    if(owner<0){solidCell[i]=float4(1,-1,1e20f,0);solidFace[i]=float4(1,1,1,0);return;}
    float radius=.5f*length(h);if(best>=radius){solidCell[i]=float4(1,float(owner),best,0);solidFace[i]=float4(1,1,1,0);return;}if(best<=-radius){solidCell[i]=float4(0,float(owner),best,0);solidFace[i]=float4(0,0,0,0);return;}
    BodyGPU body=bodies[owner];float epsilon=max(min(min(h.x,h.y),h.z)*.2f,1e-5f);float3 gradient=float3(bodySDF(body,world+float3(epsilon,0,0))-bodySDF(body,world-float3(epsilon,0,0)),bodySDF(body,world+float3(0,epsilon,0))-bodySDF(body,world-float3(0,epsilon,0)),bodySDF(body,world+float3(0,0,epsilon))-bodySDF(body,world-float3(0,0,epsilon)));float3 normal=gradient/max(length(gradient),1e-8f);
    float open=1-solidFraction3(best,normal,h);float sx=bodySDF(body,world+float3(h.x*.5f,0,0)),sy=bodySDF(body,world+float3(0,h.y*.5f,0)),sz=bodySDF(body,world+float3(0,0,h.z*.5f));float ax=1-solidFraction2(sx,normal.yz,h.yz),ay=1-solidFraction2(sy,normal.xz,h.xz),az=1-solidFraction2(sz,normal.xy,h.xy);
    solidCell[i]=float4(open,float(owner),best,0);solidFace[i]=float4(ax,ay,az,0);
}

inline float sampleVolume(device const float *v, float3 p, int3 d) {
    float3 q = clamp(p - 0.5f, float3(0.0f), float3(d - 1));
    int3 b = int3(floor(q)); float3 f = fract(q);
    float a = mix(volumeAt(v,b,d), volumeAt(v,b+int3(1,0,0),d), f.x);
    float b0 = mix(volumeAt(v,b+int3(0,1,0),d), volumeAt(v,b+int3(1,1,0),d), f.x);
    float c = mix(volumeAt(v,b+int3(0,0,1),d), volumeAt(v,b+int3(1,0,1),d), f.x);
    float e = mix(volumeAt(v,b+int3(0,1,1),d), volumeAt(v,b+int3(1,1,1),d), f.x);
    return mix(mix(a,b0,f.y), mix(c,e,f.y), f.z);
}

inline float3 sampleVelocity(device const float4 *v, float3 p, int3 d) {
    float3 q = clamp(p - 0.5f, float3(0.0f), float3(d - 1));
    int3 b = int3(floor(q)); float3 f = fract(q);
    float3 a = mix(velocityAt(v,b,d), velocityAt(v,b+int3(1,0,0),d), f.x);
    float3 b0 = mix(velocityAt(v,b+int3(0,1,0),d), velocityAt(v,b+int3(1,1,0),d), f.x);
    float3 c = mix(velocityAt(v,b+int3(0,0,1),d), velocityAt(v,b+int3(1,0,1),d), f.x);
    float3 e = mix(velocityAt(v,b+int3(0,1,1),d), velocityAt(v,b+int3(1,1,1),d), f.x);
    return mix(mix(a,b0,f.y), mix(c,e,f.y), f.z);
}

inline float interfaceFraction(float a,float b){return clamp((a-.5f)/max(abs(a-b),1e-6f),.05f,1.0f);}
inline float3 volumeGradient(device const float*v,int3 id,int3 d,float3 h){return float3(volumeAt(v,id+int3(1,0,0),d)-volumeAt(v,id-int3(1,0,0),d),volumeAt(v,id+int3(0,1,0),d)-volumeAt(v,id-int3(0,1,0),d),volumeAt(v,id+int3(0,0,1),d)-volumeAt(v,id-int3(0,0,1),d))/(2*h);}
inline float3 interfaceNormal(device const float*v,int3 id,int3 d,float3 h){float3 g=volumeGradient(v,id,d,h);return g/max(length(g),1e-6f);}
inline float curvature(device const float*v,int3 id,int3 d,float3 h){return -((interfaceNormal(v,id+int3(1,0,0),d,h).x-interfaceNormal(v,id-int3(1,0,0),d,h).x)/(2*h.x)+(interfaceNormal(v,id+int3(0,1,0),d,h).y-interfaceNormal(v,id-int3(0,1,0),d,h).y)/(2*h.y)+(interfaceNormal(v,id+int3(0,0,1),d,h).z-interfaceNormal(v,id-int3(0,0,1),d,h).z)/(2*h.z));}
inline float3 laplacian(device const float4*v,int3 id,int3 d,float3 h){float3 c=velocityAt(v,id,d);return(velocityAt(v,id+int3(1,0,0),d)-2*c+velocityAt(v,id-int3(1,0,0),d))/(h.x*h.x)+(velocityAt(v,id+int3(0,1,0),d)-2*c+velocityAt(v,id-int3(0,1,0),d))/(h.y*h.y)+(velocityAt(v,id+int3(0,0,1),d)-2*c+velocityAt(v,id-int3(0,0,1),d))/(h.z*h.z);}
inline float strain(device const float4*v,int3 id,int3 d,float3 h){float3 dx=(velocityAt(v,id+int3(1,0,0),d)-velocityAt(v,id-int3(1,0,0),d))/(2*h.x),dy=(velocityAt(v,id+int3(0,1,0),d)-velocityAt(v,id-int3(0,1,0),d))/(2*h.y),dz=(velocityAt(v,id+int3(0,0,1),d)-velocityAt(v,id-int3(0,0,1),d))/(2*h.z);float sxy=.5f*(dx.y+dy.x),sxz=.5f*(dx.z+dz.x),syz=.5f*(dy.z+dz.y);return sqrt(2*(dx.x*dx.x+dy.y*dy.y+dz.z*dz.z+2*(sxy*sxy+sxz*sxz+syz*syz)));}
inline float rawFlux(device const float4*vel,device const float*vol,device const float4*solidFace,int3 id,int axis,float dt,int3 d,float3 h){if(!valid(id,d))return 0;int3 o=axis==0?int3(1,0,0):(axis==1?int3(0,1,0):int3(0,0,1));float speed=velocityAt(vel,id,d)[axis],aperture=aperturesAt(solidFace,id,d)[axis];return aperture*dt/h[axis]*speed*(speed>=0?volumeAt(vol,id,d):volumeAt(vol,id+o,d));}
inline float outward(device const float4*vel,device const float*vol,device const float4*solidFace,int3 id,float dt,int3 d,float3 h){return max(rawFlux(vel,vol,solidFace,id,0,dt,d,h),0.0f)+max(-rawFlux(vel,vol,solidFace,id-int3(1,0,0),0,dt,d,h),0.0f)+max(rawFlux(vel,vol,solidFace,id,1,dt,d,h),0.0f)+max(-rawFlux(vel,vol,solidFace,id-int3(0,1,0),1,dt,d,h),0.0f)+max(rawFlux(vel,vol,solidFace,id,2,dt,d,h),0.0f)+max(-rawFlux(vel,vol,solidFace,id-int3(0,0,1),2,dt,d,h),0.0f);}
inline float inward(device const float4*vel,device const float*vol,device const float4*solidFace,int3 id,float dt,int3 d,float3 h){return max(-rawFlux(vel,vol,solidFace,id,0,dt,d,h),0.0f)+max(rawFlux(vel,vol,solidFace,id-int3(1,0,0),0,dt,d,h),0.0f)+max(-rawFlux(vel,vol,solidFace,id,1,dt,d,h),0.0f)+max(rawFlux(vel,vol,solidFace,id-int3(0,1,0),1,dt,d,h),0.0f)+max(-rawFlux(vel,vol,solidFace,id,2,dt,d,h),0.0f)+max(rawFlux(vel,vol,solidFace,id-int3(0,0,1),2,dt,d,h),0.0f);}
inline float4 auxiliaryAt(device const float4*a,int3 q,int3 d){return valid(q,d)?a[index3(q,d)]:float4(0,0,0,0);}
inline float limitedFlux(device const float4*vel,device const float*vol,device const float4*aux,device const float4*solidFace,int3 id,int axis,float dt,int3 d,float3 h){int3 o=axis==0?int3(1,0,0):(axis==1?int3(0,1,0):int3(0,0,1));float f=rawFlux(vel,vol,solidFace,id,axis,dt,d,h);int3 donor=f>=0?id:id+o,receiver=f>=0?id+o:id;return f*min(auxiliaryAt(aux,donor,d).x,auxiliaryAt(aux,receiver,d).y);}

// Conservative swept-solid remap. Each directed proposal is evaluated identically
// by its donor and receiver, so the gather update preserves liquid exactly without
// requiring floating-point atomics. Repeating the triplet propagates liquid out of
// clusters of newly covered cells while receiver scaling enforces local capacity.
kernel void prepareSolidRemap(device const float*volume[[buffer(0)]],device const float4*solidCell[[buffer(1)]],device float4*remap[[buffer(2)]],constant Params&p[[buffer(3)]],uint3 gid[[thread_position_in_grid]]){
    int3 d=dimensions(p),id=int3(gid);if(!valid(id,d))return;float open=solidAt(solidCell,id,d).x,phi=volumeAt(volume,id,d),excess=max(phi-open,0.0f),capacity=max(open-min(phi,open),0.0f),neighborCapacity=0;const int3 offsets[6]={int3(-1,0,0),int3(1,0,0),int3(0,-1,0),int3(0,1,0),int3(0,0,-1),int3(0,0,1)};for(int n=0;n<6;n++){int3 q=id+offsets[n];float qo=solidAt(solidCell,q,d).x,qv=volumeAt(volume,q,d);neighborCapacity+=max(qo-min(qv,qo),0.0f);}remap[index3(id,d)]=float4(excess,capacity,neighborCapacity,0);
}
inline float remapProposal(device const float4*remap,int3 donor,int3 receiver,int3 d){float4 source=auxiliaryAt(remap,donor,d),target=auxiliaryAt(remap,receiver,d);return source.x*target.y/max(source.z,1e-12f);}
kernel void limitSolidRemap(device const float4*remap[[buffer(0)]],device float*receiverScale[[buffer(1)]],constant Params&p[[buffer(2)]],uint3 gid[[thread_position_in_grid]]){
    int3 d=dimensions(p),id=int3(gid);if(!valid(id,d))return;const int3 offsets[6]={int3(-1,0,0),int3(1,0,0),int3(0,-1,0),int3(0,1,0),int3(0,0,-1),int3(0,0,1)};float incoming=0;for(int n=0;n<6;n++)incoming+=remapProposal(remap,id+offsets[n],id,d);float capacity=auxiliaryAt(remap,id,d).y;receiverScale[index3(id,d)]=min(1.0f,capacity/max(incoming,1e-12f));
}
kernel void applySolidRemap(device const float*volumeIn[[buffer(0)]],device float*volumeOut[[buffer(1)]],device const float4*remap[[buffer(2)]],device const float*receiverScale[[buffer(3)]],constant Params&p[[buffer(4)]],uint3 gid[[thread_position_in_grid]]){
    int3 d=dimensions(p),id=int3(gid);if(!valid(id,d))return;const int3 offsets[6]={int3(-1,0,0),int3(1,0,0),int3(0,-1,0),int3(0,1,0),int3(0,0,-1),int3(0,0,1)};float outgoing=0,incoming=0,selfScale=receiverScale[index3(id,d)];for(int n=0;n<6;n++){int3 q=id+offsets[n];if(!valid(q,d))continue;outgoing+=remapProposal(remap,id,q,d)*receiverScale[index3(q,d)];incoming+=remapProposal(remap,q,id,d)*selfScale;}volumeOut[index3(id,d)]=volumeIn[index3(id,d)]-outgoing+incoming;
}

kernel void buildAuxiliary(device const float4*velocity[[buffer(0)]],device const float*volume[[buffer(1)]],device float4*auxiliary[[buffer(2)]],device const float4*solidCell[[buffer(3)]],device const float4*solidFace[[buffer(4)]],constant Params&p[[buffer(5)]],uint3 gid[[thread_position_in_grid]]){int3 d=dimensions(p),id=int3(gid);if(!valid(id,d))return;float dt=p.dimsDt.w;float3 h=p.cellGravity.xyz;float phi=volumeAt(volume,id,d),capacity=solidAt(solidCell,id,d).x;float donor=min(1.0f,phi/max(outward(velocity,volume,solidFace,id,dt,d,h),1e-9f)),receiver=min(1.0f,max(0.0f,capacity-phi)/max(inward(velocity,volume,solidFace,id,dt,d,h),1e-9f));float k=clamp(curvature(volume,id,d,h),-2/min(min(h.x,h.y),h.z),2/min(min(h.x,h.y),h.z));auxiliary[index3(id,d)]=float4(donor,receiver,k,0);}

kernel void advect(device const float4 *velocityIn [[buffer(0)]],
                   device float4 *velocityOut [[buffer(1)]],
                   device const float *volumeIn [[buffer(2)]],
                   device float *volumeOut [[buffer(3)]],
                   device const float4 *auxiliary [[buffer(4)]],
                   device const float4 *solidFace [[buffer(5)]],
                   constant Params &p [[buffer(6)]], uint3 gid [[thread_position_in_grid]]) {
    int3 d = dimensions(p), id = int3(gid); if (!valid(id,d)) return;
    int i = index3(id,d); float dt = p.dimsDt.w; float3 h = p.cellGravity.xyz;
    float3 position = float3(id) + 0.5f;
    float3 first = sampleVelocity(velocityIn, position, d);
    float3 midpoint = position - 0.5f * dt * first / h;
    float3 transported = sampleVelocity(velocityIn, position - dt * sampleVelocity(velocityIn, midpoint, d) / h, d);
    float phi=volumeAt(volumeIn,id,d);if(phi>0){float delta=pow(h.x*h.y*h.z,1.0f/3.0f),viscosity=p.physical.x/p.containerDensity.w+pow(.17f*delta,2)*strain(velocityIn,id,d,h);transported+=dt*viscosity*laplacian(velocityIn,id,d,h);}
    if (volumeAt(volumeIn,id,d) > 0.001f || volumeAt(volumeIn,id+int3(0,1,0),d) > 0.001f) transported.y += p.cellGravity.w * dt;
    float capillary=p.physical.y/p.containerDensity.w;
    if(valid(id+int3(1,0,0),d))transported.x+=dt*capillary*.5f*(auxiliaryAt(auxiliary,id,d).z+auxiliaryAt(auxiliary,id+int3(1,0,0),d).z)*(volumeAt(volumeIn,id+int3(1,0,0),d)-phi)/h.x;
    if(valid(id+int3(0,1,0),d))transported.y+=dt*capillary*.5f*(auxiliaryAt(auxiliary,id,d).z+auxiliaryAt(auxiliary,id+int3(0,1,0),d).z)*(volumeAt(volumeIn,id+int3(0,1,0),d)-phi)/h.y;
    if(valid(id+int3(0,0,1),d))transported.z+=dt*capillary*.5f*(auxiliaryAt(auxiliary,id,d).z+auxiliaryAt(auxiliary,id+int3(0,0,1),d).z)*(volumeAt(volumeIn,id+int3(0,0,1),d)-phi)/h.z;
    if (id.x == d.x-1) transported.x = 0; if (id.y == d.y-1) transported.y = 0; if (id.z == d.z-1) transported.z = 0;
    velocityOut[i] = float4(transported, 0);

    // Bounded conservative donor-cell VOF update. Shared with the browser path's
    // governing flux convention, but expressed over linear buffers for Apple GPUs.
    float fxp=limitedFlux(velocityIn,volumeIn,auxiliary,solidFace,id,0,dt,d,h),fxm=limitedFlux(velocityIn,volumeIn,auxiliary,solidFace,id-int3(1,0,0),0,dt,d,h);
    float fyp=limitedFlux(velocityIn,volumeIn,auxiliary,solidFace,id,1,dt,d,h),fym=limitedFlux(velocityIn,volumeIn,auxiliary,solidFace,id-int3(0,1,0),1,dt,d,h);
    float fzp=limitedFlux(velocityIn,volumeIn,auxiliary,solidFace,id,2,dt,d,h),fzm=limitedFlux(velocityIn,volumeIn,auxiliary,solidFace,id-int3(0,0,1),2,dt,d,h);
    volumeOut[i]=phi-(fxp-fxm+fyp-fym+fzp-fzm);
}

inline bool cutLiquid(device const float*volume,device const float4*solidCell,int3 q,int3 d){float open=solidAt(solidCell,q,d).x;return open>1e-4f&&volumeAt(volume,q,d)/open>=.5f;}
inline int faceOwner(device const float4*solidCell,int3 a,int3 b,int3 d){float4 sa=solidAt(solidCell,a,d),sb=solidAt(solidCell,b,d);if(sa.x>.99999f&&sb.x>.99999f)return -1;if(sa.x<.99999f&&sb.x>.99999f)return int(round(sa.y));if(sb.x<.99999f&&sa.x>.99999f)return int(round(sb.y));return int(round(select(sa.y,sb.y,sb.z<sa.z)));}
inline float3 rigidVelocity(device const BodyGPU*bodies,int owner,float3 world){if(owner<0)return float3(0);BodyGPU body=bodies[owner];return body.linearVelocity.xyz+cross(body.angularVelocity.xyz,world-body.positionShape.xyz);}
inline float cutDivergence(device const float4*velocity,device const float4*solidCell,device const float4*solidFace,device const BodyGPU*bodies,int3 id,int3 d,constant Params&p){float3 h=p.cellGravity.xyz,world=float3(-p.containerDensity.x*.5f+(id.x+.5f)*h.x,(id.y+.5f)*h.y,-p.containerDensity.z*.5f+(id.z+.5f)*h.z);float3 plus=aperturesAt(solidFace,id,d),mx=aperturesAt(solidFace,id-int3(1,0,0),d),my=aperturesAt(solidFace,id-int3(0,1,0),d),mz=aperturesAt(solidFace,id-int3(0,0,1),d);float3 vp=velocityAt(velocity,id,d),vx=velocityAt(velocity,id-int3(1,0,0),d),vy=velocityAt(velocity,id-int3(0,1,0),d),vz=velocityAt(velocity,id-int3(0,0,1),d);float sx=rigidVelocity(bodies,faceOwner(solidCell,id,id+int3(1,0,0),d),world+float3(h.x*.5f,0,0)).x,smx=rigidVelocity(bodies,faceOwner(solidCell,id-int3(1,0,0),id,d),world-float3(h.x*.5f,0,0)).x;float sy=rigidVelocity(bodies,faceOwner(solidCell,id,id+int3(0,1,0),d),world+float3(0,h.y*.5f,0)).y,smy=rigidVelocity(bodies,faceOwner(solidCell,id-int3(0,1,0),id,d),world-float3(0,h.y*.5f,0)).y;float sz=rigidVelocity(bodies,faceOwner(solidCell,id,id+int3(0,0,1),d),world+float3(0,0,h.z*.5f)).z,smz=rigidVelocity(bodies,faceOwner(solidCell,id-int3(0,0,1),id,d),world-float3(0,0,h.z*.5f)).z;return (plus.x*vp.x+(1-plus.x)*sx-(mx.x*vx.x+(1-mx.x)*smx))/h.x+(plus.y*vp.y+(1-plus.y)*sy-(my.y*vy.y+(1-my.y)*smy))/h.y+(plus.z*vp.z+(1-plus.z)*sz-(mz.z*vz.z+(1-mz.z)*smz))/h.z;}

kernel void jacobi(device const float4*velocity[[buffer(0)]],device const float*volume[[buffer(1)]],device const float*pressureIn[[buffer(2)]],device float*pressureOut[[buffer(3)]],device const float4*solidCell[[buffer(4)]],device const float4*solidFace[[buffer(5)]],device const BodyGPU*bodies[[buffer(6)]],constant Params&p[[buffer(7)]],uint3 gid[[thread_position_in_grid]]){
    int3 d=dimensions(p),id=int3(gid);if(!valid(id,d))return;int i=index3(id,d);if(!cutLiquid(volume,solidCell,id,d)){pressureOut[i]=0;return;}float3 h=p.cellGravity.xyz;const int3 offsets[6]={int3(-1,0,0),int3(1,0,0),int3(0,-1,0),int3(0,1,0),int3(0,0,-1),int3(0,0,1)};float sum=0,diagonal=0;
    for(int n=0;n<6;n++){int3 q=id+offsets[n];if(!valid(q,d))continue;int axis=n/2;float aperture=n%2==0?aperturesAt(solidFace,q,d)[axis]:aperturesAt(solidFace,id,d)[axis];if(aperture<1e-5f)continue;float invH2=1/(h[axis]*h[axis]),theta=cutLiquid(volume,solidCell,q,d)?1:interfaceFraction(volume[i]/max(solidAt(solidCell,id,d).x,1e-6f),volumeAt(volume,q,d)/max(solidAt(solidCell,q,d).x,1e-6f));float coefficient=aperture*invH2/theta;diagonal+=coefficient;if(cutLiquid(volume,solidCell,q,d))sum+=pressureIn[index3(q,d)]*coefficient;}
    float rhs=p.containerDensity.w*cutDivergence(velocity,solidCell,solidFace,bodies,id,d,p)/p.dimsDt.w;pressureOut[i]=mix(pressureIn[i],(sum-rhs)/max(diagonal,1e-8f),.8f);
}

kernel void projectAndCommit(device const float4*velocityIn[[buffer(0)]],device float4*velocityOut[[buffer(1)]],device const float*pressure[[buffer(2)]],device const float*volumeIn[[buffer(3)]],device float*volumeOut[[buffer(4)]],device const float4*solidCell[[buffer(5)]],device const float4*solidFace[[buffer(6)]],device const BodyGPU*bodies[[buffer(7)]],constant Params&p[[buffer(8)]],uint3 gid[[thread_position_in_grid]]){
    int3 d=dimensions(p),id=int3(gid);if(!valid(id,d))return;int i=index3(id,d);float3 h=p.cellGravity.xyz,world=float3(-p.containerDensity.x*.5f+(id.x+.5f)*h.x,(id.y+.5f)*h.y,-p.containerDensity.z*.5f+(id.z+.5f)*h.z),v=velocityIn[i].xyz,ap=aperturesAt(solidFace,id,d);float scale=p.dimsDt.w/p.containerDensity.w,p0=cutLiquid(volumeIn,solidCell,id,d)?pressure[i]:0;int3 offsets[3]={int3(1,0,0),int3(0,1,0),int3(0,0,1)};
    for(int axis=0;axis<3;axis++){int3 q=id+offsets[axis];bool l0=cutLiquid(volumeIn,solidCell,id,d),l1=cutLiquid(volumeIn,solidCell,q,d);if(ap[axis]<1e-5f){v[axis]=rigidVelocity(bodies,faceOwner(solidCell,id,q,d),world+float3(offsets[axis])*h*.5f)[axis];}else if(l0||l1){float theta=l0==l1?1:interfaceFraction(l0?volumeIn[i]/max(solidAt(solidCell,id,d).x,1e-6f):volumeAt(volumeIn,q,d)/max(solidAt(solidCell,q,d).x,1e-6f),l0?volumeAt(volumeIn,q,d)/max(solidAt(solidCell,q,d).x,1e-6f):volumeIn[i]/max(solidAt(solidCell,id,d).x,1e-6f));float p1=l1?pressure[index3(q,d)]:0;v[axis]-=scale*(p1-p0)/(h[axis]*theta);}else v[axis]=0;}
    if(id.x==d.x-1)v.x=0;if(id.y==d.y-1)v.y=0;if(id.z==d.z-1)v.z=0;velocityOut[i]=float4(v,0);volumeOut[i]=volumeIn[i];
}

inline bool insideBody(BodyGPU b,float3 world){float3 q=qinverse(b.orientation,world-b.positionShape.xyz),d=b.dimensions.xyz;int shape=int(round(b.positionShape.w));if(shape==0)return length(q)<=d.x;if(shape==1)return all(abs(q)<=d*.5f);if(shape==2){float y=clamp(q.y,-d.y*.5f,d.y*.5f);return length(float3(q.x,q.y-y,q.z))<=d.x;}return q.x*q.x+q.z*q.z<=d.x*d.x&&abs(q.y)<=d.y*.5f;}

kernel void accumulatePressureTraction(device const float*pressure[[buffer(0)]],device const float*volume[[buffer(1)]],device const float4*solidCell[[buffer(2)]],device const float4*solidFace[[buffer(3)]],device const BodyGPU*bodies[[buffer(4)]],device atomic_int*exchange[[buffer(5)]],constant Params&p[[buffer(6)]],uint3 gid[[thread_position_in_grid]]){int3 d=dimensions(p),id=int3(gid);if(!valid(id,d))return;int i=index3(id,d);float4 solid=solidCell[i];int owner=int(round(solid.y));if(owner<0||solid.x<=1e-5f||solid.x>=.99999f||volume[i]<=1e-5f)return;float3 h=p.cellGravity.xyz,plus=aperturesAt(solidFace,id,d),mx=aperturesAt(solidFace,id-int3(1,0,0),d),my=aperturesAt(solidFace,id-int3(0,1,0),d),mz=aperturesAt(solidFace,id-int3(0,0,1),d);float3 area=float3((mx.x-plus.x)*h.y*h.z,(my.y-plus.y)*h.x*h.z,(mz.z-plus.z)*h.x*h.y);float3 impulse=pressure[i]*area*p.dimsDt.w;float3 world=float3(-p.containerDensity.x*.5f+(id.x+.5f)*h.x,(id.y+.5f)*h.y,-p.containerDensity.z*.5f+(id.z+.5f)*h.z),torque=cross(world-bodies[owner].positionShape.xyz,impulse);int base=owner*8;atomic_fetch_add_explicit(&exchange[base],int(round(impulse.x*1e6f)),memory_order_relaxed);atomic_fetch_add_explicit(&exchange[base+1],int(round(impulse.y*1e6f)),memory_order_relaxed);atomic_fetch_add_explicit(&exchange[base+2],int(round(impulse.z*1e6f)),memory_order_relaxed);atomic_fetch_add_explicit(&exchange[base+3],int(round(torque.x*1e6f)),memory_order_relaxed);atomic_fetch_add_explicit(&exchange[base+4],int(round(torque.y*1e6f)),memory_order_relaxed);atomic_fetch_add_explicit(&exchange[base+5],int(round(torque.z*1e6f)),memory_order_relaxed);atomic_fetch_add_explicit(&exchange[base+6],int(round((1-solid.x)*65536)),memory_order_relaxed);}

kernel void integrateRigidBodies(device BodyGPU*bodies[[buffer(0)]],device atomic_int*exchange[[buffer(1)]],constant Params&p[[buffer(2)]],uint gid[[thread_position_in_grid]]){
    if(gid>=uint(p.boundary.z))return;BodyGPU body=bodies[gid];if(body.angularVelocity.w>.5f){bodies[gid]=body;return;}int base=int(gid)*8;float3 impulse=float3(atomic_load_explicit(&exchange[base],memory_order_relaxed),atomic_load_explicit(&exchange[base+1],memory_order_relaxed),atomic_load_explicit(&exchange[base+2],memory_order_relaxed))/1e6f;float3 angularImpulse=float3(atomic_load_explicit(&exchange[base+3],memory_order_relaxed),atomic_load_explicit(&exchange[base+4],memory_order_relaxed),atomic_load_explicit(&exchange[base+5],memory_order_relaxed))/1e6f;float dt=p.dimsDt.w,invMass=body.inverseMassRestitutionFriction.x,restitution=body.inverseMassRestitutionFriction.y,friction=body.inverseMassRestitutionFriction.z;
    body.linearVelocity.xyz+=impulse*invMass+float3(0,p.cellGravity.w,0)*dt;float3 localAngular=qinverse(body.orientation,angularImpulse);body.angularVelocity.xyz+=qrotate(body.orientation,localAngular*body.inverseInertia.xyz);body.positionShape.xyz+=body.linearVelocity.xyz*dt;float spin=length(body.angularVelocity.xyz);if(spin>1e-7f){float halfAngle=.5f*spin*dt;float4 dq=float4(cos(halfAngle),body.angularVelocity.xyz/spin*sin(halfAngle));body.orientation=normalize(qmultiply(dq,body.orientation));}
    int shape=int(round(body.positionShape.w));float3 d=body.dimensions.xyz;float radius=shape==0?d.x:(shape==1?.5f*length(d):(shape==2?d.x+.5f*d.y:length(float2(d.x,.5f*d.y))));float3 position=body.positionShape.xyz,velocity=body.linearVelocity.xyz;float4 limits=float4(-p.containerDensity.x*.5f+radius,p.containerDensity.x*.5f-radius,-p.containerDensity.z*.5f+radius,p.containerDensity.z*.5f-radius);
    if(position.y<radius){position.y=radius;if(velocity.y<0)velocity.y=-velocity.y*restitution;velocity.xz*=max(0.0f,1-friction*dt*30);}if(position.x<limits.x){position.x=limits.x;if(velocity.x<0)velocity.x=-velocity.x*restitution;}if(position.x>limits.y){position.x=limits.y;if(velocity.x>0)velocity.x=-velocity.x*restitution;}if(position.z<limits.z){position.z=limits.z;if(velocity.z<0)velocity.z=-velocity.z*restitution;}if(position.z>limits.w){position.z=limits.w;if(velocity.z>0)velocity.z=-velocity.z*restitution;}if(p.boundary.y>.5f&&position.y>p.containerDensity.y-radius){position.y=p.containerDensity.y-radius;if(velocity.y>0)velocity.y=-velocity.y*restitution;}body.positionShape.xyz=position;body.linearVelocity.xyz=velocity;bodies[gid]=body;
}

kernel void reduceDiagnostics(device const float4*velocity[[buffer(0)]],device const float*volume[[buffer(1)]],device atomic_uint*result[[buffer(2)]],constant Params&p[[buffer(3)]],uint3 gid[[thread_position_in_grid]]){int3 d=dimensions(p),id=int3(gid);if(!valid(id,d))return;int i=index3(id,d);atomic_fetch_add_explicit(&result[0],uint(clamp(volume[i],0.0f,1.0f)*2048+.5f),memory_order_relaxed);atomic_fetch_max_explicit(&result[1],as_type<uint>(length(velocity[i].xyz)),memory_order_relaxed);}

inline float2 boxHit(float3 ro,float3 rd,float3 lo,float3 hi){float3 inv=1/rd;float3 a=(lo-ro)*inv,b=(hi-ro)*inv;float3 n=min(a,b),f=max(a,b);return float2(max(max(n.x,n.y),n.z),min(min(f.x,f.y),f.z));}
inline float field(device const float *v,float3 world,constant Params&p){float3 size=p.containerDensity.xyz;float3 uv=(world+float3(size.x*.5f,0,size.z*.5f))/size;return sampleVolume(v,uv*p.dimsDt.xyz,dimensions(p));}
inline float4 sphereHit(float3 ro,float3 rd,float radius){float b=dot(ro,rd),c=dot(ro,ro)-radius*radius,h=b*b-c;if(h<0)return float4(1e20,0,1,0);float t=-b-sqrt(h);if(t<1e-4)t=-b+sqrt(h);return t>1e-4?float4(t,normalize(ro+rd*t)):float4(1e20,0,1,0);}
inline float4 cylinderHit(float3 ro,float3 rd,float radius,float halfHeight,bool capped){float4 best=float4(1e20,0,1,0);float a=dot(rd.xz,rd.xz),b=dot(ro.xz,rd.xz),c=dot(ro.xz,ro.xz)-radius*radius;if(a>1e-8){float h=b*b-a*c;if(h>=0){float t=(-b-sqrt(h))/a;if(t<1e-4)t=(-b+sqrt(h))/a;float y=ro.y+rd.y*t;if(t>1e-4&&abs(y)<=halfHeight){float3 q=ro+rd*t;best=float4(t,normalize(float3(q.x,0,q.z)));}}}if(capped&&abs(rd.y)>1e-8){for(int side=-1;side<=1;side+=2){float t=(side*halfHeight-ro.y)/rd.y;float3 q=ro+rd*t;if(t>1e-4&&t<best.x&&dot(q.xz,q.xz)<=radius*radius)best=float4(t,0,side,0);}}return best;}
inline float4 bodyHit(float3 ro,float3 rd,BodyGPU body){float3 o=qinverse(body.orientation,ro-body.positionShape.xyz),r=qinverse(body.orientation,rd),d=body.dimensions.xyz;int shape=int(round(body.positionShape.w));float4 hit=float4(1e20,0,1,0);if(shape==0)hit=sphereHit(o,r,d.x);else if(shape==1){float2 q=boxHit(o,r,-d*.5f,d*.5f);float t=q.x>1e-4?q.x:q.y;if(t>1e-4&&q.x<=q.y){float3 point=o+r*t,n=float3(0);float3 ratio=abs(point/max(d*.5f,float3(1e-6)));if(ratio.x>=ratio.y&&ratio.x>=ratio.z)n.x=sign(point.x);else if(ratio.y>=ratio.z)n.y=sign(point.y);else n.z=sign(point.z);hit=float4(t,n);}}else if(shape==2){hit=cylinderHit(o,r,d.x,d.y*.5f,false);float4 a=sphereHit(o-float3(0,d.y*.5f,0),r,d.x),b=sphereHit(o+float3(0,d.y*.5f,0),r,d.x);if(a.x<hit.x)hit=a;if(b.x<hit.x)hit=b;}else hit=cylinderHit(o,r,d.x,d.y*.5f,true);hit.yzw=qrotate(body.orientation,hit.yzw);return hit;}

kernel void raymarch(device const float *volume [[buffer(0)]], constant Params &p [[buffer(1)]],device const BodyGPU*bodies[[buffer(2)]],
                     texture2d<float,access::write> output [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    if(any(gid>=uint2(output.get_width(),output.get_height())))return;
    float2 uv=(float2(gid)+0.5f)/float2(output.get_width(),output.get_height());float aspect=float(output.get_width())/float(output.get_height());
    float3 ro=p.cameraPosition.xyz,target=p.cameraTarget.xyz,forward=normalize(target-ro),right=normalize(cross(forward,float3(0,1,0))),up=cross(right,forward);
    // Metal drawable coordinates start at the top-left. Camera space uses +Y
    // as up, so screen-space Y must be inverted when constructing the ray.
    float2 ndc=float2(uv.x*2-1,1-uv.y*2);
    float3 rd=normalize(forward+right*(ndc.x*aspect*.72f)+up*(ndc.y*.72f));float3 size=p.containerDensity.xyz;
    float3 lo=float3(-size.x*.5f,0,-size.z*.5f),hi=float3(size.x*.5f,size.y,size.z*.5f);float2 hit=boxHit(ro,rd,lo,hi);
    float3 sky=mix(float3(.018,.027,.048),float3(.13,.20,.26),clamp(rd.y*.5f+.5f,0.0f,1.0f));float3 color=sky;float floorT=(-.025f-ro.y)/rd.y;if(floorT>0){float3 fp=ro+rd*floorT;float grid=(1-smoothstep(0.0f,.05f,min(abs(sin(fp.x*31.4159f)),abs(sin(fp.z*31.4159f)))));color=mix(color,float3(.035,.075,.07)+grid*.08f,.3f*exp(-length(fp.xz)*.7f));}
    float fluidEntry=1e20,fluidExit=1e20;
    if(hit.x<hit.y&&hit.y>0){float t=max(hit.x,0.0f),step=max(max(size.x/p.dimsDt.x,size.y/p.dimsDt.y),size.z/p.dimsDt.z)*.65f;float previous=field(volume,ro+rd*t,p);if(previous>=.5f)fluidEntry=t;
        for(int s=0;s<768&&t<hit.y;s++,t+=step){float value=field(volume,ro+rd*t,p);if(fluidEntry>1e19&&previous<.5f&&value>=.5f)fluidEntry=t-step*.5f;if(fluidEntry<1e19&&previous>=.5f&&value<.5f){fluidExit=t-step*.5f;break;}previous=value;}
        if(fluidEntry<1e19){if(fluidExit>1e19)fluidExit=hit.y;float3 pos=ro+rd*fluidEntry;float e=max(step*.75f,.001f);float3 n=normalize(float3(field(volume,pos+float3(e,0,0),p)-field(volume,pos-float3(e,0,0),p),field(volume,pos+float3(0,e,0),p)-field(volume,pos-float3(0,e,0),p),field(volume,pos+float3(0,0,e),p)-field(volume,pos-float3(0,0,e),p)));
            float fresnel=.0204f+.9796f*pow(1-clamp(abs(dot(n,-rd)),0.0f,1.0f),5.0f);float diffuse=.22f+.78f*max(dot(n,normalize(float3(-.4f,.9f,.25f))),0.0f);float thickness=max(0.0f,fluidExit-fluidEntry);float3 transmission=exp(-float3(.95f,.28f,.16f)*thickness),scatter=float3(.018f,.34f,.29f)*(1-transmission),reflected=mix(float3(.025,.07,.08),float3(.2,.42,.48),clamp(reflect(rd,n).y*.5f+.5f,0.0f,1.0f));color=mix(color*transmission+scatter,reflected,fresnel)*diffuse;color+=float3(.2,.75,.65)*pow(max(dot(reflect(rd,n),normalize(float3(-.5,.8,.25))),0.0f),64.0f);if(p.boundary.w>.5f){float lines=1-smoothstep(0.0f,.08f,min(abs(sin(pos.x/max(p.cellGravity.x,.001f)*3.14159f)),abs(sin(pos.z/max(p.cellGravity.z,.001f)*3.14159f))));color=mix(color,float3(.42,.96,.82),lines*.45f);}}
        if(p.physical.z>.5f){float3 entryPoint=ro+rd*max(hit.x,0.0f),center=(lo+hi)*.5f,q=abs((entryPoint-center)/max(size*.5f,float3(.001f)));float edge=max(max(min(q.x,q.y),min(q.x,q.z)),min(q.y,q.z));float edgeAlpha=smoothstep(.90f,.995f,edge),glassFresnel=pow(1-abs(dot(rd,normalize(entryPoint-center))),3.0f);color=mix(color,float3(.42,.78,.72),.025f+glassFresnel*.045f+edgeAlpha*.62f);}
    }
    float4 nearest=float4(1e20,0,1,0);int nearestIndex=-1;for(int b=0;b<int(p.boundary.z);b++){float4 candidate=bodyHit(ro,rd,bodies[b]);if(candidate.x<nearest.x){nearest=candidate;nearestIndex=b;}}
    if(nearestIndex>=0&&nearest.x<(fluidEntry>1e19?1e20:fluidExit)){float3 palette[4]={float3(.95,.63,.29),float3(.48,.66,.96),float3(.84,.42,.48),float3(.66,.52,.92)};int shape=int(round(bodies[nearestIndex].positionShape.w));float diffuse=.2f+.8f*max(dot(normalize(nearest.yzw),normalize(float3(-.45,.8,.3))),0.0f),rim=pow(1-max(dot(-rd,normalize(nearest.yzw)),0.0f),3.0f);float3 bodyColor=palette[shape]*diffuse+float3(.18,.42,.37)*rim;if(nearest.x>fluidEntry)bodyColor=bodyColor*float3(.35,.72,.7)+float3(.01,.11,.1);if(nearestIndex==int(round(p.physical.w)))bodyColor+=float3(.16,.5,.38)*(.25f+rim);color=bodyColor;}
    color=color/(color+1.0f);color=pow(color,float3(1/2.2f));output.write(float4(color,1),gid);
}
