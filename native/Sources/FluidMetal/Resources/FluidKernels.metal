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

struct BodyGPU { float4 positionShape; float4 dimensions; float4 orientation; float4 linearVelocity; float4 angularVelocity; };

inline int3 dimensions(constant Params &p) { return int3(p.dimsDt.xyz); }
inline int index3(int3 q, int3 d) { return q.x + d.x * (q.y + d.y * q.z); }
inline bool valid(int3 q, int3 d) { return all(q >= 0) && all(q < d); }
inline float volumeAt(device const float *v, int3 q, int3 d) { return valid(q, d) ? v[index3(q, d)] : 0.0f; }
inline float3 velocityAt(device const float4 *v, int3 q, int3 d) { return valid(q, d) ? v[index3(q, d)].xyz : float3(0.0f); }

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
inline float rawFlux(device const float4*vel,device const float*vol,int3 id,int axis,float dt,int3 d,float3 h){if(!valid(id,d))return 0;int3 o=axis==0?int3(1,0,0):(axis==1?int3(0,1,0):int3(0,0,1));float speed=velocityAt(vel,id,d)[axis];return dt/h[axis]*speed*(speed>=0?volumeAt(vol,id,d):volumeAt(vol,id+o,d));}
inline float outward(device const float4*vel,device const float*vol,int3 id,float dt,int3 d,float3 h){return max(rawFlux(vel,vol,id,0,dt,d,h),0.0f)+max(-rawFlux(vel,vol,id-int3(1,0,0),0,dt,d,h),0.0f)+max(rawFlux(vel,vol,id,1,dt,d,h),0.0f)+max(-rawFlux(vel,vol,id-int3(0,1,0),1,dt,d,h),0.0f)+max(rawFlux(vel,vol,id,2,dt,d,h),0.0f)+max(-rawFlux(vel,vol,id-int3(0,0,1),2,dt,d,h),0.0f);}
inline float inward(device const float4*vel,device const float*vol,int3 id,float dt,int3 d,float3 h){return max(-rawFlux(vel,vol,id,0,dt,d,h),0.0f)+max(rawFlux(vel,vol,id-int3(1,0,0),0,dt,d,h),0.0f)+max(-rawFlux(vel,vol,id,1,dt,d,h),0.0f)+max(rawFlux(vel,vol,id-int3(0,1,0),1,dt,d,h),0.0f)+max(-rawFlux(vel,vol,id,2,dt,d,h),0.0f)+max(rawFlux(vel,vol,id-int3(0,0,1),2,dt,d,h),0.0f);}
inline float4 auxiliaryAt(device const float4*a,int3 q,int3 d){return valid(q,d)?a[index3(q,d)]:float4(0,0,0,0);}
inline float limitedFlux(device const float4*vel,device const float*vol,device const float4*aux,int3 id,int axis,float dt,int3 d,float3 h){int3 o=axis==0?int3(1,0,0):(axis==1?int3(0,1,0):int3(0,0,1));float f=rawFlux(vel,vol,id,axis,dt,d,h);int3 donor=f>=0?id:id+o,receiver=f>=0?id+o:id;return f*min(auxiliaryAt(aux,donor,d).x,auxiliaryAt(aux,receiver,d).y);}

kernel void buildAuxiliary(device const float4*velocity[[buffer(0)]],device const float*volume[[buffer(1)]],device float4*auxiliary[[buffer(2)]],constant Params&p[[buffer(3)]],uint3 gid[[thread_position_in_grid]]){int3 d=dimensions(p),id=int3(gid);if(!valid(id,d))return;float dt=p.dimsDt.w;float3 h=p.cellGravity.xyz;float phi=volumeAt(volume,id,d);float donor=min(1.0f,phi/max(outward(velocity,volume,id,dt,d,h),1e-9f)),receiver=min(1.0f,(1-phi)/max(inward(velocity,volume,id,dt,d,h),1e-9f));float k=clamp(curvature(volume,id,d,h),-2/min(min(h.x,h.y),h.z),2/min(min(h.x,h.y),h.z));auxiliary[index3(id,d)]=float4(donor,receiver,k,0);}

kernel void advect(device const float4 *velocityIn [[buffer(0)]],
                   device float4 *velocityOut [[buffer(1)]],
                   device const float *volumeIn [[buffer(2)]],
                   device float *volumeOut [[buffer(3)]],
                   device const float4 *auxiliary [[buffer(4)]],
                   constant Params &p [[buffer(5)]], uint3 gid [[thread_position_in_grid]]) {
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
    float fxp=limitedFlux(velocityIn,volumeIn,auxiliary,id,0,dt,d,h),fxm=limitedFlux(velocityIn,volumeIn,auxiliary,id-int3(1,0,0),0,dt,d,h);
    float fyp=limitedFlux(velocityIn,volumeIn,auxiliary,id,1,dt,d,h),fym=limitedFlux(velocityIn,volumeIn,auxiliary,id-int3(0,1,0),1,dt,d,h);
    float fzp=limitedFlux(velocityIn,volumeIn,auxiliary,id,2,dt,d,h),fzm=limitedFlux(velocityIn,volumeIn,auxiliary,id-int3(0,0,1),2,dt,d,h);
    volumeOut[i]=phi-(fxp-fxm+fyp-fym+fzp-fzm);
}

kernel void jacobi(device const float4 *velocity [[buffer(0)]],
                   device const float *volume [[buffer(1)]],
                   device const float *pressureIn [[buffer(2)]],
                   device float *pressureOut [[buffer(3)]],
                   constant Params &p [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
    int3 d=dimensions(p), id=int3(gid); if(!valid(id,d)) return; int i=index3(id,d);
    if(volume[i] < 0.5f){pressureOut[i]=0;return;}
    float3 h=p.cellGravity.xyz;
    float divergence=(velocityAt(velocity,id,d).x-velocityAt(velocity,id-int3(1,0,0),d).x)/h.x
                    +(velocityAt(velocity,id,d).y-velocityAt(velocity,id-int3(0,1,0),d).y)/h.y
                    +(velocityAt(velocity,id,d).z-velocityAt(velocity,id-int3(0,0,1),d).z)/h.z;
    const int3 offsets[6]={int3(-1,0,0),int3(1,0,0),int3(0,-1,0),int3(0,1,0),int3(0,0,-1),int3(0,0,1)};
    float sum=0, diagonal=0;
    for(int n=0;n<6;n++){int3 q=id+offsets[n];if(!valid(q,d))continue;float invH2=n<2?1/(h.x*h.x):(n<4?1/(h.y*h.y):1/(h.z*h.z));float coefficient=volumeAt(volume,q,d)>=.5f?invH2:invH2/interfaceFraction(volume[i],volumeAt(volume,q,d));diagonal+=coefficient;if(volumeAt(volume,q,d)>=0.5f)sum+=pressureIn[index3(q,d)]*coefficient;}
    float rhs=p.containerDensity.w*divergence/p.dimsDt.w;
    pressureOut[i]=mix(pressureIn[i],(sum-rhs)/max(diagonal,1e-8f),0.8f);
}

kernel void projectAndCommit(device const float4 *velocityIn [[buffer(0)]],
                             device float4 *velocityOut [[buffer(1)]],
                             device const float *pressure [[buffer(2)]],
                             device const float *volumeIn [[buffer(3)]],
                             device float *volumeOut [[buffer(4)]],
                             constant Params &p [[buffer(5)]], uint3 gid [[thread_position_in_grid]]) {
    int3 d=dimensions(p), id=int3(gid);if(!valid(id,d))return;int i=index3(id,d);float3 h=p.cellGravity.xyz;float scale=p.dimsDt.w/p.containerDensity.w;
    float3 v=velocityIn[i].xyz;float p0=volumeIn[i]>=0.5f?pressure[i]:0;
    int3 ex=id+int3(1,0,0),ey=id+int3(0,1,0),ez=id+int3(0,0,1);
    bool l0=volumeIn[i]>=.5f,lx=volumeAt(volumeIn,ex,d)>=.5f,ly=volumeAt(volumeIn,ey,d)>=.5f,lz=volumeAt(volumeIn,ez,d)>=.5f;
    if(id.x==d.x-1)v.x=0;else if(l0||lx){float theta=l0==lx?1:interfaceFraction(l0?volumeIn[i]:volumeAt(volumeIn,ex,d),l0?volumeAt(volumeIn,ex,d):volumeIn[i]);v.x-=scale*((lx?pressure[index3(ex,d)]:0)-p0)/(h.x*theta);}else v.x=0;
    if(id.y==d.y-1)v.y=0;else if(l0||ly){float theta=l0==ly?1:interfaceFraction(l0?volumeIn[i]:volumeAt(volumeIn,ey,d),l0?volumeAt(volumeIn,ey,d):volumeIn[i]);v.y-=scale*((ly?pressure[index3(ey,d)]:0)-p0)/(h.y*theta);}else v.y=0;
    if(id.z==d.z-1)v.z=0;else if(l0||lz){float theta=l0==lz?1:interfaceFraction(l0?volumeIn[i]:volumeAt(volumeIn,ez,d),l0?volumeAt(volumeIn,ez,d):volumeIn[i]);v.z-=scale*((lz?pressure[index3(ez,d)]:0)-p0)/(h.z*theta);}else v.z=0;
    velocityOut[i]=float4(v,0);volumeOut[i]=volumeIn[i];
}

inline float3 qrotate(float4 q,float3 v){float3 uv=cross(q.yzw,v),uuv=cross(q.yzw,uv);return v+2*(q.x*uv+uuv);}
inline float3 qinverse(float4 q,float3 v){return qrotate(float4(q.x,-q.yzw),v);}
inline bool insideBody(BodyGPU b,float3 world){float3 q=qinverse(b.orientation,world-b.positionShape.xyz),d=b.dimensions.xyz;int shape=int(round(b.positionShape.w));if(shape==0)return length(q)<=d.x;if(shape==1)return all(abs(q)<=d*.5f);if(shape==2){float y=clamp(q.y,-d.y*.5f,d.y*.5f);return length(float3(q.x,q.y-y,q.z))<=d.x;}return q.x*q.x+q.z*q.z<=d.x*d.x&&abs(q.y)<=d.y*.5f;}

kernel void coupleRigid(device float4 *velocity [[buffer(0)]],device const float *volume [[buffer(1)]],device const BodyGPU*bodies [[buffer(2)]],device atomic_int*exchange [[buffer(3)]],constant Params&p [[buffer(4)]],uint3 gid[[thread_position_in_grid]]){
    int3 d=dimensions(p),id=int3(gid);if(!valid(id,d))return;int i=index3(id,d);float phi=volume[i];if(phi<=0)return;float3 h=p.cellGravity.xyz;float3 world=float3(-p.containerDensity.x*.5f+(id.x+.5f)*h.x,(id.y+.5f)*h.y,-p.containerDensity.z*.5f+(id.z+.5f)*h.z);float3 v=velocity[i].xyz;
    for(int b=0;b<int(p.boundary.z);b++){BodyGPU body=bodies[b];if(!insideBody(body,world))continue;float3 arm=world-body.positionShape.xyz,solid=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);float mass=p.containerDensity.w*h.x*h.y*h.z*phi;float3 impulse=mass*(solid-v)*clamp(45*p.dimsDt.w,0.0f,1.0f);v+=impulse/max(mass,1e-8f);float3 reaction=-impulse,t=cross(arm,reaction);int base=b*8;atomic_fetch_add_explicit(&exchange[base],int(round(reaction.x*1e6f)),memory_order_relaxed);atomic_fetch_add_explicit(&exchange[base+1],int(round(reaction.y*1e6f)),memory_order_relaxed);atomic_fetch_add_explicit(&exchange[base+2],int(round(reaction.z*1e6f)),memory_order_relaxed);atomic_fetch_add_explicit(&exchange[base+3],int(round(t.x*1e6f)),memory_order_relaxed);atomic_fetch_add_explicit(&exchange[base+4],int(round(t.y*1e6f)),memory_order_relaxed);atomic_fetch_add_explicit(&exchange[base+5],int(round(t.z*1e6f)),memory_order_relaxed);atomic_fetch_add_explicit(&exchange[base+6],int(round(phi*65536)),memory_order_relaxed);break;}velocity[i]=float4(v,0);
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
