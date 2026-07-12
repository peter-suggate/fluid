#include <metal_stdlib>
using namespace metal;

struct Params {
    float4 dimsDt;
    float4 cellGravity;
    float4 containerDensity;
    float4 cameraPosition;
    float4 cameraTarget;
    float4 viewport;
};

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

kernel void advect(device const float4 *velocityIn [[buffer(0)]],
                   device float4 *velocityOut [[buffer(1)]],
                   device const float *volumeIn [[buffer(2)]],
                   device float *volumeOut [[buffer(3)]],
                   constant Params &p [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
    int3 d = dimensions(p), id = int3(gid); if (!valid(id,d)) return;
    int i = index3(id,d); float dt = p.dimsDt.w; float3 h = p.cellGravity.xyz;
    float3 position = float3(id) + 0.5f;
    float3 first = sampleVelocity(velocityIn, position, d);
    float3 midpoint = position - 0.5f * dt * first / h;
    float3 transported = sampleVelocity(velocityIn, position - dt * sampleVelocity(velocityIn, midpoint, d) / h, d);
    if (volumeAt(volumeIn,id,d) > 0.001f || volumeAt(volumeIn,id+int3(0,1,0),d) > 0.001f) transported.y += p.cellGravity.w * dt;
    if (id.x == d.x-1) transported.x = 0; if (id.y == d.y-1) transported.y = 0; if (id.z == d.z-1) transported.z = 0;
    velocityOut[i] = float4(transported, 0);

    // Bounded conservative donor-cell VOF update. Shared with the browser path's
    // governing flux convention, but expressed over linear buffers for Apple GPUs.
    float phi = volumeAt(volumeIn,id,d);
    float3 vp = velocityAt(velocityIn,id,d);
    float3 vm = float3(velocityAt(velocityIn,id-int3(1,0,0),d).x,
                       velocityAt(velocityIn,id-int3(0,1,0),d).y,
                       velocityAt(velocityIn,id-int3(0,0,1),d).z);
    float fxp = vp.x * (vp.x >= 0 ? phi : volumeAt(volumeIn,id+int3(1,0,0),d));
    float fxm = vm.x * (vm.x >= 0 ? volumeAt(volumeIn,id-int3(1,0,0),d) : phi);
    float fyp = vp.y * (vp.y >= 0 ? phi : volumeAt(volumeIn,id+int3(0,1,0),d));
    float fym = vm.y * (vm.y >= 0 ? volumeAt(volumeIn,id-int3(0,1,0),d) : phi);
    float fzp = vp.z * (vp.z >= 0 ? phi : volumeAt(volumeIn,id+int3(0,0,1),d));
    float fzm = vm.z * (vm.z >= 0 ? volumeAt(volumeIn,id-int3(0,0,1),d) : phi);
    volumeOut[i] = clamp(phi - dt * ((fxp-fxm)/h.x + (fyp-fym)/h.y + (fzp-fzm)/h.z), 0.0f, 1.0f);
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
    for(int n=0;n<6;n++){int3 q=id+offsets[n];if(!valid(q,d))continue;float invH2=n<2?1/(h.x*h.x):(n<4?1/(h.y*h.y):1/(h.z*h.z));diagonal+=invH2;if(volumeAt(volume,q,d)>=0.5f)sum+=pressureIn[index3(q,d)]*invH2;}
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
    if(id.x==d.x-1)v.x=0;else v.x-=scale*((volumeAt(volumeIn,ex,d)>=0.5f?pressure[index3(ex,d)]:0)-p0)/h.x;
    if(id.y==d.y-1)v.y=0;else v.y-=scale*((volumeAt(volumeIn,ey,d)>=0.5f?pressure[index3(ey,d)]:0)-p0)/h.y;
    if(id.z==d.z-1)v.z=0;else v.z-=scale*((volumeAt(volumeIn,ez,d)>=0.5f?pressure[index3(ez,d)]:0)-p0)/h.z;
    velocityOut[i]=float4(v,0);volumeOut[i]=volumeIn[i];
}

inline float2 boxHit(float3 ro,float3 rd,float3 lo,float3 hi){float3 inv=1/rd;float3 a=(lo-ro)*inv,b=(hi-ro)*inv;float3 n=min(a,b),f=max(a,b);return float2(max(max(n.x,n.y),n.z),min(min(f.x,f.y),f.z));}
inline float field(device const float *v,float3 world,constant Params&p){float3 size=p.containerDensity.xyz;float3 uv=(world+float3(size.x*.5f,0,size.z*.5f))/size;return sampleVolume(v,uv*p.dimsDt.xyz,dimensions(p));}

kernel void raymarch(device const float *volume [[buffer(0)]], constant Params &p [[buffer(1)]],
                     texture2d<float,access::write> output [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    if(any(gid>=uint2(output.get_width(),output.get_height())))return;
    float2 uv=(float2(gid)+0.5f)/float2(output.get_width(),output.get_height());float aspect=float(output.get_width())/float(output.get_height());
    float3 ro=p.cameraPosition.xyz,target=p.cameraTarget.xyz,forward=normalize(target-ro),right=normalize(cross(forward,float3(0,1,0))),up=cross(right,forward);
    // Metal drawable coordinates start at the top-left. Camera space uses +Y
    // as up, so screen-space Y must be inverted when constructing the ray.
    float2 ndc=float2(uv.x*2-1,1-uv.y*2);
    float3 rd=normalize(forward+right*(ndc.x*aspect*.72f)+up*(ndc.y*.72f));float3 size=p.containerDensity.xyz;
    float2 hit=boxHit(ro,rd,float3(-size.x*.5f,0,-size.z*.5f),float3(size.x*.5f,size.y,size.z*.5f));
    float3 sky=mix(float3(.018,.027,.048),float3(.13,.20,.26),clamp(rd.y*.5f+.5f,0.0f,1.0f));float3 color=sky;
    if(hit.x<hit.y&&hit.y>0){float t=max(hit.x,0.0f),step=max(max(size.x/p.dimsDt.x,size.y/p.dimsDt.y),size.z/p.dimsDt.z)*.7f;float previous=field(volume,ro+rd*t,p);float entry=-1,exit=-1;
        for(int s=0;s<768&&t<hit.y;s++,t+=step){float value=field(volume,ro+rd*t,p);if(entry<0&&previous<.5f&&value>=.5f)entry=t;if(entry>=0&&previous>=.5f&&value<.5f){exit=t;break;}previous=value;}
        if(entry>=0){if(exit<0)exit=hit.y;float3 pos=ro+rd*entry;float e=max(step*.75f,.001f);float3 n=normalize(float3(field(volume,pos+float3(e,0,0),p)-field(volume,pos-float3(e,0,0),p),field(volume,pos+float3(0,e,0),p)-field(volume,pos-float3(0,e,0),p),field(volume,pos+float3(0,0,e),p)-field(volume,pos-float3(0,0,e),p)));
            float fresnel=.02f+.98f*pow(1-clamp(abs(dot(n,-rd)),0.0f,1.0f),5.0f);float diffuse=.25f+.75f*max(dot(n,normalize(float3(-.4f,.9f,.25f))),0.0f);float thickness=max(0.0f,exit-entry);float3 water=float3(.025,.38,.48)*diffuse;float3 absorption=exp(-float3(2.2f,.48f,.2f)*thickness);color=mix(water*absorption+float3(.01,.08,.1),sky,fresnel);}
    }
    color=color/(color+1.0f);color=pow(color,float3(1/2.2f));output.write(float4(color,1),gid);
}
