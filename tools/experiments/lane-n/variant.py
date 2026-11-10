# Lane N experiment 1: generate compile-only variants of the particle move (and the other samplers)
# with the velocity / phi samples replaced by hardware-filtered taps through one brick-record lookup.
# usage: python3 variant.py <base.wgsl> <out.wgsl> <edit> [<edit> ...]
# edits: hwvel, hwdepth, nowalk, norigid, stub:<fn>:<body>, sub:<old>::<new>, skel (lane T skeleton set)
import sys

def span(text, fn):
    i = text.index('\nfn ' + fn + '(') + 1
    j = text.index('{', i)
    d = 0
    k = j
    while True:
        c = text[k]
        if c == '{':
            d += 1
        elif c == '}':
            d -= 1
            if d == 0:
                break
        k += 1
    return j, k

def stub(text, fn, body):
    j, k = span(text, fn)
    return text[:j] + '{' + body + '}' + text[k + 1:]

# One filtering sampler for the atlas taps (a new binding: the move binds no sampler today).
SAMPLER = '@group(1) @binding(120) var nlSamp:sampler;\n'
# One brick-record lookup: a vec4 per 4^3 tile (xyz: the brick's atlas texel origin less the
# tile's sample origin, w: cells per atlas texel, 1 for an h brick, 0.25 for a 4h region).
# Stands in for the record an apron atlas keeps; one dependent storage load per sample.
REC = 'let t=vec3u(clamp(floor(p),vec3f(0.0),vec3f(4095.0)))>>vec3u(2u);let rec=umSolidRecord[(t.x+64u*(t.y+64u*t.z))&0xfffffu];'
HWVEL = (REC + 'let inv=1.0/vec3f(textureDimensions(unitHalo));var v=vec3f(0.0);'
         'for(var a=0u;a<3u;a++){var off=vec3f(0.5);off[a]=1.0;'
         'v[a]=textureSampleLevel(unitHalo,nlSamp,(rec.xyz+p*rec.w-off+0.5)*inv,0.0)[a];}return v;')
HWDEPTH = (REC.replace('(t.x+64u*(t.y+64u*t.z))', '(t.x+64u*(t.y+64u*t.z)+262144u)') +
           'let inv=1.0/vec3f(textureDimensions(phi));'
           'return textureSampleLevel(phi,nlSamp,(rec.xyz+p*rec.w+0.5)*inv,0.0).x/rec.w;')

SK = {'nbMoveSample': 'return textureLoad(coarseVelocity,vec3i(p),0).xyz;',
      'nbMoveDepth': 'return textureLoad(coarseVelocity,vec3i(p),0).w;',
      'nbParticleBlocked': 'return q.x>q.y;', 'nbSparseAge': 'return age+p.x;',
      'nbSeparation': 'return q*m;', 'nbSurfaceShift': 'return d+q;', 'nbMergeRun': 'return record;',
      'nbMergeAbsorbed': 'return i>start+stride;',
      'nbRigidContact': 'return NBRigidContact(point,velocity);',
      'nbFinish': 'var p=record;p.position=vec4f(endpoint+end+start,p.position.w);return NbMoved(p,select(0u,1u,ballistic||uncut));',
      'gridSupported': 'return p.z>p.x;', 'nbTraceEndpointIn': 'return p+vec3f(home)*inset;',
      'umPointKind': 'return u32(p.x);', 'nbTileThin': 'return q.x>1.0;', 'nbMoveHeat': 'return p.z;',
      'nbMoveCold': 'return q.y>2.0;', 'nbHullUncut': 'return nbHullLo.x<nbHullHi.y;'}

def apply(src, edits):
    need_sampler = False
    for e in edits:
        if e == 'hwvel':
            src = stub(src, 'nbMoveSample', HWVEL); need_sampler = True
        elif e == 'hwdepth':
            src = stub(src, 'nbMoveDepth', HWDEPTH); need_sampler = True
        elif e == 'hwseedvel':  # the seed's / transfer's general sampler as one hardware sample
            src = stub(src, 'sampleVelocity', HWVEL.replace('unitHalo', 'unitVelocity')); need_sampler = True
        elif e == 'hwseeddepth':
            src = stub(src, 'particleDepthIn', HWDEPTH); need_sampler = True
        elif e == 'hwsurfvel':  # the surface trace's general and fast samplers as one hardware sample each
            src = stub(src, 'umSampleVelocity', HWVEL); src = stub(src, 'umFastVelocity', HWVEL); need_sampler = True
        elif e == 'hwsurfphi':  # the departure phi (trilinear cell and cubic) as one hardware tap
            src = stub(src, 'umVertexCell', 'var c:UMVertexCell;c.origin=vec3u(p);c.width=1u;c.value=textureSampleLevel(phi,nlSamp,(p+0.5)/vec3f(textureDimensions(phi)),0.0).x;return c;')
            src = stub(src, 'umCubicPhi', 'return textureSampleLevel(phi,nlSamp,(q+0.5)/vec3f(textureDimensions(phi)),0.0).x+cell.value;'); need_sampler = True
        elif e == 'nowalk':
            # nbFinish keeps arrival, heat and the split; the solid walk and its contacts move out.
            j, k = span(src, 'nbFinish')
            body = src[j:k + 1]
            a = body.index('if(!umSolidEnabled()||uncut){q=endpoint;}else{')
            b = body.index('{', a + len('if(!umSolidEnabled()||uncut){q=endpoint;}else'))
            d = 0; m = b
            while True:
                c = body[m]
                if c == '{': d += 1
                elif c == '}':
                    d -= 1
                    if d == 0: break
                m += 1
            body = body[:a] + 'q=endpoint;' + body[m + 1:]
            src = src[:j] + body + src[k + 1:]
        elif e == 'norigid':
            src = stub(src, 'nbRigidContact', SK['nbRigidContact'])
        elif e.startswith('skel'):  # lane T's skeleton: every section stubbed except those listed after 'skel:'
            keep = set(e.split(':')[1].split(',')) if ':' in e else set()
            for fn, body in SK.items():
                if fn not in keep:
                    src = stub(src, fn, body)
        elif e.startswith('stub:'):
            _, fn, body = e.split(':', 2); src = stub(src, fn, body)
        elif e.startswith('sub:'):
            a, b = e[4:].replace('\\n', '\n').split('::', 1); assert src.count(a) >= 1, a; src = src.replace(a, b)
        else:
            raise SystemExit('unknown edit ' + e)
    if need_sampler:
        i = src.index('\n@group(') + 1
        src = src[:i] + SAMPLER + src[i:]
    return src

if __name__ == '__main__':
    base, out = sys.argv[1], sys.argv[2]
    open(out, 'w').write(apply(open(base).read(), sys.argv[3:]))
