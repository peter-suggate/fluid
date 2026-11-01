"""Rock shelf shared by authoring/export; mirrored in the runtime heightfield."""
import json
import math
from pathlib import Path

LAYOUT = json.loads((Path(__file__).resolve().parents[2]/
    'lib/core/imported-assets/atlantic-lighthouse-rock-layout.json').read_text())


def terrain_height(x, y):
    z = -y
    height = .15
    relief = 0
    for r in LAYOUT:
        dx, dz = x-r['x'], z-r['z']
        c, s = math.cos(r['yaw']), math.sin(r['yaw'])
        u, v = (c*dx+s*dz)/r['rx'], (-s*dx+c*dz)/r['rz']
        q = max(abs(u), abs(v), abs(u+v)*.65, abs(u-v)*.65)
        if 'relief' in r:
            relief = max(relief, max(0, min(1, (1-q)/(1-r['core']))) *
                         max(0, r['relief']+dx*r['tiltX']+dz*r['tiltZ']))
            continue
        top = r['height'] + dx*r['tiltX'] + dz*r['tiltZ']
        fall = max(0, min(1, (q-r['core'])/(1-r['core'])))
        if r['height'] < 3.5:
            # Tidal boulders have full shoulders and rounded crowns cut by
            # angular bedding planes, rather than pointed pyramid profiles.
            profile = max(0, 1-fall*fall)**1.2
            block = .15 + (top-.15)*profile
        else:
            block = top-max(0, q-r['core'])*r['height']/(1-r['core'])
        height = max(height, block)
    height += .07*math.sin(x*3.1+z*1.7)*math.sin(z*2.3-x*.4)*min(1, max(0, height-.15))
    # Decimetre fractured bedding, confined above the existing waterline.
    dry = min(1, max(0, (height-1.8)/1.4))
    seam = abs(math.sin(x*2.7+z*.83+.75*math.sin(z*2.1)+.38*math.sin(x*1.3)))
    cross = abs(math.sin(z*3.4-x*.61+.6*math.sin(x*2.4)))
    # Intersecting joints cut the broad slab into angular plates. A finer
    # bedding layer breaks their faces; all three bands are actual geometry.
    detail = .18*math.sin(x*2.3+z*.8+1.7*math.sin(z*1.5))*math.sin(z*2.1-x*.4+.8*math.sin(x*.7))
    detail += .025*math.sin(x*11+z*7)*math.sin(z*9-x*3)
    detail -= .16*max(0, 1-seam/.32) + .11*max(0, 1-cross/.26)
    height += dry*(detail+relief)
    r = math.hypot(x-7, z)
    footing = max(0, min(1, (2.2-r)/.45))
    footing = footing*footing*(3-2*footing)
    return height*(1-footing) + 5.8*footing
