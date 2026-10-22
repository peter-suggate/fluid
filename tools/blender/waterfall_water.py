"""Art-directed water geometry for the forest waterfall still.

No simulation cache or external textures are needed. ``build_water`` adds
editable, named geometry to the current scene, without changing the camera.
"""
import math
import random

import bpy
from mathutils import Vector


def _collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def _mesh(name, verts, faces, col, material, smooth=True):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    mesh.materials.append(material)
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    return obj


def _principled(name, color, roughness, transmission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    p = mat.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Roughness'].default_value = roughness
    p.inputs['IOR'].default_value = 1.333
    p.inputs['Transmission Weight'].default_value = transmission
    return mat, p


def _materials():
    water, p = _principled('Water • clear mountain spring / IOR 1.333', (.76, .93, .89), .075, 1)
    n, l = water.node_tree.nodes, water.node_tree.links
    tex = n.new('ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value = 2.8
    tex.inputs['Detail'].default_value = 3.5
    tex.inputs['Roughness'].default_value = .67
    coord = n.new('ShaderNodeTexCoord')
    l.new(coord.outputs['Object'], tex.inputs['Vector'])
    bump = n.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = .28
    bump.inputs['Distance'].default_value = .026
    l.new(tex.outputs['Fac'], bump.inputs['Height'])
    l.new(bump.outputs['Normal'], p.inputs['Normal'])
    absorb = n.new('ShaderNodeVolumeAbsorption')
    absorb.inputs['Color'].default_value = (.33, .79, .67, 1)
    absorb.inputs['Density'].default_value = .11
    l.new(absorb.outputs['Volume'], n.get('Material Output').inputs['Volume'])

    sheet, p = _principled('Falls • aerated translucent silver ribbons', (.74, .86, .88), .22, .47)
    n, l = sheet.node_tree.nodes, sheet.node_tree.links
    coord = n.new('ShaderNodeTexCoord')
    mapping = n.new('ShaderNodeVectorMath')
    mapping.operation = 'MULTIPLY'
    mapping.inputs[1].default_value = (8, 7, 1.1)
    l.new(coord.outputs['Object'], mapping.inputs[0])
    tex = n.new('ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value = 1.8
    tex.inputs['Detail'].default_value = 3.8
    tex.inputs['Roughness'].default_value = .7
    l.new(mapping.outputs['Vector'], tex.inputs['Vector'])
    ramp = n.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = .25
    ramp.color_ramp.elements[0].color = (.21, .44, .43, 1)
    ramp.color_ramp.elements[1].position = .66
    ramp.color_ramp.elements[1].color = (.94, .98, 1, 1)
    l.new(tex.outputs['Fac'], ramp.inputs[0])
    l.new(ramp.outputs['Color'], p.inputs['Base Color'])
    bump = n.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = .46
    bump.inputs['Distance'].default_value = .075
    l.new(tex.outputs['Fac'], bump.inputs['Height'])
    l.new(bump.outputs['Normal'], p.inputs['Normal'])

    foam, p = _principled('Foam • microbubbles, ivory in sunlight', (.92, .97, .95), .49, .08)
    p.inputs['Subsurface Weight'].default_value = .12
    n, l = foam.node_tree.nodes, foam.node_tree.links
    tex = n.new('ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value = 28
    tex.inputs['Detail'].default_value = 2
    coord = n.new('ShaderNodeTexCoord')
    l.new(coord.outputs['Object'], tex.inputs['Vector'])
    bump = n.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = .27
    bump.inputs['Distance'].default_value = .018
    l.new(tex.outputs['Fac'], bump.inputs['Height'])
    l.new(bump.outputs['Normal'], p.inputs['Normal'])

    lace = foam.copy()
    lace.name = 'Foam • lace-like broken surface film'
    n, l = lace.node_tree.nodes, lace.node_tree.links
    coord = n.new('ShaderNodeTexCoord')
    tex = n.new('ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value = 11
    tex.inputs['Detail'].default_value = 3
    tex.inputs['Roughness'].default_value = .68
    l.new(coord.outputs['Object'], tex.inputs['Vector'])
    ramp = n.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = .43
    ramp.color_ramp.elements[1].position = .59
    l.new(tex.outputs['Fac'], ramp.inputs[0])
    trans = n.new('ShaderNodeBsdfTransparent')
    mix = n.new('ShaderNodeMixShader')
    l.new(ramp.outputs['Color'], mix.inputs[0])
    l.new(trans.outputs[0], mix.inputs[1])
    l.new(n.get('Principled BSDF').outputs[0], mix.inputs[2])
    l.new(mix.outputs[0], n.get('Material Output').inputs['Surface'])
    return water, sheet, foam, lace


def _surface_z(x, y):
    r = math.sqrt((x + 1.1) ** 2 + ((y - 5.2) * 1.05) ** 2)
    return .10 + .019 * math.sin(x * 1.6 + y * 2.5) + .010 * math.sin(x * 4.2 - y * 3.1) + .032 * math.sin(r * 8) * math.exp(-r / 8)


def _pool(col, material):
    # A closed lens gives Cycles genuine water depth and absorption.
    verts = [(0, -2, _surface_z(0, -2))]
    faces = []
    segments, rings = 240, 100
    for j in range(1, rings + 1):
        r = j / rings
        for i in range(segments):
            a = i / segments * math.tau
            x, y = 10.92 * r * math.cos(a), -2 + 10.4 * r * math.sin(a)
            verts.append((x, y, _surface_z(x, y)))
    for i in range(segments):
        faces.append((0, 1 + i, 1 + (i + 1) % segments))
    for j in range(rings - 1):
        base, nxt = 1 + j * segments, 1 + (j + 1) * segments
        for i in range(segments):
            k = (i + 1) % segments
            faces.append((base + i, nxt + i, nxt + k, base + k))
    bottom = len(verts)
    for i in range(segments):
        a = i / segments * math.tau
        verts.append((10.92 * math.cos(a), -2 + 10.4 * math.sin(a), -2.5))
    edge = 1 + (rings - 1) * segments
    for i in range(segments):
        k = (i + 1) % segments
        faces.append((edge + i, bottom + i, bottom + k, edge + k))
    faces.append(tuple(reversed(range(bottom, bottom + segments))))
    _mesh('Pool | rippled clear water with 2.6 m optical depth', verts, faces, col, material)


def _upstream(col, material):
    # A shallow feeder links the forest plateau to the visible lip, so the
    # falling sheets read as one continuous watercourse.
    verts, faces = [], []
    rows, across = 44, 24
    for j in range(rows + 1):
        t = j / rows
        y = 7.74 + 3.26 * t
        center = -1 + .07 * math.sin(t * 4)
        half_width = 2.02 - .30 * t + .06 * math.sin(t * 11)
        for i in range(across + 1):
            u = 2 * i / across - 1
            x = center + half_width * u
            z = 19.05 + .5 * t ** .65
            z += .017 * math.sin(x * 14 + y * 8) * math.sin(math.pi * t)
            verts.append((x, y, z))
    for j in range(rows):
        for i in range(across):
            a = j * (across + 1) + i
            faces.append((a, a + 1, a + across + 2, a + across + 1))
    obj = _mesh('Cascade | shallow feeder over the high stone lip', verts, faces, col, material)
    solid = obj.modifiers.new('Shallow water over the lip', 'SOLIDIFY')
    solid.thickness = .065


def _fall_position(t, x, seed):
    z = 19.05 * (1 - t) + .16 * t
    # Water peels from the lip, meets two irregular shoulders, and fans out.
    y = 7.74 - 1.55 * min(1, t * 5) - 1.25 * t
    upper_fan = math.exp(-((z - 10.8) / 1.5) ** 2)
    lower_fan = math.exp(-((z - 3.8) / 1.4) ** 2)
    y -= .43 * upper_fan + .51 * lower_fan
    x += (x + 1) * (.16 * upper_fan + .24 * lower_fan + .1 * t)
    x += .18 * math.sin(t * 10 + seed) + .09 * math.sin(t * 38 + seed * 2)
    y += .12 * math.sin(t * 35 + seed) + .06 * math.sin(t * 110 + seed)
    return x, y, z


def _ribbon(name, col, material, center, width, seed, begin=0, end=1):
    rng = random.Random(seed)
    verts, faces = [], []
    rows, across = max(25, int(180 * (end - begin))), 8
    phase = rng.uniform(0, math.tau)
    for j in range(rows + 1):
        t = begin + (end - begin) * j / rows
        x, y, z = _fall_position(t, center, phase)
        swelling = 1 + .32 * math.sin(t * 26 + phase) + .16 * math.sin(t * 67 + phase)
        swelling *= 1 + .23 * t
        for i in range(across + 1):
            u = i / across * 2 - 1
            xx = x + u * width * .5 * swelling
            yy = y + .08 * math.sin(u * 13 + t * 36 + phase) + .05 * math.sin(u * 43 - t * 110)
            zz = z + .065 * math.sin(u * 19 + t * 85 + phase)
            verts.append((xx, yy, zz))
    for j in range(rows):
        for i in range(across):
            a = j * (across + 1) + i
            faces.append((a, a + 1, a + across + 2, a + across + 1))
    obj = _mesh(name, verts, faces, col, material)
    solid = obj.modifiers.new('Thin flowing sheet', 'SOLIDIFY')
    solid.thickness = .012
    return obj


def _tube(verts, faces, points, radii, sides=5):
    start = len(verts)
    for i, p in enumerate(points):
        direction = Vector(points[min(i + 1, len(points) - 1)]) - Vector(points[max(0, i - 1)])
        direction.normalize()
        ref = Vector((0, 0, 1)) if abs(direction.z) < .9 else Vector((1, 0, 0))
        u = direction.cross(ref).normalized()
        v = direction.cross(u).normalized()
        for k in range(sides):
            a = math.tau * k / sides
            q = Vector(p) + radii[i] * (math.cos(a) * u + math.sin(a) * v)
            verts.append(tuple(q))
    for i in range(len(points) - 1):
        for k in range(sides):
            kk = (k + 1) % sides
            a, b = start + i * sides, start + (i + 1) * sides
            faces.append((a + k, a + kk, b + kk, b + k))


def _droplet(verts, faces, p, radius, stretch=1):
    # Octahedral particles are subpixel at the intended full-scene camera.
    start = len(verts)
    x, y, z = p
    verts.extend([(x + radius, y, z), (x, y + radius, z), (x - radius, y, z), (x, y - radius, z), (x, y, z + radius * stretch), (x, y, z - radius * stretch)])
    for i in range(4):
        j = (i + 1) % 4
        faces.extend([(start + i, start + j, start + 4), (start + j, start + i, start + 5)])


def _spray(col, sheet, foam):
    rng = random.Random(57231)
    verts, faces = [], []
    for i in range(320):
        ox, oy, oz = rng.uniform(-3, 1.1), rng.uniform(4.8, 5.8), rng.uniform(.14, .35)
        a = rng.uniform(0, math.tau)
        speed = rng.uniform(.5, 2.6)
        height = rng.uniform(.15, 1.2)
        # Discrete ballistic droplets, without continuous wire-like arcs.
        for j in range(rng.randint(10, 24)):
            t = rng.uniform(.3, 1)
            p = (ox + math.cos(a) * speed * t, oy + math.sin(a) * speed * t, oz + 4 * height * t * (1 - t) + rng.uniform(-.06, .06))
            _droplet(verts, faces, p, rng.uniform(.006, .026))
    for i in range(3300):
        # Fine suspended droplets around the water/air interface, thinning outward.
        z = rng.uniform(.4, 18.7)
        t = 1 - z / 19.05
        x, y, _ = _fall_position(t, rng.uniform(-2.9, .9), 1.4)
        spread = .08 + .48 * t
        x += rng.gauss(0, spread)
        y -= abs(rng.gauss(0, spread))
        _droplet(verts, faces, (x, y, z), rng.uniform(.008, .023), rng.uniform(1.3, 3))
    _mesh('Spray | airborne jets and thousands of discrete droplets', verts, faces, col, foam)

    # Two small sideways sprays where the water meets rock shoulders.
    verts, faces = [], []
    for z0, x0, y0 in [(11.3, -.8, 5.85), (5.25, -1.8, 5.55)]:
        for i in range(420):
            t = rng.random()
            p = (x0 + rng.gauss(0, .75) * t, y0 - rng.uniform(.15, 1.1) * t, z0 + .7 * t - 2.5 * t * t)
            _droplet(verts, faces, p, rng.uniform(.012, .033), 1.7)
    _mesh('Spray | ledge impacts', verts, faces, col, foam)


def _impact_sheets(col, sheet, lace):
    """Broken fan-shaped films make the two rock strikes and pool impact read."""
    rng = random.Random(63421)
    for k, (cx, cy, z0, length, width) in enumerate([
        (-1.15, 5.75, 12.15, 2.05, 3.2), (-.8, 5.35, 5.25, 1.9, 3.6)
    ]):
        verts, faces = [], []
        rows, across = 34, 32
        for j in range(rows + 1):
            t = j / rows
            for i in range(across + 1):
                u = i / across * 2 - 1
                w = width * (.37 + .26 * t)
                x = cx + u * w + .07 * math.sin(t * 19 + u * 4)
                y = cy - .68 * math.sqrt(t) - .12 * math.sin(t * 11 + u * 12)
                z = z0 - length * t + .07 * math.sin(t * 35 + u * 16)
                verts.append((x, y, z))
        for j in range(rows):
            for i in range(across):
                # Thin fans disintegrate into ragged, aerated edges.
                t = j / rows
                if t > .55 and rng.random() < (t - .55) * 1.5:
                    continue
                a = j * (across + 1) + i
                faces.append((a, a + 1, a + across + 2, a + across + 1))
        _mesh('Cascade | broken spreading film over ledge %d' % (k + 1), verts, faces, col, sheet)
        _mesh('Cascade | frothy aeration over ledge %d' % (k + 1),
              [(x, y - .012, z) for x, y, z in verts], faces, col, lace)

    verts, faces = [], []
    for k in range(32):
        cx, cy = rng.uniform(-3.2, 1.3), rng.uniform(4.55, 5.1)
        direction = rng.uniform(math.pi, math.tau)
        reach = rng.uniform(.5, 1.9)
        half = rng.uniform(.12, .48)
        height = rng.uniform(.16, .64)
        start = len(verts)
        rows, across = 18, 10
        phase = rng.uniform(0, math.tau)
        for j in range(rows + 1):
            t = j / rows
            for i in range(across + 1):
                u = 2 * i / across - 1
                lateral = u * half * (.3 + t)
                x = cx + reach * t * math.cos(direction) - lateral * math.sin(direction)
                y = cy + reach * t * math.sin(direction) + lateral * math.cos(direction)
                z = .15 + height * math.sin(math.pi * t) + .045 * math.sin(t * 28 + u * 9 + phase)
                verts.append((x, y, z))
        for j in range(rows):
            for i in range(across):
                if j > rows * .55 and rng.random() < .24:
                    continue
                a = start + j * (across + 1) + i
                faces.append((a, a + 1, a + across + 2, a + across + 1))
    _mesh('Splash | crumpled aerated films at the plunge', verts, faces, col, lace)


def _impact_mist(col):
    mat = bpy.data.materials.new('Spray | very local soft aerosol')
    mat.use_nodes = True
    n, l = mat.node_tree.nodes, mat.node_tree.links
    n.clear()
    out = n.new('ShaderNodeOutputMaterial')
    volume = n.new('ShaderNodeVolumeScatter')
    volume.inputs['Color'].default_value = (.85, .92, .94, 1)
    volume.inputs['Anisotropy'].default_value = .15
    coord = n.new('ShaderNodeTexCoord')
    dist = n.new('ShaderNodeVectorMath')
    dist.operation = 'DISTANCE'
    dist.inputs[1].default_value = (.5, .5, .5)
    l.new(coord.outputs['Generated'], dist.inputs[0])
    falloff = n.new('ShaderNodeMapRange')
    falloff.inputs['From Min'].default_value = .04
    falloff.inputs['From Max'].default_value = .5
    falloff.inputs['To Min'].default_value = .14
    falloff.inputs['To Max'].default_value = 0
    l.new(dist.outputs['Value'], falloff.inputs['Value'])
    tex = n.new('ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value = 5
    tex.inputs['Detail'].default_value = 2
    l.new(coord.outputs['Generated'], tex.inputs['Vector'])
    mul = n.new('ShaderNodeMath')
    mul.operation = 'MULTIPLY'
    l.new(falloff.outputs[0], mul.inputs[0])
    l.new(tex.outputs['Fac'], mul.inputs[1])
    l.new(mul.outputs[0], volume.inputs['Density'])
    l.new(volume.outputs[0], out.inputs['Volume'])
    bpy.ops.mesh.primitive_cube_add(size=2, location=(-1, 4.55, .7))
    obj = bpy.context.object
    obj.name = 'Spray | softly fading localized impact mist'
    obj.scale = (3.3, 2.0, 1.3)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    obj.data.materials.append(mat)
    obj.display_type = 'WIRE'


def _foam(col, material, lace):
    rng = random.Random(1238)
    # A broken, irregular impact raft, with a ragged advancing edge.
    verts, faces = [], []
    segments, rings = 160, 20
    verts.append((-1.0, 4.8, .17))
    for j in range(1, rings + 1):
        r = j / rings
        for i in range(segments):
            a = math.tau * i / segments
            edge = 1 + .12 * math.sin(a * 7) + .08 * math.sin(a * 19)
            x = -1 + math.cos(a) * 3.0 * r * edge
            y = 4.8 + math.sin(a) * 1.9 * r * edge
            verts.append((x, y, _surface_z(x, y) + .027 + .045 * (1 - r)))
    for i in range(segments):
        faces.append((0, 1 + i, 1 + (i + 1) % segments))
    for j in range(rings - 1):
        a, b = 1 + j * segments, 1 + (j + 1) * segments
        for i in range(segments):
            k = (i + 1) % segments
            faces.append((a + i, b + i, b + k, a + k))
    _mesh('Foam | broken lace at plunge pool', verts, faces, col, lace)

    verts, faces = [], []
    # Advection filaments form softly curling trails, rather than solid circles.
    for i in range(115):
        x0 = rng.uniform(-3.8, 1.7)
        y0 = rng.uniform(3.4, 5.1)
        length = rng.uniform(.6, 5.5)
        width = rng.uniform(.012, .05)
        phase = rng.uniform(0, math.tau)
        steps = 35
        start = len(verts)
        for j in range(steps):
            t = j / (steps - 1)
            y = y0 - length * t
            x = x0 + .18 * math.sin(t * 6 + phase) + (x0 + 1) * .27 * t
            w = width * (1 - t) * (.75 + .25 * math.sin(t * 35 + phase))
            verts.extend([(x - w, y, _surface_z(x, y) + .016), (x + w, y, _surface_z(x, y) + .016)])
        for j in range(steps - 1):
            a = start + 2 * j
            if rng.random() > .13:
                faces.append((a, a + 1, a + 3, a + 2))
    # Small, discontinuous wave crests around the fall.
    for i in range(55):
        r = rng.uniform(1.7, 7.8)
        a0 = rng.uniform(math.pi * .92, math.pi * 1.95)
        length = rng.uniform(.045, .23)
        width = rng.uniform(.009, .024) * (1 - r / 11)
        start = len(verts)
        for j in range(24):
            t = j / 23
            a = a0 + t * length
            for dr in [-width, width]:
                x = -1 + (r + dr) * math.cos(a)
                y = 5 + (r + dr) * math.sin(a)
                verts.append((x, y, _surface_z(x, y) + .02))
        for j in range(23):
            a = start + 2 * j
            faces.append((a, a + 1, a + 3, a + 2))
    _mesh('Foam | drifting tendrils and broken ripple crests', verts, faces, col, material)


def build_water():
    """Create the waterfall and plunge pool; all coordinates are in metres."""
    pool = _collection('WATER • clear forest plunge pool')
    falls = _collection('WATER • nineteen metre vertical cascade')
    splash = _collection('WATER • spray and foam')
    water, sheet, foam, lace = _materials()
    _pool(pool, water)
    _upstream(falls, sheet)
    # Main irregular sheets overlap but leave translucent spaces between ribbons.
    for i, (x, width) in enumerate([(-2.60, .57), (-2.08, .72), (-1.47, .90), (-.72, .83), (.02, .66), (.57, .43)]):
        _ribbon('Cascade | main channel %02d' % (i + 1), falls, sheet, x, width, 701 + i)
    rng = random.Random(920)
    # Foam striations are interrupted down the fall, not full-length white tubes.
    for i in range(78):
        begin = rng.uniform(0, .88)
        end = min(1, begin + rng.uniform(.07, .36))
        _ribbon('Cascade | aerated striation %02d' % (i + 1), falls, foam,
                rng.uniform(-2.85, .83), rng.uniform(.014, .057), 970 + i, begin, end)
    # Outer strands separate from the wider streams and carry stray light.
    for i in range(12):
        _ribbon('Cascade | detached edge filament %02d' % (i + 1), falls, sheet,
                rng.choice([-3.0, .95]) + rng.uniform(-.18, .18), rng.uniform(.02, .055), 1090 + i,
                rng.uniform(.10, .5), rng.uniform(.73, 1))
    _spray(splash, sheet, foam)
    _impact_sheets(splash, sheet, lace)
    _impact_mist(splash)
    _foam(splash, foam, lace)
    return {'height_m': 19, 'width_m': 4.4, 'pool_radii_m': [10.92, 10.4], 'water_level_m': .1,
            'waterfall_center': [-1, 6.2, 9.5], 'objects': len(pool.objects) + len(falls.objects) + len(splash.objects)}
