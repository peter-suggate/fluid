"""Botanical geometry for the forest-waterfall scene.

The library is entirely procedural and self-contained.  Trees use branching bark
meshes and thousands of individually folded leaves, never opaque canopy blobs.
Call ``build_foliage(terrain_height)`` from a Blender scene-authoring script.
"""
import math
import random
import bpy
from mathutils import Vector

TAU = math.tau


def _collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def _material(name, color, roughness=.6, leaf=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bs = nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (*color, 1)
    bs.inputs['Roughness'].default_value = roughness
    if leaf:
        bs.inputs['Subsurface Weight'].default_value = .035
        bs.inputs['Subsurface Radius'].default_value = (.18, .5, .12)
        bs.inputs['Coat Weight'].default_value = .15
        bs.inputs['Coat Roughness'].default_value = .3
        translucent = nodes.new('ShaderNodeBsdfTranslucent')
        translucent.inputs['Color'].default_value = (*[min(1, c*1.3) for c in color], 1)
        mix = nodes.new('ShaderNodeMixShader')
        mix.inputs[0].default_value = .19
        links.new(bs.outputs['BSDF'], mix.inputs[1])
        links.new(translucent.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], nodes.get('Material Output').inputs['Surface'])
        noise = nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 70
        noise.inputs['Detail'].default_value = 2
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .1
        bump.inputs['Distance'].default_value = .004
        links.new(noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs[0], bs.inputs['Normal'])
    return mat


def _bark_material():
    mat = _material('Forest • deeply fissured umber bark', (.075, .044, .021), .9)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bs = nodes.get('Principled BSDF')
    tex = nodes.new('ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value = 5
    tex.inputs['Detail'].default_value = 5
    mapping = nodes.new('ShaderNodeVectorMath')
    mapping.operation = 'MULTIPLY'
    mapping.inputs[1].default_value = (3, 3, .3)
    co = nodes.new('ShaderNodeTexCoord')
    links.new(co.outputs['Object'], mapping.inputs[0])
    links.new(mapping.outputs[0], tex.inputs['Vector'])
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = .2
    ramp.color_ramp.elements[0].color = (.018, .012, .007, 1)
    ramp.color_ramp.elements[1].position = .8
    ramp.color_ramp.elements[1].color = (.19, .115, .056, 1)
    links.new(tex.outputs['Fac'], ramp.inputs[0])
    links.new(ramp.outputs['Color'], bs.inputs['Base Color'])
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = .6
    bump.inputs['Distance'].default_value = .045
    links.new(tex.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs[0], bs.inputs['Normal'])
    return mat


class _Mesh:
    def __init__(self):
        self.v, self.f, self.m = [], [], []

    def tube(self, points, radii, sides=7, material=0):
        start = len(self.v)
        for i, point in enumerate(points):
            point = Vector(point)
            tangent = Vector(points[min(i+1, len(points)-1)])-Vector(points[max(0, i-1)])
            tangent.normalize()
            side = tangent.cross(Vector((0, 1, 0)))
            if side.length < .001:
                side = tangent.cross(Vector((1, 0, 0)))
            side.normalize()
            up = tangent.cross(side).normalized()
            for j in range(sides):
                a = TAU*j/sides
                rad = radii[i]*(1+.09*math.sin(j*4.4+i*.8))
                self.v.append(tuple(point+rad*(side*math.cos(a)+up*math.sin(a))))
        for i in range(len(points)-1):
            for j in range(sides):
                self.f.append((start+i*sides+j, start+i*sides+(j+1)%sides,
                               start+(i+1)*sides+(j+1)%sides, start+(i+1)*sides+j))
                self.m.append(material)
        self.f.append(tuple(start+j for j in reversed(range(sides))))
        self.m.append(material)
        self.f.append(tuple(start+(len(points)-1)*sides+j for j in range(sides)))
        self.m.append(material)

    def leaf(self, center, axis, length, width, material, roll=0, bend=.13):
        center, axis = Vector(center), Vector(axis).normalized()
        side = axis.cross(Vector((0, 0, 1)))
        if side.length < .01:
            side = Vector((1, 0, 0))
        side.normalize()
        up = axis.cross(side).normalized()
        side, up = side*math.cos(roll)+up*math.sin(roll), up*math.cos(roll)-side*math.sin(roll)
        # Folded midrib and pointed tip. The long axis runs from petiole to apex.
        points = [(0, 0, 0), (.28, -.79, -.022), (.28, 0, .035), (.28, .79, -.022),
                  (.66, -.67, -.045), (.66, 0, -.005), (.66, .67, -.045), (1, 0, -bend)]
        offset = len(self.v)
        for x, y, z in points:
            self.v.append(tuple(center+axis*(x*length)+side*(y*width/2)+up*(z*length)))
        for face in [(0, 1, 2), (0, 2, 3), (1, 4, 5, 2), (2, 5, 6, 3), (4, 7, 5), (5, 7, 6)]:
            self.f.append(tuple(offset+i for i in face))
            self.m.append(material)

    def data(self, name, materials):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(self.v, [], self.f)
        mesh.update()
        for mat in materials:
            mesh.materials.append(mat)
        for poly, index in zip(mesh.polygons, self.m):
            poly.material_index = index
            poly.use_smooth = True
        return mesh


def _instance(name, mesh, collection, location, scale=1, rotation=0):
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location = location
    obj.scale = (scale, scale, scale) if isinstance(scale, (float, int)) else scale
    obj.rotation_euler[2] = rotation
    return obj


def _tree(seed, leaf_materials, bark):
    rng = random.Random(seed)
    wood, leaves = _Mesh(), _Mesh()
    height = rng.uniform(10.5, 15)
    radius = rng.uniform(.38, .53)
    lean = Vector((rng.uniform(-.7, .7), rng.uniform(-.6, .6), 0))
    # Broadleaf architecture forks low and reaches outward before rising into
    # a spreading, irregular crown. A short central leader avoids the bare
    # Christmas-tree silhouette of a single trunk with horizontal spiral arms.
    trunk = [lean*(i/20)**1.5+Vector((.12*math.sin(i*.4), .12*math.cos(i*.3), height*.78*i/20)) for i in range(21)]
    # Basal flare blends into an irregular tapered bole. More rings and radial
    # sides preserve an organic trunk at the finite runtime sampling pitch.
    wood.tube(trunk, [radius*(1-i/22)**1.15+.018+radius*.7*math.exp(-i*.8) for i in range(21)], 19)
    for i in range(7):
        angle = i*TAU/7+rng.uniform(-.2, .2)
        d = Vector((math.cos(angle), math.sin(angle), 0))
        wood.tube([d*radius*.45+Vector((0, 0, 1.15)), d*.65+Vector((0, 0, .45)), d*1.05+Vector((0, 0, .12)), d*rng.uniform(1.5, 2.2)+Vector((0, 0, -.08))], [radius*.7, radius*.5, radius*.25, .035], 11)
    leaf_count = 0
    for branch_i in range(13):
        fraction=branch_i/12
        z = height*(.18+.032*branch_i)
        angle = branch_i*2.399+rng.uniform(-.5, .5)
        extent = (4.5-2.3*fraction)*rng.uniform(.85, 1.2)
        direction = Vector((math.cos(angle), math.sin(angle), 0))
        origin = lean*(z/height)**1.5+Vector((0, 0, z))
        end = lean+direction*extent+Vector((0, 0, height*(.62+.35*fraction)+rng.uniform(-.45,.45)))
        midpoint = origin.lerp(end, .46)+Vector((0, 0, .35))
        wood.tube([origin, origin.lerp(midpoint,.5), midpoint, midpoint.lerp(end,.55), end], [radius*.62, radius*.5, radius*.3, radius*.14, .025], 11)
        for twig_i in range(5):
            t = .53+.115*twig_i
            base = origin.lerp(end, t)
            sign = -1 if twig_i%2 else 1
            da = angle+sign*rng.uniform(.4, 1.65)
            twig_dir = Vector((math.cos(da), math.sin(da), rng.uniform(.15, .65)))
            twig_end = base+twig_dir*rng.uniform(.65, 1.7)
            wood.tube([base, base.lerp(twig_end, .6)+Vector((0, 0, .18)), twig_end], [.05*(1-t)+.018, .02, .005], 5)
            # A lush crown needs real leaf area: these many overlapping folded
            # blades create density while retaining a naturally broken outline.
            for j in range(230):
                az = rng.uniform(0, TAU)
                rad = math.sqrt(rng.random())
                location = twig_end+Vector((math.cos(az)*rad*1.3, math.sin(az)*rad*1.05, rng.uniform(-.65, .8)))
                leaf_axis = Vector((math.cos(az+rng.uniform(-.9, .9)), math.sin(az+rng.uniform(-.9, .9)), rng.uniform(-.5, .6)))
                length = rng.uniform(.24, .46)
                leaves.leaf(location, leaf_axis, length, length*rng.uniform(.46, .76), rng.choices(range(len(leaf_materials)), [4, 4, 3, 2, 1, 1])[0], rng.uniform(-1.1, 1.1))
                leaf_count += 1
        # Branch-tip shoots fill the upper center with foliage, so the visible
        # crown consists of leafy masses rather than disconnected twig tufts.
        for shoot_i in range(3):
            sa=angle+shoot_i*2.1+rng.uniform(-.35,.35)
            shoot=end+Vector((math.cos(sa)*.6,math.sin(sa)*.6,.35))
            wood.tube([end,shoot],[.016,.003],5)
            for j in range(95):
                az=rng.uniform(0,TAU)
                rad=math.sqrt(rng.random())*.78
                loc=shoot+Vector((math.cos(az)*rad,math.sin(az)*rad,rng.uniform(-.35,.55)))
                axis=(math.cos(az),math.sin(az),rng.uniform(-.2,.4))
                length=rng.uniform(.25,.43)
                leaves.leaf(loc,axis,length,length*rng.uniform(.5,.73),rng.randrange(len(leaf_materials)),rng.uniform(-.8,.8))
                leaf_count+=1
    return wood.data('Tree branching architecture '+str(seed), [bark]), leaves.data('Individual folded canopy leaves '+str(seed), leaf_materials), leaf_count


def tree_species(seed, species, leaf_materials, bark):
    """Distinct authored architectures; each crown consists of folded leaves.

    Alder has several leaning pale stems and an airy oval crown. Cedar carries
    downward-swept evergreen sprays on irregular tiers and a persistent leader.
    These are actual mesh forms, not scaled copies of the broadleaf prototype.
    """
    if species == 'broadleaf':
        return _tree(seed, leaf_materials, bark)
    rng = random.Random(seed)
    wood, leaves = _Mesh(), _Mesh()
    count = 0
    if species == 'alder':
        for stem in range(3):
            az = stem*TAU/3+.35
            side = Vector((math.cos(az),math.sin(az),0))
            height = [13.4,11.5,12.3][stem]
            lean = side*rng.uniform(.9,1.7)
            pts = [side*.17+lean*(i/18)**1.3+Vector((.055*math.sin(i*.7),.045*math.cos(i*.8),height*i/18)) for i in range(19)]
            wood.tube(pts,[.24*(1-i/19)**.9+.018+.13*math.exp(-i) for i in range(19)],15)
            for root in range(3):
                d=Vector((math.cos(az+root*2.1),math.sin(az+root*2.1),0))
                wood.tube([side*.17+Vector((0,0,.6)),d*.7+Vector((0,0,.12)),d*1.25+Vector((0,0,-.08))],[.18,.12,.025],9)
            for b in range(12):
                t=.33+b*.05
                angle=az+b*2.399+rng.uniform(-.4,.4)
                d=Vector((math.cos(angle),math.sin(angle),0))
                origin=side*.17+lean*t**1.3+Vector((0,0,height*t))
                end=origin+d*rng.uniform(1.2,2.6)*(1-.45*t)+Vector((0,0,rng.uniform(.8,1.5)))
                wood.tube([origin,origin.lerp(end,.45)+Vector((0,0,.25)),end],[.09,.05,.012],9)
                for twig in range(3):
                    angle2=angle+(twig-1)*.8
                    tip=end+Vector((math.cos(angle2)*.7,math.sin(angle2)*.7,.2-.15*twig))
                    wood.tube([end,tip],[.018,.005],6)
                    for j in range(105):
                        a=rng.uniform(0,TAU);r=math.sqrt(rng.random())*.9
                        p=tip+Vector((math.cos(a)*r,math.sin(a)*r,rng.uniform(-.55,.65)))
                        length=rng.uniform(.23,.37)
                        leaves.leaf(p,(math.cos(a),math.sin(a),rng.uniform(-.4,.5)),length,length*.78,rng.randrange(len(leaf_materials)),rng.uniform(-1,1))
                        count+=1
    elif species == 'cedar':
        height=15.3
        pts=[Vector((.12*math.sin(i*.3),.07*math.sin(i*.5),height*i/24)) for i in range(25)]
        wood.tube(pts,[.43*(1-i/25)**1.2+.018+.24*math.exp(-i*.8) for i in range(25)],19)
        for root in range(7):
            a=root*TAU/7;d=Vector((math.cos(a),math.sin(a),0))
            wood.tube([Vector((0,0,.8)),d*.65+Vector((0,0,.16)),d*1.6+Vector((0,0,-.08))],[.25,.18,.025],9)
        for tier in range(13):
            t=.20+tier*.06
            for b in range(5):
                a=b*TAU/5+tier*.43+rng.uniform(-.2,.2)
                d=Vector((math.cos(a),math.sin(a),0));side=Vector((-d.y,d.x,0))
                length=(4.6*(1-t)**.7+.15)*rng.uniform(.85,1.12)
                start=Vector((0,0,height*t));end=start+d*length+Vector((0,0,-.4))
                wood.tube([start,start.lerp(end,.45)+Vector((0,0,-.18)),end],[.13*(1-t)+.02,.055,.008],9)
                for spray in range(5):
                    k=.35+spray*.15
                    tip=start.lerp(end,k)+side*(.38 if spray%2 else -.38)
                    wood.tube([start.lerp(end,k-.12),tip],[.024,.004],5)
                    for j in range(34):
                        u=rng.uniform(-.45,.45);v=rng.uniform(-.6,.6)
                        p=tip+d*u+side*v+Vector((0,0,rng.uniform(-.22,.25)))
                        axis=(d+side*rng.uniform(-1.2,1.2)+Vector((0,0,-.15))).normalized()
                        le=rng.uniform(.27,.48)
                        leaves.leaf(p,axis,le,le*.38,rng.randrange(len(leaf_materials)),rng.uniform(-.45,.45),.2)
                        count+=1
        for j in range(280):
            t=rng.uniform(.9,1.025);a=rng.uniform(0,TAU);r=(1.04-t)*5
            p=Vector((math.cos(a)*r,math.sin(a)*r,height*t))
            leaves.leaf(p,(math.cos(a),math.sin(a),.4),.35,.11,rng.randrange(len(leaf_materials)),0,.18)
            count+=1
    else:
        raise ValueError('Unknown tree species '+species)
    return wood.data('Tree branching architecture '+species+' '+str(seed),[bark]),leaves.data('Individual folded canopy leaves '+species+' '+str(seed),leaf_materials),count


def _fern(seed, leaf_materials, stems):
    rng = random.Random(seed)
    mesh = _Mesh()
    for frond_i in range(9):
        az = TAU*frond_i/9+rng.uniform(-.3, .3)
        direction = Vector((math.cos(az), math.sin(az), 0))
        lateral = Vector((-math.sin(az), math.cos(az), 0))
        length = rng.uniform(1.1, 1.75)
        points = []
        for i in range(23):
            t = i/22
            points.append(direction*(length*t)+Vector((0, 0, .06+length*(1.65*t-1.30*t*t))))
        mesh.tube(points, [.012*(1-i/25) for i in range(23)], 4, len(leaf_materials))
        for j in range(2, 22):
            t = j/22
            leaflet_len = length*.23*math.sin(math.pi*t)**.65*(1-.45*t)
            for sign in [-1, 1]:
                base = points[j]
                axis = (lateral*sign*.85+direction*.42+Vector((0, 0, .08))).normalized()
                mesh.leaf(base, axis, leaflet_len, leaflet_len*.27, rng.randrange(4), rng.uniform(-.12, .12), .08)
                # Each pinna has additional pointed lobes: a delicate twice-divided fern.
                for k in range(1, 5):
                    lbase = base+axis*(leaflet_len*k/6)
                    for sub_sign in [-1, 1]:
                        sub_axis = (axis*.35+direction*sub_sign*.8).normalized()
                        mesh.leaf(lbase, sub_axis, leaflet_len*.23*(1-k/6), leaflet_len*.07, rng.randrange(4), .12*sub_sign)
    return mesh.data('Pinnate fern with finely divided fronds '+str(seed), leaf_materials+[stems])


def _grass(seed, materials):
    rng = random.Random(seed)
    mesh = _Mesh()
    for i in range(31):
        a = rng.uniform(0, TAU)
        r = rng.uniform(0, .18)
        base = Vector((math.cos(a)*r, math.sin(a)*r, 0))
        h = rng.uniform(.26, .85)
        direction = Vector((math.cos(a), math.sin(a), 0))
        side = Vector((-math.sin(a), math.cos(a), 0))
        n = len(mesh.v)
        width = rng.uniform(.009, .024)
        for j in range(6):
            t = j/5
            p = base+direction*(h*t*t*.58)+Vector((0, 0, h*(1.35*t-.44*t*t)))
            for sign in [-1, 1]:
                mesh.v.append(tuple(p+side*sign*width*(1-t)))
        mi = rng.randrange(len(materials))
        for j in range(5):
            mesh.f.append((n+2*j, n+2*j+1, n+2*j+3, n+2*j+2))
            mesh.m.append(mi)
    return mesh.data('Riparian sedge arching blades '+str(seed), materials)


def _broadleaf(seed, materials, stem_mat):
    rng = random.Random(seed)
    mesh = _Mesh()
    for i in range(11):
        az = i*2.399+rng.uniform(-.3, .3)
        direction = Vector((math.cos(az), math.sin(az), rng.uniform(.5, 1.1))).normalized()
        stem_end = direction*rng.uniform(.25, .65)
        mesh.tube([(0, 0, .02), stem_end*.7+Vector((0, 0, .1)), stem_end], [.014, .009, .004], 5, len(materials))
        length = rng.uniform(.38, .75)
        mesh.leaf(stem_end, Vector((direction.x, direction.y, .1)), length, length*.58, rng.randrange(len(materials)), rng.uniform(-.5, .5), .2)
    return mesh.data('Glossy woodland broadleaf rosette '+str(seed), materials+[stem_mat])


def _vine_mesh(seed, materials, stem_mat):
    rng = random.Random(seed)
    mesh = _Mesh()
    points = []
    for j in range(25):
        t = j/24
        points.append(Vector((.25*math.sin(t*12), .12*math.sin(t*17), -t*4)))
    mesh.tube(points, [.016*(1-j/29) for j in range(25)], 5, len(materials))
    for j in range(1, 25):
        for sign in [-1, 1]:
            mesh.leaf(points[j], (sign, -.2, -.2), rng.uniform(.17, .35), rng.uniform(.12, .2), rng.randrange(len(materials)), rng.uniform(-1, 1))
    return mesh.data('Trailing cliff ivy '+str(seed), materials+[stem_mat])


def build_foliage(terrain_height, cliff_edge=None, cliff_top=None):
    """Populate a sixty-metre glade; return inspectable botanical instance counts."""
    rng = random.Random(22102026)
    canopy_col = _collection('FOREST | Mixed broadleaf alder and cedar canopy')
    shrub_col = _collection('FOREST | Saplings and understory')
    fern_col = _collection('FOREST | Detailed fern colonies')
    herb_col = _collection('FOREST | Sedge and broadleaf bank plants')
    vine_col = _collection('FOREST | Hanging cliff vegetation')
    bark = _bark_material()
    stem = _material('Forest • fern rachis and living stems', (.08, .14, .018), .62)
    greens = [(.026,.085,.009), (.041,.13,.014), (.062,.19,.022), (.105,.25,.035), (.16,.29,.047), (.054,.15,.063)]
    leaves = [_material('Forest • canopy leaf '+str(i), rgb, .45, True) for i, rgb in enumerate(greens)]
    fern_leaves = [_material('Forest • fern leaflet '+str(i), rgb, .5, True) for i, rgb in enumerate([(.03,.12,.012),(.052,.18,.02),(.09,.24,.027),(.15,.28,.033)])]
    grasses = [_material('Forest • sedge blade '+str(i), rgb, .58, True) for i, rgb in enumerate([(.045,.10,.01),(.09,.19,.02),(.16,.26,.036),(.22,.29,.065)])]
    broad_materials = [_material('Forest • waxy bank leaf '+str(i), rgb, .32, True) for i, rgb in enumerate([(.018,.09,.027),(.028,.14,.037),(.065,.2,.043)])]
    alder_bark = _material('Forest • pale alder trunk', (.38,.37,.31), .86)
    cedar_bark = _material('Forest • fibrous cedar trunk', (.09,.055,.029), .93)
    cedar_leaves = [_material('Forest • evergreen spray '+str(i),
        tuple(c*(.65+.16*i) for c in (.025,.075,.045)),.62,True) for i in range(6)]
    tree_data = [_tree(340+i*71, leaves, bark) for i in range(3)]
    tree_data += [tree_species(553+i*71,'alder',leaves,alder_bark) for i in range(2)]
    tree_data += [tree_species(695+i*71,'cedar',cedar_leaves,cedar_bark) for i in range(2)]
    fern_data = [_fern(700+i, fern_leaves, stem) for i in range(5)]
    grass_data = [_grass(530+i, grasses) for i in range(5)]
    broad_data = [_broadleaf(810+i, broad_materials, stem) for i in range(4)]
    vine_data = [_vine_mesh(970+i, leaves[:4], stem) for i in range(3)]
    stats = {'mature_trees': 0, 'saplings': 0, 'ferns': 0, 'sedges': 0, 'broadleaf_plants': 0, 'trailing_vines': 0, 'unique_tree_leaf_count': sum(t[2] for t in tree_data)}

    def supported_ground(x, y, maximum_slope=1.5):
        # Avoid planting upright trees or fern roots on the near-vertical ramp
        # joining the basin to the plateau. Ground cover elsewhere follows soil.
        dx = (terrain_height(x+.25,y)-terrain_height(x-.25,y))/.5
        dy = (terrain_height(x,y+.25)-terrain_height(x,y-.25))/.5
        return math.hypot(dx,dy) < maximum_slope

    def place_tree(x, y, scale, mature=True):
        if not supported_ground(x,y):
            return
        z = terrain_height(x, y)-.04
        wood, leaf, _ = rng.choice(tree_data)
        angle = rng.uniform(0, TAU)
        col = canopy_col if mature else shrub_col
        label = 'Canopy tree' if mature else 'Young understory tree'
        index = stats['mature_trees']+stats['saplings']
        _instance(label+' %03d | branching trunk'%index, wood, col, (x,y,z), scale, angle)
        _instance(label+' %03d | thousands of leaves'%index, leaf, col, (x,y,z), scale, angle)
        stats['mature_trees' if mature else 'saplings'] += 1

    # Tall distant forest closes the horizon; irregular spacing avoids orchard rows.
    for row_y in [16, 24, 33, 41]:
        for x0 in range(-33, 36, 6):
            x, y = x0+rng.uniform(-2.2,2.2), row_y+rng.uniform(-2.3,2.3)
            if row_y == 16 and -5<x<4:
                continue
            scale = rng.uniform(.55,.8) if row_y == 16 else rng.uniform(.72,1.25)
            place_tree(x, y, scale)
    for side in [-1, 1]:
        for i in range(15):
            x = side*rng.uniform(15,31)
            y = rng.uniform(-8,18)
            place_tree(x, y, rng.uniform(.75,1.35))
    # A few closer trunks frame the water, with open center and visible waterfall.
    for x,y,s in [(-15,-10,1.18),(-19,-18,1.3),(19,0,.98),(23,-9,1.15),(-12,4,.72),(12,6,.68)]:
        place_tree(x,y,s)
    for i in range(65):
        x,y = rng.uniform(-30,30),rng.uniform(-14,29)
        pool = (x/10.5)**2+((y+2)/10)**2
        if pool<1.13 or (-5<x<4 and 3<y<15):
            continue
        # Keep foreground low enough that it does not become a solid hedge.
        if y < -5 and x > 4:
            continue
        place_tree(x,y,rng.uniform(.16,.34),False)

    # Natural drifts of fern plants under the canopy and along the wet margins.
    for i in range(470):
        x,y = rng.uniform(-29,29),rng.uniform(-20,30)
        pool = (x/10.5)**2+((y+2)/10)**2
        if pool < 1.10 or (-4.8<x<3.8 and 4<y<14):
            continue
        if y < -12 and -3<x<17:
            continue
        if not supported_ground(x,y,2.0):
            continue
        z = terrain_height(x,y)
        scale = rng.uniform(.42,.95)
        _instance('Fern colony %03d'%i, rng.choice(fern_data), fern_col, (x,y,z), scale, rng.uniform(0,TAU))
        stats['ferns']+=1
    # Art-directed large foreground fronds show the actual leaf construction.
    for x,y,s in [(-8.5,-9.6,1.35),(-12,-10,1.2),(-13,-6,1.15),(10,-7.8,1.1),(12,-10,1.3),(14,-7,1.1),(-5,-12,1.1)]:
        _instance('Hero fern | paired finely divided fronds',rng.choice(fern_data),fern_col,(x,y,terrain_height(x,y)),s,rng.uniform(0,TAU))
        stats['ferns']+=1
    for i in range(1050):
        x,y=rng.uniform(-27,27),rng.uniform(-20,24)
        pool=(x/10.5)**2+((y+2)/10)**2
        if pool<1.10 or (-4.5<x<3.8 and 4<y<14):
            continue
        if not supported_ground(x,y,2.0):
            continue
        z=terrain_height(x,y)
        if rng.random()<.65:
            _instance('Riparian sedge %04d'%i,rng.choice(grass_data),herb_col,(x,y,z),rng.uniform(.6,1.5),rng.uniform(0,TAU))
            stats['sedges']+=1
        else:
            _instance('Broadleaf bank rosette %04d'%i,rng.choice(broad_data),herb_col,(x,y,z),rng.uniform(.55,1.5),rng.uniform(0,TAU))
            stats['broadleaf_plants']+=1
    # Concentrated mixed colonies make the visible waterline luxuriant rather
    # than relying on uniform placement over the much larger forest floor.
    for i in range(580):
        az=rng.uniform(0,TAU)
        rr=rng.uniform(1.045,1.34)
        x=10.92*math.cos(az)*rr
        y=-2+10.4*math.sin(az)*rr
        if (-4.8<x<3.8 and 4<y<14) or not supported_ground(x,y,2.0):
            continue
        z=terrain_height(x,y)
        choice=rng.random()
        # The near approach retains low sedges so the pond remains legible.
        foreground_opening=y < -10 and -3<x<16
        if choice<.3 or foreground_opening:
            _instance('Waterline colony | arching sedges',rng.choice(grass_data),herb_col,
                      (x,y,z),rng.uniform(.55,1.25),rng.uniform(0,TAU))
            stats['sedges']+=1
        elif choice<.66:
            _instance('Waterline colony | broad glossy leaves',rng.choice(broad_data),herb_col,
                      (x,y,z),rng.uniform(.75,1.45),rng.uniform(0,TAU))
            stats['broadleaf_plants']+=1
        else:
            _instance('Waterline colony | overlapping fern fronds',rng.choice(fern_data),fern_col,
                      (x,y,z),rng.uniform(.55,1.05),rng.uniform(0,TAU))
            stats['ferns']+=1
    for side in [-1,1]:
        for i in range(9):
            x=side*rng.uniform(11.9,15.8)
            y=rng.uniform(-5,6)
            place_tree(x,y,rng.uniform(.12,.23),False)

    # Rich shoulder drifts break the rock/sky silhouette without spanning the
    # water corridor. Different plant heights layer over the high cliff lip.
    if cliff_edge is not None and cliff_top is not None:
        for side in [-1,1]:
            for i in range(38):
                x=side*rng.uniform(4.1,14)
                y=cliff_edge(x)+rng.uniform(1.28,2.8)
                z=terrain_height(x,y)
                if not supported_ground(x,y,2.0):
                    continue
                if i%3==0:
                    _instance('Cliff shoulder | lush broadleaf drift',rng.choice(broad_data),herb_col,
                              (x,y,z),rng.uniform(1.2,2.1),rng.uniform(0,TAU))
                    stats['broadleaf_plants']+=1
                else:
                    _instance('Cliff shoulder | overhanging fern drift',rng.choice(fern_data),fern_col,
                              (x,y,z),rng.uniform(.95,1.6),rng.uniform(0,TAU))
                    stats['ferns']+=1
            for i in range(18):
                x=side*rng.uniform(4.7,18)
                y=cliff_edge(x)+rng.uniform(1.4,3.3)
                place_tree(x,y,rng.uniform(.17,.29),False)

    # Ferns, broadleaf rosettes and short trailing vines emerge from shaded
    # crevices. Their roots sit inside the rock, while tilted leaves project
    # forward into light and drape over the irregular cliff face.
    if cliff_edge is not None and cliff_top is not None:
        for side in [-1,1]:
            for i in range(42):
                x=side*rng.uniform(4.6,22)
                y=cliff_edge(x)-rng.uniform(.45,.8)
                z=rng.uniform(2.4,cliff_top(x,y)-1.2)
                if i%3==0:
                    ob=_instance('Cliff crevice | waxy broadleaf colony',rng.choice(broad_data),vine_col,
                                 (x,y,z),rng.uniform(1,1.9),rng.uniform(0,TAU))
                    ob.rotation_euler[0]=rng.uniform(.7,1.25)
                    stats['broadleaf_plants']+=1
                else:
                    ob=_instance('Cliff crevice | projecting fern colony',rng.choice(fern_data),vine_col,
                                 (x,y,z),rng.uniform(.7,1.25),rng.uniform(0,TAU))
                    ob.rotation_euler[0]=rng.uniform(.4,.95)
                    stats['ferns']+=1
            for i in range(16):
                x=side*rng.uniform(4.8,20)
                y=cliff_edge(x)-.55
                z=rng.uniform(5,cliff_top(x,y)-1)
                _instance('Cliff crevice | trailing vine',rng.choice(vine_data),vine_col,
                          (x,y,z),rng.uniform(.4,.85),rng.uniform(-.4,.4))
                stats['trailing_vines']+=1

    # Ivy begins at the actual cliff top. The fallback detects the plateau from
    # the height field so this module also works without the optional callbacks.
    for side in [-1,1]:
        for i in range(26):
            x=side*rng.uniform(4.3,17)
            if cliff_edge is not None and cliff_top is not None:
                y=cliff_edge(x)-.38
                z=cliff_top(x,y)+.06
            else:
                candidates=[6+j*.1 for j in range(75)]
                edge=max(candidates,key=lambda sy:terrain_height(x,sy+.1)-terrain_height(x,sy-.1))
                y=edge-.75
                z=terrain_height(x,edge+1.4)
            _instance('Cliff ivy strand | anchored at upper lip',rng.choice(vine_data),vine_col,
                      (x,y,z),rng.uniform(.55,1.6),rng.uniform(-.35,.35))
            stats['trailing_vines']+=1
    # A wide, staggered distant forest closes the left-hand horizon and gives
    # the gorge a larger setting. These reuse the same botanical meshes; a
    # separate placement seed preserves all established foreground composition.
    distant_rng=random.Random(22107226)
    for row_y in [45,58,72]:
        x=-82+distant_rng.uniform(-3,3)
        while x<68:
            y=row_y+distant_rng.uniform(-3.1,3.1)
            width=distant_rng.uniform(.88,1.35)
            height=distant_rng.uniform(.84,1.4)
            place_tree(x,y,(width,width*distant_rng.uniform(.88,1.12),height),True)
            x+=distant_rng.uniform(6,9)

    bpy.context.scene['forest_botanical_inventory']=str(stats)
    print('BOTANICAL INVENTORY',stats,flush=True)
    return stats
