"""Orient authored shading detail against a signed solid's outward gradient.

Closed source meshes can have inward winding while parity still determines
their interior correctly. Winding is therefore not the orientation authority
for the stored optical field. Valid hemisphere detail is returned unchanged.
"""


def outward_shading_normal(shading, outward_gradient):
    return -shading if shading.dot(outward_gradient) < 0 else shading


def signed_outward_gradient(point, nearest, distance, sign, source_normal, inside, probe_distance):
    """Parity determines orientation even when the lattice lies on a face.

    Away from a face the signed distance derivative is sufficient. At zero
    distance, one bounded offset probe distinguishes inward source winding;
    source normals alone cannot determine the occupied side.
    """
    # Blender's BVH/Vector arithmetic is float32: an exact face query can
    # return a tiny tangential nearest-point error. Dividing by that distance
    # invents a tangent gradient, so use a bounded positional roundoff band.
    roundoff = (8 * 2**-23) * max(1.0, *(abs(v) for v in point))
    if distance > roundoff:
        return (point-nearest)*(sign/distance)
    probe = max(probe_distance, roundoff*4)
    return -source_normal if inside(point+source_normal*probe) else source_normal
