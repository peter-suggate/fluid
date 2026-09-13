//! Shared reconstructed-distance publication in source XYZ coordinates.
//! Follows geometric-interface-resident.wgsl.ts: polygon centroids, connected
//! point-neighbour RDF values, rank-revealing affine vertex fits and accepted
//! phase bounds. Topology vertices are cached per connected fluid component.
use crate::numerics::{owner_at, LIQUID_ISOVALUE};
use crate::topology::{BrickSeed, CompiledTopology};
use crate::{Fields, Graph, ValidationError};
use std::collections::HashMap;
type V = [f32; 3];
fn add(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] + b[i])
}
fn sub(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] - b[i])
}
fn scale(a: V, s: f32) -> V {
    a.map(|v| v * s)
}
fn dot(a: V, b: V) -> f32 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}
fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalize(a: V) -> V {
    scale(a, 1.0 / dot(a, a).sqrt())
}
fn min_width(a: V) -> f32 {
    a[0].min(a[1].min(a[2]))
}
fn bound(value: f32, accepted: f32, phase: i32) -> f32 {
    if phase < 0 && value >= 0.0 {
        accepted.min(0.0)
    } else if phase > 0 && value <= 0.0 {
        accepted.max(0.0)
    } else {
        value
    }
}
#[derive(Clone, Copy, Debug)]
struct Record {
    center: V,
    widths: V,
    active: Option<usize>,
}
#[derive(Clone, Copy, Debug)]
struct Interface {
    normal: V,
    offset: f32,
    centroid: V,
    valid: bool,
}
fn distance(plane: Interface, delta: V) -> f32 {
    dot(plane.normal, delta) - plane.offset
}
fn centroid(center: V, widths: V, normal: V, offset: f32) -> V {
    let half = scale(widths, 0.5);
    let mut points = Vec::<V>::with_capacity(12);
    for axis in 0..3 {
        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;
        for su in 0..2 {
            for sv in 0..2 {
                let mut a = [0.0; 3];
                let mut b = [0.0; 3];
                a[axis] = -half[axis];
                b[axis] = half[axis];
                a[u] = if su == 0 { -half[u] } else { half[u] };
                b[u] = a[u];
                a[v] = if sv == 0 { -half[v] } else { half[v] };
                b[v] = a[v];
                let fa = dot(normal, a) - offset;
                let fb = dot(normal, b) - offset;
                if ((fa <= 0.0 && fb >= 0.0) || (fa >= 0.0 && fb <= 0.0)) && (fa - fb).abs() > 1e-12
                {
                    let t = (fa / (fa - fb)).clamp(0.0, 1.0);
                    // WGSL mix is a*(1-t)+b*t, not an FMA or a+(b-a)*t.
                    let p = add(scale(a, 1.0 - t), scale(b, t));
                    if points.iter().all(|&q| dot(sub(q, p), sub(q, p)) > 1e-10) {
                        points.push(p);
                    }
                }
            }
        }
    }
    let fallback = add(center, scale(normal, offset));
    if points.len() < 3 {
        return fallback;
    }
    let arithmetic = scale(
        points.iter().copied().fold([0.0; 3], add),
        1.0 / points.len() as f32,
    );
    let abs = normal.map(f32::abs);
    let reference = if abs[1] <= abs[0] && abs[1] <= abs[2] {
        [0.0, 1.0, 0.0]
    } else if abs[2] <= abs[0] && abs[2] <= abs[1] {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let u = normalize(cross(normal, reference));
    let v = cross(normal, u);
    let mut angles: Vec<_> = points
        .iter()
        .map(|&p| {
            let d = sub(p, arithmetic);
            dot(d, v).atan2(dot(d, u))
        })
        .collect();
    for i in 0..points.len() {
        let mut first = i;
        for j in i + 1..points.len() {
            if angles[j] < angles[first] {
                first = j;
            }
        }
        points.swap(i, first);
        angles.swap(i, first);
    }
    let mut area = 0.0;
    let mut c = [0.0; 2];
    for i in 0..points.len() {
        let a = sub(points[i], arithmetic);
        let b = sub(points[(i + 1) % points.len()], arithmetic);
        let ax = dot(a, u);
        let ay = dot(a, v);
        let bx = dot(b, u);
        let by = dot(b, v);
        let cross = ax * by - bx * ay;
        area += cross;
        c[0] += cross * (ax + bx);
        c[1] += cross * (ay + by);
    }
    if area.abs() <= 1e-10 {
        return fallback;
    }
    c[0] /= 3.0 * area;
    c[1] /= 3.0 * area;
    add(add(add(center, arithmetic), scale(u, c[0])), scale(v, c[1]))
}

/// Symmetric matrix ordered xx,xy,xz,yy,yz,zz, as in the shader.
fn determinant(m: [f32; 6]) -> f32 {
    let [xx, xy, xz, yy, yz, zz] = m;
    xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
}
fn solve(m: [f32; 6], r: V, d: f32) -> V {
    let [xx, xy, xz, yy, yz, zz] = m;
    [
        (r[0] * (yy * zz - yz * yz) - xy * (r[1] * zz - yz * r[2]) + xz * (r[1] * yz - yy * r[2]))
            / d,
        (xx * (r[1] * zz - yz * r[2]) - r[0] * (xy * zz - yz * xz) + xz * (xy * r[2] - r[1] * xz))
            / d,
        (xx * (yy * r[2] - r[1] * yz) - xy * (xy * r[2] - r[1] * xz) + r[0] * (xy * yz - yy * xz))
            / d,
    ]
}
fn accumulate(m: &mut [f32; 6], r: &mut V, d: V, v: f32) {
    m[0] += d[0] * d[0];
    m[1] += d[0] * d[1];
    m[2] += d[0] * d[2];
    m[3] += d[1] * d[1];
    m[4] += d[1] * d[2];
    m[5] += d[2] * d[2];
    *r = add(*r, scale(d, v));
}
/// Free affine fit retains observable slopes at domain faces and edges.
fn affine(samples: &[(V, f32)], point: V) -> f32 {
    let count = samples.len();
    let inverse = 1.0 / count as f32;
    let mut mean = [0.0; 3];
    let mut value = 0.0;
    for &(p, v) in samples {
        mean = add(mean, p);
        value += v;
    }
    mean = scale(mean, inverse);
    value *= inverse;
    let mut m = [0.0; 6];
    let mut rhs = [0.0; 3];
    let mut deltas = Vec::with_capacity(count);
    let mut maximum = 0.0;
    let mut first = 0;
    for (i, &(p, v)) in samples.iter().enumerate() {
        let d = sub(p, mean);
        let diff = v - value;
        deltas.push((d, diff));
        let squared = dot(d, d);
        if squared > maximum {
            maximum = squared;
            first = i;
        }
        accumulate(&mut m, &mut rhs, d, diff);
    }
    let det = determinant(m);
    let ms = m[0].max(m[3].max(m[5]));
    let cutoff = ((64.0 * f32::EPSILON) * ms) * ms * ms;
    let mut gradient = [0.0; 3];
    if det.abs() > cutoff {
        gradient = solve(m, rhs, det);
    } else if maximum > 1e-12 {
        let rank_cutoff = (64.0 * f32::EPSILON) * maximum;
        let e0 = normalize(deltas[first].0);
        let mut basis = [e0, [0.0; 3], [0.0; 3]];
        let mut rank = 1;
        for axis in 1..3 {
            let mut best = 0.0;
            let mut index = 0;
            for (i, &(d, _)) in deltas.iter().enumerate() {
                let mut residual = d;
                for &e in &basis[..axis] {
                    residual = sub(residual, scale(e, dot(d, e)));
                }
                let squared = dot(residual, residual);
                if squared > best {
                    best = squared;
                    index = i;
                }
            }
            if best <= rank_cutoff {
                break;
            }
            let mut e = deltas[index].0;
            for &prior in &basis[..axis] {
                e = sub(e, scale(prior, dot(deltas[index].0, prior)));
            }
            let original = e;
            for &prior in &basis[..axis] {
                e = sub(e, scale(prior, dot(original, prior)));
            }
            basis[axis] = normalize(e);
            rank = axis + 1;
        }
        let mut b = [0.0; 6];
        let mut c = [0.0; 3];
        for &(d, diff) in &deltas {
            accumulate(
                &mut b,
                &mut c,
                [dot(d, basis[0]), dot(d, basis[1]), dot(d, basis[2])],
                diff,
            );
        }
        let mut h = [c[0] / b[0].max(1e-12), 0.0, 0.0];
        if rank >= 2 {
            let d = b[0] * b[3] - b[1] * b[1];
            if d > 1e-10 * (b[0] * b[3]).max(1e-12) {
                h = [
                    (c[0] * b[3] - b[1] * c[1]) / d,
                    (b[0] * c[1] - b[1] * c[0]) / d,
                    0.0,
                ];
            }
        }
        if rank == 3 {
            let d = determinant(b);
            if d.abs() > 1e-10 * (b[0] * b[3] * b[5]).max(1e-12) {
                h = solve(b, c, d);
            }
        }
        gradient = add(
            add(scale(basis[0], h[0]), scale(basis[1], h[1])),
            scale(basis[2], h[2]),
        );
    }
    value + dot(gradient, sub(point, mean))
}

struct Directory<'a> {
    topology: &'a CompiledTopology<3>,
    widths: Vec<i32>,
    bricks: HashMap<(i32, [i32; 3]), usize>,
}
impl<'a> Directory<'a> {
    fn new(topology: &'a CompiledTopology<3>) -> Self {
        let mut widths = Vec::new();
        let mut bricks = HashMap::new();
        for (i, b) in topology.bricks.iter().enumerate() {
            let width = (8 * b.seed.span_bricks) as i32;
            if !widths.contains(&width) {
                widths.push(width);
            }
            bricks.insert((width, b.seed.coordinate.map(|v| v * 8)), i);
        }
        widths.sort_unstable();
        Self {
            topology,
            widths,
            bricks,
        }
    }
    fn record(&self, q: [i32; 3], fields: &Fields, air: bool) -> Option<Record> {
        let g = &self.topology.graph;
        if (0..3).any(|a| q[a] < 0 || q[a] as f32 >= g.dimensions[a]) {
            return None;
        }
        let at = q[0] as usize
            + g.dimensions[0] as usize * (q[1] as usize + g.dimensions[1] as usize * q[2] as usize);
        if g.solid_voxel_fraction.get(at).is_some_and(|&v| v > 0.0) {
            return None;
        }
        if let Some(i) = owner_at(g, q.map(|v| v as f32 + 0.5)) {
            if fields.capacity[i] < 0.999999 {
                return None;
            }
            let c = &g.cells[i];
            return Some(Record {
                center: c.center,
                widths: c.widths,
                active: Some(i),
            });
        }
        if !air {
            return None;
        }
        for &width in &self.widths {
            let lower = q.map(|v| v.div_euclid(width) * width);
            if let Some(&i) = self.bricks.get(&(width, lower)) {
                let b = &self.topology.bricks[i].seed;
                if b.active {
                    continue;
                }
                return Some(air_record(b, q, g.dimensions));
            }
        }
        None
    }
}
fn air_record(b: &BrickSeed, q: [i32; 3], dimensions: V) -> Record {
    let edge = (8 * b.span_bricks) as i32;
    let scale = edge / b.resolution as i32;
    let min = std::array::from_fn::<_, 3, _>(|a| {
        let start = 8 * b.coordinate[a];
        start + (q[a] - start).div_euclid(scale) * scale
    });
    let widths =
        std::array::from_fn(|a| (min[a] + scale).min(dimensions[a] as i32) as f32 - min[a] as f32);
    Record {
        center: std::array::from_fn(|a| min[a] as f32 + 0.5 * widths[a]),
        widths,
        active: None,
    }
}
fn vertex(g: &Graph, i: usize, corner: usize) -> [i32; 3] {
    let c = &g.cells[i];
    std::array::from_fn(|a| {
        (c.center[a] - 0.5 * c.widths[a]).round() as i32
            + if corner & (1 << a) != 0 {
                c.widths[a].round() as i32
            } else {
                0
            }
    })
}
fn connected(
    directory: &Directory,
    fields: &Fields,
    i: usize,
    v: [i32; 3],
    air: bool,
) -> [Option<Record>; 8] {
    let mut records = std::array::from_fn(|octant| {
        directory.record(
            std::array::from_fn(|a| v[a] + if octant & (1 << a) == 0 { -1 } else { 0 }),
            fields,
            air,
        )
    });
    let mut reachable = records.map(|r| r.is_some_and(|r| r.active == Some(i)));
    for _ in 0..8 {
        for octant in 0..8 {
            if reachable[octant] {
                for axis in 0..3 {
                    let adjacent = octant ^ (1 << axis);
                    if records[adjacent].is_some() {
                        reachable[adjacent] = true;
                    }
                }
            }
        }
    }
    for octant in 0..8 {
        if !reachable[octant] {
            records[octant] = None;
        }
    }
    records
}
fn contribution(target: V, source: usize, g: &Graph, planes: &[Interface]) -> [f32; 2] {
    let p = planes[source];
    if !p.valid {
        return [0.0; 2];
    }
    let delta = sub(target, p.centroid);
    let d = distance(p, sub(target, g.cells[source].center));
    let weight = d * d / dot(delta, delta).max(1e-12);
    [weight * d, weight]
}
fn phase(i: usize, p: V, g: &Graph, f: &Fields, planes: &[Interface]) -> i32 {
    let fill = (f.density[i] / f.capacity[i].max(1e-6)).clamp(0.0, 1.0);
    if fill >= 1.0 - 1e-6 {
        return -1;
    }
    if fill <= 1e-6 {
        return 1;
    }
    if !planes[i].valid {
        return 0;
    }
    if distance(planes[i], sub(p, g.cells[i].center)) <= 0.0 {
        -1
    } else {
        1
    }
}
fn bound_owner(i: usize, p: V, value: f32, g: &Graph, f: &Fields, planes: &[Interface]) -> f32 {
    let phase = phase(i, p, g, f, planes);
    if phase == 0 {
        return value;
    }
    if planes[i].valid {
        return bound(value, distance(planes[i], sub(p, g.cells[i].center)), phase);
    }
    let c = &g.cells[i];
    let interior = std::array::from_fn(|a| (p[a] - c.minimum[a]).min(c.maximum[a] - p[a]));
    let margin = min_width(interior).max(0.0);
    bound(value, if phase < 0 { -margin } else { margin }, phase)
}

pub struct Rdf3d {
    pub center_values: Vec<f32>,
    pub corner_values: Vec<[[f32; 2]; 8]>,
}
impl Rdf3d {
    pub fn reconstruct(
        topology: &CompiledTopology<3>,
        fields: &Fields,
    ) -> Result<Self, ValidationError> {
        let g = &topology.graph;
        fields.validate_for(g)?;
        if g.dimension != 3 {
            return Err(ValidationError("3D RDF requires dimension 3".into()));
        }
        let directory = Directory::new(topology);
        let planes: Vec<_> = g
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let normal = std::array::from_fn(|a| fields.interface_normal[3 * i + a]);
                let offset = fields.interface_offset[i];
                let valid = dot(normal, normal) > 1e-12 && offset.is_finite();
                Interface {
                    normal,
                    offset,
                    valid,
                    centroid: if valid {
                        centroid(c.center, c.widths, normal, offset)
                    } else {
                        c.center
                    },
                }
            })
            .collect();
        let mut values = vec![0.0; g.cells.len()];
        for (i, c) in g.cells.iter().enumerate() {
            let fill = (fields.density[i] / fields.capacity[i].max(1e-6)).clamp(0.0, 1.0);
            let fallback = (LIQUID_ISOVALUE - fill) * 4.0 * min_width(c.widths);
            if fields.capacity[i] < 0.999999 {
                values[i] = fallback;
                continue;
            }
            let mut neighbors = Vec::with_capacity(64);
            for corner in 0..8 {
                for record in connected(&directory, fields, i, vertex(g, i, corner), false)
                    .into_iter()
                    .flatten()
                {
                    let j = record.active.unwrap();
                    if !neighbors.contains(&j) {
                        neighbors.push(j);
                    }
                }
            }
            let mut sum = [0.0; 2];
            for j in neighbors {
                let v = contribution(c.center, j, g, &planes);
                sum[0] += v[0];
                sum[1] += v[1];
            }
            let value = if sum[1] > 1e-8 {
                sum[0] / sum[1]
            } else {
                fallback
            };
            values[i] = bound_owner(i, c.center, value, g, fields, &planes);
        }
        let mut cache: HashMap<([i32; 3], u8), [f32; 2]> = HashMap::new();
        let mut corners = vec![[[0.0; 2]; 8]; g.cells.len()];
        for i in 0..g.cells.len() {
            if fields.capacity[i] < 0.999999 {
                continue;
            }
            for corner in 0..8 {
                let v = vertex(g, i, corner);
                let records = connected(&directory, fields, i, v, true);
                let mask = records
                    .iter()
                    .enumerate()
                    .fold(0u8, |m, (j, r)| m | if r.is_some() { 1 << j } else { 0 });
                corners[i][corner] = *cache.entry((v, mask)).or_insert_with(|| {
                    let mut active = Vec::with_capacity(8);
                    for r in records.into_iter().flatten() {
                        if let Some(j) = r.active {
                            if !active.contains(&j) {
                                active.push(j);
                            }
                        }
                    }
                    if active.is_empty() {
                        return [0.0; 2];
                    }
                    let mut samples = Vec::with_capacity(8);
                    for &j in &active {
                        samples.push((g.cells[j].center, values[j]));
                    }
                    for r in records.into_iter().flatten().filter(|r| r.active.is_none()) {
                        if samples
                            .iter()
                            .any(|&(p, _)| dot(sub(p, r.center), sub(p, r.center)) <= 1e-10)
                        {
                            continue;
                        }
                        let mut sum = [0.0; 2];
                        for &j in &active {
                            let v = contribution(r.center, j, g, &planes);
                            sum[0] += v[0];
                            sum[1] += v[1];
                        }
                        let margin = 0.5 * min_width(r.widths);
                        let fallback = 4.0 * LIQUID_ISOVALUE * min_width(r.widths);
                        samples.push((
                            r.center,
                            margin.max(if sum[1] > 1e-8 {
                                sum[0] / sum[1]
                            } else {
                                fallback
                            }),
                        ));
                    }
                    let point = v.map(|a| a as f32);
                    let mut value = affine(&samples, point);
                    let liquid = active
                        .iter()
                        .any(|&j| phase(j, point, g, fields, &planes) < 0);
                    let air = samples.len() > active.len()
                        || active
                            .iter()
                            .any(|&j| phase(j, point, g, fields, &planes) > 0);
                    if liquid && !air {
                        value = value.min(0.0);
                    } else if air && !liquid {
                        value = value.max(0.0);
                    }
                    [value, 1.0]
                });
            }
        }
        Ok(Self {
            center_values: values,
            corner_values: corners,
        })
    }
    pub fn sample(&self, i: usize, p: V, g: &Graph, f: &Fields) -> Option<f32> {
        if f.capacity[i] < 0.999999 {
            return None;
        }
        let c = &g.cells[i];
        let fraction: V =
            std::array::from_fn(|a| ((p[a] - c.minimum[a]) / c.widths[a]).clamp(0.0, 1.0));
        let mut sum = [0.0; 2];
        for corner in 0..8 {
            let w: V = std::array::from_fn(|a| {
                if corner & (1 << a) != 0 {
                    fraction[a]
                } else {
                    1.0 - fraction[a]
                }
            });
            let weight = w[0] * w[1] * w[2];
            sum[0] += weight * self.corner_values[i][corner][0];
            sum[1] += weight * self.corner_values[i][corner][1];
        }
        if sum[1] <= 1e-8 {
            return None;
        }
        let normal: V = std::array::from_fn(|a| f.interface_normal[3 * i + a]);
        let valid = dot(normal, normal) > 1e-12;
        let fill = (f.density[i] / f.capacity[i].max(1e-6)).clamp(0.0, 1.0);
        let accepted = dot(normal, sub(p, c.center)) - f.interface_offset[i];
        let phase = if fill >= 1.0 - 1e-6 {
            -1
        } else if fill <= 1e-6 {
            1
        } else if valid {
            if accepted <= 0.0 {
                -1
            } else {
                1
            }
        } else {
            0
        };
        let margin = min_width(std::array::from_fn(|a| {
            (p[a] - c.minimum[a]).min(c.maximum[a] - p[a])
        }))
        .max(0.0);
        Some(bound(
            sum[0] / sum[1],
            if valid {
                accepted
            } else if phase < 0 {
                -margin
            } else {
                margin
            },
            phase,
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_vertices_reproduce_a_planar_free_surface() {
        use crate::geometry::BoundaryMode;
        use crate::topology::{compile_topology, TopologySeed};
        let topology = compile_topology::<3>(TopologySeed {
            dimensions: [8; 3],
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks: vec![BrickSeed {
                id: 0,
                key: 0,
                coordinate: [0; 3],
                span_bricks: 1,
                resolution: 8,
                active: true,
                density: vec![],
                gamma: vec![],
                refinement_region_scale: None,
            }],
        })
        .unwrap();
        let g = &topology.graph;
        let n = g.cells.len();
        let r = g.rows.len();
        let mut f = Fields {
            density: vec![0.0; n],
            gamma: vec![1.0; n],
            capacity: vec![1.0; n],
            cell_velocity: vec![0.0; 3 * n],
            face_velocity: vec![0.0; r],
            pressure: vec![0.0; n],
            pressure_rhs: vec![0.0; n],
            pressure_diagonal: vec![0.0; n],
            pressure_member: vec![0; n],
            extension_depth: vec![0; n],
            interface_normal: vec![0.0; 3 * n],
            interface_offset: vec![0.0; n],
            ..Default::default()
        };
        for c in &g.cells {
            let i = c.id as usize;
            f.capacity[i] = 1.0;
            f.density[i] = (3.7 - c.minimum[1]).clamp(0.0, 1.0);
            if f.density[i] > 0.0 && f.density[i] < 1.0 {
                f.interface_normal[3 * i + 1] = 1.0;
                f.interface_offset[i] = 3.7 - c.center[1];
            }
        }
        let rdf = Rdf3d::reconstruct(&topology, &f).unwrap();
        for x in 2..6 {
            for z in 2..6 {
                for y in [3.4, 3.6, 3.8] {
                    let p = [x as f32 + 0.5, y, z as f32 + 0.5];
                    let i = owner_at(g, p).unwrap();
                    let value = rdf.sample(i, p, g, &f).unwrap();
                    assert!((value - (y - 3.7)).abs() < 2e-6, "{p:?}: {value}");
                }
            }
        }
    }
    #[test]
    fn affine_retains_observable_boundary_slopes() {
        for samples in [
            vec![([0.0, 0.0, 0.0], 1.0), ([2.0, 0.0, 0.0], 5.0)],
            vec![
                ([0.0, 0.0, 0.0], 1.0),
                ([2.0, 0.0, 0.0], 5.0),
                ([0.0, 2.0, 0.0], 7.0),
            ],
            vec![
                ([0.0, 0.0, 0.0], 1.0),
                ([2.0, 0.0, 0.0], 5.0),
                ([0.0, 2.0, 0.0], 7.0),
                ([0.0, 0.0, 2.0], 9.0),
            ],
        ] {
            let p = if samples.len() == 2 {
                [1.0, 0.0, 0.0]
            } else if samples.len() == 3 {
                [1.0, 1.0, 0.0]
            } else {
                [1.0, 1.0, 1.0]
            };
            assert!(
                (affine(&samples, p) - (1.0 + 2.0 * p[0] + 3.0 * p[1] + 4.0 * p[2])).abs() < 2e-6
            );
        }
    }
    #[test]
    fn box_polygon_centroid_is_on_plane_and_axis_permutation_equivariant() {
        let p = centroid([2.0, 3.0, 4.0], [2.0, 4.0, 6.0], [1.0, 0.0, 0.0], 0.3);
        assert!((p[0] - 2.3).abs() < 1e-6);
        assert_eq!(p[1], 3.0);
        assert_eq!(p[2], 4.0);
        let n = normalize([1.0, 2.0, 3.0]);
        let c = centroid([0.0; 3], [2.0; 3], n, 0.4);
        assert!((dot(n, c) - 0.4).abs() < 1e-6);
    }
}
