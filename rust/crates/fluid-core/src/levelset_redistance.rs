//! Exact signed-distance queries against the accepted 2-D RDF contour.
//!
//! The published RDF remains unchanged.  This module builds a small BVH over
//! its explicit line segments and uses the RDF's own centre-fan scalar field
//! only to choose the side of the contour.

use crate::presentation::RdfSurface;
use crate::types::ValidationError;

#[derive(Clone, Copy)]
struct Segment {
    a: [f64; 2],
    b: [f64; 2],
    minimum: [f64; 2],
    maximum: [f64; 2],
}

#[derive(Clone, Copy)]
enum Node {
    Leaf { minimum: [f64; 2], maximum: [f64; 2], start: usize, end: usize },
    Branch { minimum: [f64; 2], maximum: [f64; 2], left: usize, right: usize },
}

pub struct RedistanceField<'a> {
    surface: &'a RdfSurface,
    segments: Vec<Segment>,
    order: Vec<usize>,
    nodes: Vec<Node>,
    root: Option<usize>,
}

impl<'a> RedistanceField<'a> {
    pub fn new(surface: &'a RdfSurface) -> Result<Self, ValidationError> {
        if surface.segments_fine.len() % 4 != 0 {
            return Err(ValidationError("RDF contour has an incomplete segment".into()));
        }
        let mut segments = Vec::with_capacity(surface.segments_fine.len() / 4);
        for coordinates in surface.segments_fine.chunks_exact(4) {
            if coordinates.iter().any(|value| !value.is_finite()) {
                return Err(ValidationError("RDF contour has a non-finite segment".into()));
            }
            let a = [coordinates[0] as f64, coordinates[1] as f64];
            let b = [coordinates[2] as f64, coordinates[3] as f64];
            segments.push(Segment {
                a,
                b,
                minimum: [a[0].min(b[0]), a[1].min(b[1])],
                maximum: [a[0].max(b[0]), a[1].max(b[1])],
            });
        }
        let mut order: Vec<_> = (0..segments.len()).collect();
        let mut nodes = Vec::new();
        let root = if order.is_empty() {
            None
        } else {
            Some(build_node(&segments, &mut order, 0, &mut nodes))
        };
        Ok(Self { surface, segments, order, nodes, root })
    }

    pub fn segment_count(&self) -> usize { self.segments.len() }

    /// Returns None where the accepted RDF has no contour or cannot prove a
    /// finite phase at the query point.  Callers retain their existing finite
    /// fallback in that case; redistancing never invents a phase.
    pub fn sample(&self, point: [f32; 2]) -> Option<f32> {
        let root = self.root?;
        let sign = sample_scalar(self.surface, point)?;
        if sign == 0.0 { return Some(0.0); }
        let p = [point[0] as f64, point[1] as f64];
        let mut squared = f64::INFINITY;
        nearest(root, &self.nodes, &self.order, &self.segments, p, &mut squared);
        if !squared.is_finite() { return None; }
        let distance = squared.sqrt() as f32;
        Some(if sign < 0.0 { -distance } else { distance })
    }
}

pub fn sample_scalar(surface: &RdfSurface, point: [f32; 2]) -> Option<f32> {
    let [nx, ny] = surface.dimensions.map(|value| value as usize);
    if nx == 0 || ny == 0 || surface.vertex_phi_fine.len() != (nx + 1) * (ny + 1) {
        return None;
    }
    let x = (point[0] as f64).clamp(0.0, nx as f64);
    let y = (point[1] as f64).clamp(0.0, ny as f64);
    let ix = (x.floor() as usize).min(nx - 1);
    let iy = (y.floor() as usize).min(ny - 1);
    let u = x - ix as f64;
    let v = y - iy as f64;
    let stride = nx + 1;
    let values = [
        surface.vertex_phi_fine[ix + stride * iy] as f64,
        surface.vertex_phi_fine[ix + 1 + stride * iy] as f64,
        surface.vertex_phi_fine[ix + 1 + stride * (iy + 1)] as f64,
        surface.vertex_phi_fine[ix + stride * (iy + 1)] as f64,
    ];
    if values.iter().any(|value| !value.is_finite()) { return None; }
    let centre = values.iter().sum::<f64>() * 0.25;
    let value = if v <= u && v <= 1.0 - u {
        // bottom: (0,0), (1,0), centre
        values[0] * (1.0 - u - v) + values[1] * (u - v) + centre * (2.0 * v)
    } else if u >= v && u >= 1.0 - v {
        // right: (1,0), (1,1), centre
        values[1] * (u - v) + values[2] * (u + v - 1.0) + centre * (2.0 * (1.0 - u))
    } else if v >= u && v >= 1.0 - u {
        // top: (1,1), (0,1), centre
        values[2] * (u + v - 1.0) + values[3] * (v - u) + centre * (2.0 * (1.0 - v))
    } else {
        // left: (0,1), (0,0), centre
        values[3] * (v - u) + values[0] * (1.0 - u - v) + centre * (2.0 * u)
    };
    value.is_finite().then_some(value as f32)
}

fn bounds(segments: &[Segment], order: &[usize]) -> ([f64; 2], [f64; 2]) {
    let mut minimum = [f64::INFINITY; 2];
    let mut maximum = [f64::NEG_INFINITY; 2];
    for &id in order {
        for axis in 0..2 {
            minimum[axis] = minimum[axis].min(segments[id].minimum[axis]);
            maximum[axis] = maximum[axis].max(segments[id].maximum[axis]);
        }
    }
    (minimum, maximum)
}

fn build_node(segments: &[Segment], order: &mut [usize], offset: usize, nodes: &mut Vec<Node>) -> usize {
    let (minimum, maximum) = bounds(segments, order);
    if order.len() <= 8 {
        let id = nodes.len();
        nodes.push(Node::Leaf { minimum, maximum, start: offset, end: offset + order.len() });
        return id;
    }
    let axis = usize::from(maximum[1] - minimum[1] > maximum[0] - minimum[0]);
    order.sort_unstable_by(|a, b| {
        let ca = segments[*a].a[axis] + segments[*a].b[axis];
        let cb = segments[*b].a[axis] + segments[*b].b[axis];
        ca.total_cmp(&cb)
    });
    let middle = order.len() / 2;
    let (left_order, right_order) = order.split_at_mut(middle);
    let left = build_node(segments, left_order, offset, nodes);
    let right = build_node(segments, right_order, offset + middle, nodes);
    let id = nodes.len();
    nodes.push(Node::Branch { minimum, maximum, left, right });
    id
}

fn bounds_distance_squared(minimum: [f64; 2], maximum: [f64; 2], p: [f64; 2]) -> f64 {
    (0..2).map(|axis| {
        let delta = if p[axis] < minimum[axis] { minimum[axis] - p[axis] }
            else if p[axis] > maximum[axis] { p[axis] - maximum[axis] }
            else { 0.0 };
        delta * delta
    }).sum()
}

fn segment_distance_squared(segment: Segment, p: [f64; 2]) -> f64 {
    let d = [segment.b[0] - segment.a[0], segment.b[1] - segment.a[1]];
    let length_squared = d[0] * d[0] + d[1] * d[1];
    let t = if length_squared > 0.0 {
        (((p[0] - segment.a[0]) * d[0] + (p[1] - segment.a[1]) * d[1]) / length_squared).clamp(0.0, 1.0)
    } else { 0.0 };
    let dx = p[0] - (segment.a[0] + t * d[0]);
    let dy = p[1] - (segment.a[1] + t * d[1]);
    dx * dx + dy * dy
}

fn nearest(node: usize, nodes: &[Node], order: &[usize], segments: &[Segment], p: [f64; 2], best: &mut f64) {
    match nodes[node] {
        Node::Leaf { minimum, maximum, start, end } => {
            if bounds_distance_squared(minimum, maximum, p) >= *best { return; }
            for &id in &order[start..end] { *best = (*best).min(segment_distance_squared(segments[id], p)); }
        }
        Node::Branch { minimum, maximum, left, right } => {
            if bounds_distance_squared(minimum, maximum, p) >= *best { return; }
            let distance = |id| match nodes[id] {
                Node::Leaf { minimum, maximum, .. } | Node::Branch { minimum, maximum, .. } => bounds_distance_squared(minimum, maximum, p),
            };
            let (first, second) = if distance(left) <= distance(right) { (left, right) } else { (right, left) };
            nearest(first, nodes, order, segments, p, best);
            nearest(second, nodes, order, segments, p, best);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presentation::RdfReceipt;

    fn surface(vertices: Vec<f32>, segments: Vec<f32>) -> RdfSurface {
        RdfSurface { dimensions: [2, 2], vertex_phi_fine: vertices, segments_fine: segments, receipt: RdfReceipt::default() }
    }

    #[test]
    fn planar_distance_uses_physical_fine_coordinates() {
        let source = surface(
            vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 2.0],
        );
        let field = RedistanceField::new(&source).unwrap();
        assert_eq!(field.sample([1.0, 0.75]), Some(0.0));
        assert!((field.sample([0.25, 1.0]).unwrap() + 0.75).abs() < 1e-6);
        assert!((field.sample([1.75, 1.0]).unwrap() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn enclosed_air_sign_is_preserved() {
        let source = surface(
            vec![-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
            vec![0.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 0.5],
        );
        let field = RedistanceField::new(&source).unwrap();
        assert!(field.sample([1.0, 1.0]).unwrap() > 0.0);
        assert!(field.sample([0.1, 0.1]).unwrap() < 0.0);
    }

    #[test]
    fn empty_interface_does_not_invent_a_phase() {
        let source = surface(vec![1.0; 9], Vec::new());
        let field = RedistanceField::new(&source).unwrap();
        assert_eq!(field.segment_count(), 0);
        assert_eq!(field.sample([1.0, 1.0]), None);
    }

    #[test]
    fn branched_bvh_matches_brute_force_without_mutating_source() {
        let mut segments = Vec::new();
        for x in 0..12 {
            segments.extend([x as f32 * 0.15, 0.1, x as f32 * 0.15, 1.9]);
        }
        let source = surface(vec![1.0; 9], segments);
        let vertices_before = source.vertex_phi_fine.clone();
        let segments_before = source.segments_fine.clone();
        let field = RedistanceField::new(&source).unwrap();
        for point in [[0.07, 0.3], [0.83, 1.2], [1.71, 1.8]] {
            let expected = field.segments.iter()
                .map(|segment| segment_distance_squared(*segment, point.map(f64::from)))
                .fold(f64::INFINITY, f64::min).sqrt() as f32;
            assert!((field.sample(point).unwrap() - expected).abs() < 1e-6);
        }
        assert_eq!(source.vertex_phi_fine, vertices_before);
        assert_eq!(source.segments_fine, segments_before);
    }

    #[test]
    fn polygonal_sphere_distance_preserves_inside_and_outside() {
        let dimensions = [8_u32, 8_u32];
        let vertices = (0..=8)
            .flat_map(|y| (0..=8).map(move |x| ((x as f32 - 4.0).hypot(y as f32 - 4.0)) - 2.0))
            .collect();
        let mut segments = Vec::new();
        for i in 0..16 {
            let angle_a = std::f64::consts::TAU * i as f64 / 16.0;
            let angle_b = std::f64::consts::TAU * (i + 1) as f64 / 16.0;
            segments.extend([
                (4.0 + 2.0 * angle_a.cos()) as f32,
                (4.0 + 2.0 * angle_a.sin()) as f32,
                (4.0 + 2.0 * angle_b.cos()) as f32,
                (4.0 + 2.0 * angle_b.sin()) as f32,
            ]);
        }
        let source = RdfSurface {
            dimensions,
            vertex_phi_fine: vertices,
            segments_fine: segments,
            receipt: RdfReceipt::default(),
        };
        let field = RedistanceField::new(&source).unwrap();
        assert!(field.sample([4.0, 4.0]).unwrap() < -1.9);
        assert!((field.sample([7.0, 4.0]).unwrap() - 1.0).abs() < 1e-6);
        for segment in source.segments_fine.chunks_exact(4) {
            let midpoint = [0.5 * (segment[0] + segment[2]), 0.5 * (segment[1] + segment[3])];
            assert!(field.sample(midpoint).unwrap().abs() < 1e-6);
        }
    }
}
