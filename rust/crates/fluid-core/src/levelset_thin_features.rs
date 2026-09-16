//! Detect represented liquid sheets and air gaps before adaptive restriction.
//! Opposing contour segments are tested along their normals, independently of
//! solver cell centres, density residue and the orientation of the grid.
use crate::levelset_redistance::sample_scalar;
use crate::presentation::RdfSurface;
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Copy)]
struct Segment {
    a: [f64; 2],
    b: [f64; 2],
    midpoint: [f64; 2],
    normal: [f64; 2],
}
fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

/// Return physical brick coordinates touched by a thin represented feature.
/// Buckets bound the search to the requested physical thickness, not all pairs
/// of contour segments. A zero width disables this additional detector.
pub(crate) fn protected_bricks(surface: &RdfSurface, width: f64) -> BTreeSet<[i32; 2]> {
    let mut protected = BTreeSet::new();
    if !width.is_finite() || width <= 0.0 {
        return protected;
    }
    let bucket_width = width.max(1.0);
    let bucket = |p: [f64; 2]| p.map(|v| (v / bucket_width).floor() as i32);
    let mut segments = Vec::new();
    let mut buckets: HashMap<[i32; 2], Vec<usize>> = HashMap::new();
    for s in surface.segments_fine.chunks_exact(4) {
        let a = [s[0] as f64, s[1] as f64];
        let b = [s[2] as f64, s[3] as f64];
        let midpoint = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        let length = (b[0] - a[0]).hypot(b[1] - a[1]);
        if length < 1e-10 {
            continue;
        }
        let mut normal = [-(b[1] - a[1]) / length, (b[0] - a[0]) / length];
        let sample = |sign: f64| {
            sample_scalar(
                surface,
                std::array::from_fn(|axis| (midpoint[axis] + sign * 1e-3 * normal[axis]) as f32),
            )
        };
        match (sample(1.0), sample(-1.0)) {
            (Some(plus), Some(minus)) if plus < minus => normal = normal.map(|v| -v),
            (Some(_), Some(_)) => {}
            _ => continue,
        }
        let id = segments.len();
        segments.push(Segment {
            a,
            b,
            midpoint,
            normal,
        });
        // Include every bucket touched by the segment, so a midpoint near a
        // bucket edge does not hide an intersection with its endpoint.
        let lo = bucket([a[0].min(b[0]), a[1].min(b[1])]);
        let hi = bucket([a[0].max(b[0]), a[1].max(b[1])]);
        for y in lo[1]..=hi[1] {
            for x in lo[0]..=hi[0] {
                buckets.entry([x, y]).or_default().push(id);
            }
        }
    }
    for (id, segment) in segments.iter().enumerate() {
        let home = bucket(segment.midpoint);
        for by in home[1] - 1..=home[1] + 1 {
            for bx in home[0] - 1..=home[0] + 1 {
                let Some(candidates) = buckets.get(&[bx, by]) else {
                    continue;
                };
                for &other in candidates {
                    if other == id {
                        continue;
                    }
                    let other = segments[other];
                    if segment.normal[0] * other.normal[0] + segment.normal[1] * other.normal[1]
                        > -0.25
                    {
                        continue;
                    }
                    let edge = [other.b[0] - other.a[0], other.b[1] - other.a[1]];
                    let offset = [
                        other.a[0] - segment.midpoint[0],
                        other.a[1] - segment.midpoint[1],
                    ];
                    let det = cross(segment.normal, edge);
                    if det.abs() < 1e-10 {
                        continue;
                    }
                    let t = cross(offset, edge) / det;
                    let u = cross(offset, segment.normal) / det;
                    if t.abs() <= 1e-6 || t.abs() > width || !(-1e-8..=1.0 + 1e-8).contains(&u) {
                        continue;
                    }
                    let hit = [
                        segment.midpoint[0] + t * segment.normal[0],
                        segment.midpoint[1] + t * segment.normal[1],
                    ];
                    // Preserve both faces and the material/gap between them. This
                    // also protects both bricks when the feature straddles a seam.
                    let lo = std::array::from_fn::<_, 2, _>(|axis| {
                        ((segment.a[axis].min(segment.b[axis]).min(hit[axis]) - 1e-5) / 8.0).floor()
                            as i32
                    });
                    let hi = std::array::from_fn::<_, 2, _>(|axis| {
                        ((segment.a[axis].max(segment.b[axis]).max(hit[axis]) + 1e-5) / 8.0).floor()
                            as i32
                    });
                    for y in lo[1]..=hi[1] {
                        for x in lo[0]..=hi[0] {
                            protected.insert([x, y]);
                        }
                    }
                }
            }
        }
    }
    protected
}

#[cfg(test)]
mod tests {
    use super::*;
    fn surface(f: impl Fn(f32, f32) -> f32) -> RdfSurface {
        let mut values = Vec::new();
        for y in 0..=16 {
            for x in 0..=16 {
                values.push(f(x as f32, y as f32));
            }
        }
        crate::levelset_surface::publish([16, 16], values, 0.0).unwrap()
    }
    #[test]
    fn preserves_oblique_sheets_air_gaps_and_off_centre_drops() {
        for sign in [1.0, -1.0] {
            let s = surface(|x, y| sign * ((x + y - 15.0).abs() / 2.0_f32.sqrt() - 0.55));
            assert!(!protected_bricks(&s, 2.0).is_empty());
        }
        let s = surface(|x, y| (x - 8.0).hypot(y - 8.0) - 0.4);
        let protected = protected_bricks(&s, 2.0);
        for brick in [[0, 0], [1, 0], [0, 1], [1, 1]] {
            assert!(protected.contains(&brick));
        }
    }
    #[test]
    fn planar_pool_and_broad_sheet_can_coarsen() {
        assert!(protected_bricks(&surface(|_, y| y - 8.0), 2.0).is_empty());
        assert!(protected_bricks(&surface(|_, y| (y - 8.0).abs() - 3.0), 2.0).is_empty());
    }
}
