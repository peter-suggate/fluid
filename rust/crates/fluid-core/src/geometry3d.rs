//! f32 image of `geometric-interface.wgsl.ts` for true 3-D PLIC geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Plane3 {
    pub normal: [f32; 3],
    pub offset: f32,
}

#[inline]
fn add(a: f32, b: f32) -> f32 {
    a + b
}
#[inline]
fn mul(a: f32, b: f32) -> f32 {
    a * b
}
#[inline]
fn div(a: f32, b: f32) -> f32 {
    a / b
}

fn uniform_sum_primitive(x: f32, a: f32, b: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a {
        return div(mul(mul(x, div(x, a)), div(x, b)), 6.0);
    }
    if x <= b {
        return div(add(mul(0.5, mul(x, x - a)), div(mul(a, a), 6.0)), b);
    }
    if x < add(a, b) {
        let tail = add(a, b) - x;
        return add(x, -mul(0.5, add(a, b))) + div(mul(mul(tail, div(tail, a)), div(tail, b)), 6.0);
    }
    add(x, -mul(0.5, add(a, b)))
}

/// Fraction of a centred axis-aligned box satisfying n.x <= offset.
pub fn plane_box_fraction(normal: [f32; 3], offset: f32, widths: [f32; 3]) -> f32 {
    let projected = [
        mul(normal[0].abs(), widths[0]),
        mul(normal[1].abs(), widths[1]),
        mul(normal[2].abs(), widths[2]),
    ];
    let dominant = projected[0].max(projected[1].max(projected[2]));
    if dominant <= 1e-20 {
        return if offset >= 0.0 { 1.0 } else { 0.0 };
    }
    let mut spans = [0.0; 3];
    let mut dimensions = 0;
    for value in projected.map(|v| div(v, dominant)) {
        if value >= 1e-6 {
            spans[dimensions] = value;
            dimensions += 1
        }
    }
    let total = add(add(spans[0], spans[1]), spans[2]);
    let shifted = add(div(offset, dominant), mul(0.5, total));
    if shifted <= 0.0 {
        return 0.0;
    }
    if shifted >= total {
        return 1.0;
    }
    let complement = shifted > mul(0.5, total);
    let x = if complement { total - shifted } else { shifted };
    let fraction = match dimensions {
        1 => div(x, spans[0]),
        2 => {
            let a = spans[0].min(spans[1]);
            let b = spans[0].max(spans[1]);
            if x < a {
                mul(mul(0.5, div(x, a)), div(x, b))
            } else {
                div(x - mul(0.5, a), b)
            }
        }
        _ => {
            let a = spans[0].min(spans[1].min(spans[2]));
            let c = spans[0].max(spans[1].max(spans[2]));
            let b = spans[0]
                .min(spans[1])
                .max(spans[0].max(spans[1]).min(spans[2]));
            div(
                uniform_sum_primitive(x, a, b) - uniform_sum_primitive(x - c, a, b),
                c,
            )
        }
    }
    .clamp(0.0, 1.0);
    if complement {
        1.0 - fraction
    } else {
        fraction
    }
}

/// WGSL-identical analytic 1/2-D inverse and 28-step 3-D inverse.
pub fn plane_box_offset(normal: [f32; 3], widths: [f32; 3], fill: f32) -> f32 {
    let p = [
        mul(normal[0].abs(), widths[0]),
        mul(normal[1].abs(), widths[1]),
        mul(normal[2].abs(), widths[2]),
    ];
    let dominant = p[0].max(p[1].max(p[2]));
    let radius = mul(0.5, add(add(p[0], p[1]), p[2]));
    if fill <= 0.0 {
        return -radius;
    }
    if fill >= 1.0 {
        return radius;
    }
    if fill == 0.5 {
        return 0.0;
    }
    if dominant <= 1e-20 {
        return 0.0;
    }
    let mut spans = [0.0; 3];
    let mut dimensions = 0;
    for value in p.map(|v| div(v, dominant)) {
        if value >= 1e-6 {
            spans[dimensions] = value;
            dimensions += 1
        }
    }
    if dimensions == 1 {
        return mul(mul(fill - 0.5, spans[0]), dominant);
    }
    let complement = fill > 0.5;
    let target = if complement { 1.0 - fill } else { fill };
    if dimensions == 2 {
        let a = spans[0].min(spans[1]);
        let b = spans[0].max(spans[1]);
        let mut shifted = add(mul(target, b), mul(0.5, a));
        if target < div(mul(0.5, a), b) {
            shifted = mul(mul(mul(2.0, target), a), b).sqrt()
        }
        let offset = mul(shifted - mul(0.5, add(a, b)), dominant);
        return if complement { -offset } else { offset };
    }
    let (mut lo, mut hi) = (-radius, 0.0);
    for _ in 0..28 {
        let middle = add(lo, mul(0.5, hi - lo));
        if middle == lo || middle == hi {
            break;
        }
        if plane_box_fraction(normal, middle, widths) < target {
            lo = middle
        } else {
            hi = middle
        }
    }
    let offset = add(lo, mul(0.5, hi - lo));
    if complement {
        -offset
    } else {
        offset
    }
}

pub fn interface_from_fill(fill: f32, gradient: [f32; 3], widths: [f32; 3]) -> Plane3 {
    let maximum = gradient[0]
        .abs()
        .max(gradient[1].abs().max(gradient[2].abs()));
    if !(maximum > 1e-20) {
        return Plane3::default();
    }
    let scaled = [
        div(gradient[0], maximum),
        div(gradient[1], maximum),
        div(gradient[2], maximum),
    ];
    let length = add(
        add(mul(scaled[0], scaled[0]), mul(scaled[1], scaled[1])),
        mul(scaled[2], scaled[2]),
    )
    .sqrt();
    let normal = [
        div(scaled[0], length),
        div(scaled[1], length),
        div(scaled[2], length),
    ];
    Plane3 {
        normal,
        offset: plane_box_offset(normal, widths, fill),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complement_permutation_and_inverse() {
        let n = [0.31, -0.72, 0.44];
        let w = [0.7, 1.3, 2.1];
        for fill in [0.01, 0.17, 0.5, 0.83, 0.99] {
            let o = plane_box_offset(n, w, fill);
            assert!((plane_box_fraction(n, o, w) - fill).abs() <= 2.0 * f32::EPSILON);
            assert!(
                (plane_box_fraction(n, o, w) - (1.0 - plane_box_fraction(n, -o, w))).abs() < 2e-6
            );
            let np = [n[2], n[0], n[1]];
            let wp = [w[2], w[0], w[1]];
            assert!((plane_box_fraction(n, o, w) - plane_box_fraction(np, o, wp)).abs() < 2e-6)
        }
    }
}
