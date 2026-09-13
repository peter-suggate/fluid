//! Strict scalar kernels shared by native and Wasm builds.

use crate::types::PRESSURE_REDUCTION_LANES;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A 64-lane reduction group is deliberately small to preserve the resident
/// reduction tree.  Dispatching fewer groups than this costs more than doing
/// the trees on the calling thread in the Wasm worker pool.
pub const PARALLEL_PRESSURE_GROUP_THRESHOLD: usize = 64;
pub const PARALLEL_POINTWISE_THRESHOLD: usize = 8192;
#[cfg(feature = "parallel")]
const PARALLEL_POINTWISE_CHUNK: usize = 2048;

#[inline(always)]
pub fn add(a: f32, b: f32) -> f32 {
    a + b
}
#[inline(always)]
pub fn mul(a: f32, b: f32) -> f32 {
    a * b
}
#[inline(always)]
pub fn div(a: f32, b: f32) -> f32 {
    a / b
}

/// Resident-compatible two-level, 64-lane reduction tree.
pub fn reduce_production(values: &[f32], execution_order: &[u32]) -> f32 {
    let groups = execution_order.len().div_ceil(PRESSURE_REDUCTION_LANES);
    let mut partials = vec![0.0f32; groups];
    let reduce_group = |group: usize| {
        let mut lanes = [0.0f32; PRESSURE_REDUCTION_LANES];
        for (lane, value) in lanes.iter_mut().enumerate() {
            let ordinal = group * PRESSURE_REDUCTION_LANES + lane;
            if let Some(&id) = execution_order.get(ordinal) {
                *value = values[id as usize];
            }
        }
        let mut width = 32;
        while width >= 1 {
            for lane in 0..width {
                lanes[lane] = add(lanes[lane], lanes[lane + width]);
            }
            width >>= 1;
        }
        lanes[0]
    };
    #[cfg(feature = "parallel")]
    if groups >= PARALLEL_PRESSURE_GROUP_THRESHOLD {
        partials
            .par_iter_mut()
            .enumerate()
            .for_each(|(group, partial)| *partial = reduce_group(group));
    } else {
        for (group, partial) in partials.iter_mut().enumerate() {
            *partial = reduce_group(group);
        }
    }
    #[cfg(not(feature = "parallel"))]
    for (group, partial) in partials.iter_mut().enumerate() {
        *partial = reduce_group(group);
    }
    let mut lanes = [0.0f32; PRESSURE_REDUCTION_LANES];
    for (lane, value) in lanes.iter_mut().enumerate() {
        let mut sum = 0.0;
        let mut at = lane;
        while at < groups {
            sum = add(sum, partials[at]);
            at += PRESSURE_REDUCTION_LANES;
        }
        *value = sum;
    }
    let mut width = 32;
    while width >= 1 {
        for lane in 0..width {
            lanes[lane] = add(lanes[lane], lanes[lane + width]);
        }
        width >>= 1;
    }
    lanes[0]
}

#[inline]
pub fn is_dense_order(order: &[u32], len: usize) -> bool {
    order.len() == len && order.iter().enumerate().all(|(i, &id)| id as usize == i)
}

pub fn scaled_add_dense(source: &[f32], scale: f32, destination: &mut [f32]) {
    debug_assert_eq!(source.len(), destination.len());
    #[cfg(feature = "parallel")]
    if source.len() >= PARALLEL_POINTWISE_THRESHOLD {
        destination
            .par_chunks_mut(PARALLEL_POINTWISE_CHUNK)
            .zip(source.par_chunks(PARALLEL_POINTWISE_CHUNK))
            .for_each(|(dst, src)| simd::scaled_add(src, scale, dst));
        return;
    }
    simd::scaled_add(source, scale, destination)
}

/// Independent pointwise work follows contiguous dense spans, while global
/// reductions retain their canonical execution order. Dry lanes are untouched.
pub fn scaled_add_active(source: &[f32], scale: f32, destination: &mut [f32], active: &[bool]) {
    let block = |src: &[f32], dst: &mut [f32], mask: &[bool]| {
        let mut begin = 0;
        while begin < mask.len() {
            if !mask[begin] {
                begin += 1;
                continue;
            }
            let mut end = begin + 1;
            while end < mask.len() && mask[end] {
                end += 1;
            }
            simd::scaled_add(&src[begin..end], scale, &mut dst[begin..end]);
            begin = end;
        }
    };
    #[cfg(feature = "parallel")]
    if source.len() >= PARALLEL_POINTWISE_THRESHOLD {
        destination
            .par_chunks_mut(PARALLEL_POINTWISE_CHUNK)
            .zip(source.par_chunks(PARALLEL_POINTWISE_CHUNK))
            .zip(active.par_chunks(PARALLEL_POINTWISE_CHUNK))
            .for_each(|((dst, src), mask)| block(src, dst, mask));
        return;
    }
    block(source, destination, active);
}

pub fn pressure_xrz_active(
    pressure: &mut [f32],
    residual: &mut [f32],
    z: &mut [f32],
    alpha: f32,
    direction: &[f32],
    image_direction: &[f32],
    diagonal: &[f32],
    active: &[bool],
) {
    let block = |x: &mut [f32],
                 r: &mut [f32],
                 zz: &mut [f32],
                 p: &[f32],
                 ap: &[f32],
                 d: &[f32],
                 mask: &[bool]| {
        let mut begin = 0;
        while begin < mask.len() {
            if !mask[begin] {
                begin += 1;
                continue;
            }
            let mut end = begin + 1;
            while end < mask.len() && mask[end] {
                end += 1;
            }
            simd::pressure_xrz(
                &mut x[begin..end],
                &mut r[begin..end],
                &mut zz[begin..end],
                alpha,
                &p[begin..end],
                &ap[begin..end],
                &d[begin..end],
            );
            begin = end;
        }
    };
    #[cfg(feature = "parallel")]
    if pressure.len() >= PARALLEL_POINTWISE_THRESHOLD {
        pressure
            .par_chunks_mut(PARALLEL_POINTWISE_CHUNK)
            .zip(residual.par_chunks_mut(PARALLEL_POINTWISE_CHUNK))
            .zip(z.par_chunks_mut(PARALLEL_POINTWISE_CHUNK))
            .zip(direction.par_chunks(PARALLEL_POINTWISE_CHUNK))
            .zip(image_direction.par_chunks(PARALLEL_POINTWISE_CHUNK))
            .zip(diagonal.par_chunks(PARALLEL_POINTWISE_CHUNK))
            .zip(active.par_chunks(PARALLEL_POINTWISE_CHUNK))
            .for_each(|((((((x, r), zz), p), ap), d), mask)| block(x, r, zz, p, ap, d, mask));
        return;
    }
    block(
        pressure,
        residual,
        z,
        direction,
        image_direction,
        diagonal,
        active,
    );
}

pub fn pressure_xrz_dense(
    pressure: &mut [f32],
    residual: &mut [f32],
    z: &mut [f32],
    alpha: f32,
    direction: &[f32],
    image_direction: &[f32],
    diagonal: &[f32],
) {
    debug_assert_eq!(pressure.len(), residual.len());
    #[cfg(feature = "parallel")]
    if pressure.len() >= PARALLEL_POINTWISE_THRESHOLD {
        pressure
            .par_chunks_mut(PARALLEL_POINTWISE_CHUNK)
            .zip(residual.par_chunks_mut(PARALLEL_POINTWISE_CHUNK))
            .zip(z.par_chunks_mut(PARALLEL_POINTWISE_CHUNK))
            .zip(direction.par_chunks(PARALLEL_POINTWISE_CHUNK))
            .zip(image_direction.par_chunks(PARALLEL_POINTWISE_CHUNK))
            .zip(diagonal.par_chunks(PARALLEL_POINTWISE_CHUNK))
            .for_each(|(((((x, r), zz), p), ap), d)| simd::pressure_xrz(x, r, zz, alpha, p, ap, d));
        return;
    }
    simd::pressure_xrz(
        pressure,
        residual,
        z,
        alpha,
        direction,
        image_direction,
        diagonal,
    )
}

mod simd {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub fn multiply(a: &[f32], b: &[f32], out: &mut [f32]) {
        unsafe {
            use core::arch::wasm32::*;
            let mut i = 0;
            while i + 4 <= a.len() {
                v128_store(
                    out.as_mut_ptr().add(i).cast(),
                    f32x4_mul(
                        v128_load(a.as_ptr().add(i).cast()),
                        v128_load(b.as_ptr().add(i).cast()),
                    ),
                );
                i += 4
            }
            for j in i..a.len() {
                out[j] = a[j] * b[j]
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    pub fn multiply(a: &[f32], b: &[f32], out: &mut [f32]) {
        unsafe {
            use core::arch::aarch64::*;
            let mut i = 0;
            while i + 4 <= a.len() {
                vst1q_f32(
                    out.as_mut_ptr().add(i),
                    vmulq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i))),
                );
                i += 4
            }
            for j in i..a.len() {
                out[j] = a[j] * b[j]
            }
        }
    }
    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64"
    )))]
    pub fn multiply(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] * b[i]
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub fn scaled_add(source: &[f32], scale: f32, destination: &mut [f32]) {
        unsafe {
            use core::arch::wasm32::*;
            let mut i = 0;
            let s = f32x4_splat(scale);
            while i + 4 <= source.len() {
                let a = v128_load(source.as_ptr().add(i).cast());
                let b = v128_load(destination.as_ptr().add(i).cast());
                v128_store(
                    destination.as_mut_ptr().add(i).cast(),
                    f32x4_add(a, f32x4_mul(s, b)),
                );
                i += 4
            }
            for j in i..source.len() {
                destination[j] = source[j] + scale * destination[j]
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    pub fn scaled_add(source: &[f32], scale: f32, destination: &mut [f32]) {
        unsafe {
            use core::arch::aarch64::*;
            let mut i = 0;
            let s = vdupq_n_f32(scale);
            while i + 4 <= source.len() {
                let a = vld1q_f32(source.as_ptr().add(i));
                let b = vld1q_f32(destination.as_ptr().add(i));
                vst1q_f32(
                    destination.as_mut_ptr().add(i),
                    vaddq_f32(a, vmulq_f32(s, b)),
                );
                i += 4
            }
            for j in i..source.len() {
                destination[j] = source[j] + scale * destination[j]
            }
        }
    }
    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64"
    )))]
    pub fn scaled_add(source: &[f32], scale: f32, destination: &mut [f32]) {
        for (i, v) in destination.iter_mut().enumerate() {
            *v = source[i] + scale * *v
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub fn pressure_xrz(
        x: &mut [f32],
        r: &mut [f32],
        z: &mut [f32],
        alpha: f32,
        p: &[f32],
        ap: &[f32],
        diag: &[f32],
    ) {
        unsafe {
            use core::arch::wasm32::*;
            let mut i = 0;
            let av = f32x4_splat(alpha);
            let zero = f32x4_splat(0.0);
            while i + 4 <= x.len() {
                let xv = v128_load(x.as_ptr().add(i).cast());
                let rv = v128_load(r.as_ptr().add(i).cast());
                let pv = v128_load(p.as_ptr().add(i).cast());
                let apv = v128_load(ap.as_ptr().add(i).cast());
                let dv = v128_load(diag.as_ptr().add(i).cast());
                let nr = f32x4_sub(rv, f32x4_mul(av, apv));
                v128_store(
                    x.as_mut_ptr().add(i).cast(),
                    f32x4_add(xv, f32x4_mul(av, pv)),
                );
                v128_store(r.as_mut_ptr().add(i).cast(), nr);
                v128_store(
                    z.as_mut_ptr().add(i).cast(),
                    v128_bitselect(f32x4_div(nr, dv), zero, f32x4_gt(dv, zero)),
                );
                i += 4
            }
            for j in i..x.len() {
                x[j] += alpha * p[j];
                r[j] -= alpha * ap[j];
                z[j] = if diag[j] > 0.0 { r[j] / diag[j] } else { 0.0 }
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    pub fn pressure_xrz(
        x: &mut [f32],
        r: &mut [f32],
        z: &mut [f32],
        alpha: f32,
        p: &[f32],
        ap: &[f32],
        diag: &[f32],
    ) {
        unsafe {
            use core::arch::aarch64::*;
            let mut i = 0;
            let av = vdupq_n_f32(alpha);
            let zero = vdupq_n_f32(0.0);
            while i + 4 <= x.len() {
                let xv = vld1q_f32(x.as_ptr().add(i));
                let rv = vld1q_f32(r.as_ptr().add(i));
                let pv = vld1q_f32(p.as_ptr().add(i));
                let apv = vld1q_f32(ap.as_ptr().add(i));
                let dv = vld1q_f32(diag.as_ptr().add(i));
                let nr = vsubq_f32(rv, vmulq_f32(av, apv));
                vst1q_f32(x.as_mut_ptr().add(i), vaddq_f32(xv, vmulq_f32(av, pv)));
                vst1q_f32(r.as_mut_ptr().add(i), nr);
                vst1q_f32(
                    z.as_mut_ptr().add(i),
                    vbslq_f32(vcgtq_f32(dv, zero), vdivq_f32(nr, dv), zero),
                );
                i += 4
            }
            for j in i..x.len() {
                x[j] += alpha * p[j];
                r[j] -= alpha * ap[j];
                z[j] = if diag[j] > 0.0 { r[j] / diag[j] } else { 0.0 }
            }
        }
    }
    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64"
    )))]
    pub fn pressure_xrz(
        x: &mut [f32],
        r: &mut [f32],
        z: &mut [f32],
        alpha: f32,
        p: &[f32],
        ap: &[f32],
        diag: &[f32],
    ) {
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            z[i] = if diag[i] > 0.0 { r[i] / diag[i] } else { 0.0 }
        }
    }
}

pub fn reduce_dot(a: &[f32], b: &[f32], execution_order: &[u32], scratch: &mut [f32]) -> f32 {
    // Inactive products are never reduced. Computing this contiguous plane
    // also enables SIMD for sparse pressure membership without a gather pass.
    simd::multiply(a, b, scratch);
    reduce_production(scratch, execution_order)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_active_spans_match_scalar_and_preserve_inactive_words() {
        let n = 9003;
        let mask: Vec<_> = (0..n).map(|i| i % 23 < 17).collect();
        let source: Vec<_> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
        let image: Vec<_> = (0..n).map(|i| ((i as f32) * 0.021).cos()).collect();
        let diagonal: Vec<_> = (0..n)
            .map(|i| if i % 17 == 0 { 0.0 } else { 2.5 })
            .collect();
        let mut p: Vec<_> = (0..n)
            .map(|i| {
                if mask[i] {
                    0.5
                } else {
                    f32::from_bits(0x7fc00042)
                }
            })
            .collect();
        let mut r = source.clone();
        let mut z = image.clone();
        let mut ep = p.clone();
        let mut er = r.clone();
        let mut ez = z.clone();
        for i in 0..n {
            if mask[i] {
                ep[i] = add(ep[i], mul(0.375, source[i]));
                er[i] = er[i] - mul(0.375, image[i]);
                ez[i] = if diagonal[i] > 0.0 {
                    div(er[i], diagonal[i])
                } else {
                    0.0
                };
            }
        }
        pressure_xrz_active(
            &mut p, &mut r, &mut z, 0.375, &source, &image, &diagonal, &mask,
        );
        for (got, expected) in [(&p, &ep), (&r, &er), (&z, &ez)] {
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        for i in 0..n {
            if mask[i] {
                ep[i] = add(source[i], mul(-0.25, ep[i]));
            }
        }
        scaled_add_active(&source, -0.25, &mut p, &mask);
        assert_eq!(
            p.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ep.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    #[test]
    fn reduction_keeps_resident_tree() {
        let values = [16_777_216.0, 1.0, -16_777_216.0, 1.0];
        // The resident lane tree pairs the two large terms before adding the
        // unit lanes; a left fold would produce 1 instead.
        assert_eq!(reduce_production(&values, &[0, 1, 2, 3]), 2.0);
    }

    #[test]
    fn dense_pointwise_matches_scalar_bits() {
        let source: Vec<f32> = (0..259)
            .map(|i| f32::from_bits(0x3e80_0000 + (i as u32 % 97) * 7919))
            .collect();
        let mut observed: Vec<f32> = (0..259).map(|i| (i as f32 - 80.0) * 0.03125).collect();
        let mut expected = observed.clone();
        let scale = -0.1875;
        for i in 0..expected.len() {
            expected[i] = add(source[i], mul(scale, expected[i]));
        }
        scaled_add_dense(&source, scale, &mut observed);
        assert_eq!(
            observed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let mut x = source.clone();
        let mut r = observed.clone();
        let mut z = vec![0.0; x.len()];
        let mut ex = x.clone();
        let mut er = r.clone();
        let mut ez = z.clone();
        let diagonal: Vec<f32> = (0..x.len())
            .map(|i| {
                if i % 11 == 0 {
                    0.0
                } else {
                    0.25 + (i % 7) as f32
                }
            })
            .collect();
        for i in 0..x.len() {
            ex[i] = add(ex[i], mul(scale, source[i]));
            er[i] = add(er[i], -mul(scale, observed[i]));
            ez[i] = if diagonal[i] > 0.0 {
                div(er[i], diagonal[i])
            } else {
                0.0
            };
        }
        pressure_xrz_dense(&mut x, &mut r, &mut z, scale, &source, &observed, &diagonal);
        assert_eq!(
            x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ex.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            er.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            z.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ez.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_reduction_is_bitwise_across_pool_sizes() {
        let values: Vec<f32> = (0..4097)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.0078125)
            .collect();
        let order: Vec<u32> = (0..values.len() as u32).collect();
        let expected = reduce_production(&values, &order).to_bits();
        for threads in [1, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(
                pool.install(|| reduce_production(&values, &order))
                    .to_bits(),
                expected
            )
        }
    }

    #[test]
    #[ignore = "manual throughput receipt; timings are machine-specific"]
    fn benchmark_pressure_kernels() {
        let n = 1 << 20;
        let source: Vec<f32> = (0..n).map(|i| (i % 251) as f32 / 251.0).collect();
        let mut destination = source.clone();
        let order: Vec<u32> = (0..n as u32).collect();
        let mut scratch = vec![0.0; n];
        let start = std::time::Instant::now();
        let mut checksum = 0u32;
        for _ in 0..64 {
            scaled_add_dense(&source, 0.125, &mut destination);
            checksum ^= reduce_dot(&source, &destination, &order, &mut scratch).to_bits()
        }
        eprintln!(
            "pressure-kernel n={n} rounds=64 elapsed_ms={} checksum={checksum:08x}",
            start.elapsed().as_millis()
        );
    }
}
