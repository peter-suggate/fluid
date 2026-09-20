//! Vertex phi transport and conservative volume-only sharpening.
use super::{extension::Extension, grid::Grid, options::UniformGeometricOptions};
pub fn advect(g: &mut Grid, velocity: &Extension, o: &UniformGeometricOptions, dt: f32) {
    let h = g.h[0].min(g.h[1]);
    let mut next = g.phi.clone();
    let agreement: Vec<f32> = (0..g.volume.len())
        .map(|i| {
            let p = g.point(i);
            let target = g.target(p);
            let v = g.volume[i];
            if g.capacity[i] >= 0.99999
                && (target > 0.0 || v > 0.0)
                && g.phi_at([p[0] as f32 + 0.5, p[1] as f32 + 0.5]).abs() < 1.5 * h
            {
                (v - target).clamp(-1.5, 1.5)
                    + if target > 0.0 && target < 1.0 {
                        4.0
                    } else {
                        0.0
                    }
            } else {
                0.0
            }
        })
        .collect();
    for y in 0..=g.dims[1] {
        for x in 0..=g.dims[0] {
            let p = [x as f32, y as f32];
            let advected = g.phi_at(velocity.trace(g, p, dt));
            let mut value = advected;
            let mut interior = p;
            let mut contact = false;
            for a in 0..2 {
                for upper in [false, true] {
                    let inward = if upper { -1.0 } else { 1.0 };
                    let plane = if upper { g.dims[a] as f32 } else { 0.0 };
                    if (p[a] - plane).abs() > 1e-5 || (a == 1 && upper && g.open_top) {
                        continue;
                    }
                    let mut probe = p;
                    probe[a] += inward;
                    if advected >= 0.0 && inward * velocity.sample(probe)[a] >= -1e-6 {
                        continue;
                    }
                    interior[a] += inward;
                    contact = true;
                }
            }
            if contact && g.open(g.clamp_cell(interior.map(|v| v.floor() as i32))) > 1e-5 {
                let continued = g.phi_at(velocity.trace(g, interior, dt));
                if continued < 0.0 {
                    value = value.min(continued);
                }
            }
            let mut air = -1e20_f32;
            for k in 0..4 {
                let fluid = [x as i32 - 1 + (k & 1), y as i32 - 1 + ((k >> 1) & 1)];
                if g.open(fluid) <= 1e-5 {
                    continue;
                }
                for a in 0..2 {
                    let side = if fluid[a] < p[a] as i32 { 1 } else { -1 };
                    let mut solid = fluid;
                    solid[a] += side;
                    if g.index(solid).is_none() || g.open(solid) > 1e-5 {
                        continue;
                    }
                    let mut inside = p;
                    inside[a] -= side as f32;
                    if advected < 0.0 || side as f32 * velocity.sample(inside)[a] > 1e-6 {
                        value = value.min(g.phi_at(velocity.trace(g, inside, dt)));
                    }
                    let face = if side > 0 { fluid } else { solid };
                    let travel = -side as f32 * g.relative_face(face, a) * dt;
                    if g.is_released(face, a) && travel > 1e-4 * g.h[a] {
                        air = air.max(travel);
                    }
                }
            }
            value = value.max(air);
            // First embedded solid hit supplies incoming air along a long trace.
            let v = velocity.sample(p);
            let mid = g.clamp([
                p[0] - 0.5 * dt * v[0] / g.h[0],
                p[1] - 0.5 * dt * v[1] / g.h[1],
            ]);
            let v = velocity.sample(mid);
            let end = g.clamp([p[0] - dt * v[0] / g.h[0], p[1] - dt * v[1] / g.h[1]]);
            let steps = (2.0 * (end[0] - p[0]).abs().max((end[1] - p[1]).abs()))
                .ceil()
                .max(1.0) as usize;
            let mut previous = p;
            for s in 1..=steps {
                let t = s as f32 / steps as f32;
                let q = [p[0] + t * (end[0] - p[0]), p[1] + t * (end[1] - p[1])];
                let solid = q.map(|v| v.floor() as i32);
                if g.index(solid).is_some() && g.open(solid) <= 1e-5 {
                    for a in 0..2 {
                        for side in [-1, 1] {
                            let mut fluid = solid;
                            fluid[a] += side;
                            if g.open(fluid) <= 1e-5 {
                                continue;
                            }
                            let inward = side as f32;
                            let plane = solid[a] as f32 + if side > 0 { 1.0 } else { 0.0 };
                            let distance = inward * (p[a] - plane);
                            if distance < -1e-5
                                || inward * (end[a] - p[a]) >= 0.0
                                || inward * (previous[a] - plane) < -1e-5
                                || inward * (q[a] - plane) > 1e-5
                            {
                                continue;
                            }
                            let face = if side > 0 { solid } else { fluid };
                            let travel = dt * inward * g.relative_face(face, a);
                            if g.is_released(face, a) && travel > 1e-4 * g.h[a] {
                                value = value.max(travel - distance * g.h[a]);
                            }
                        }
                    }
                    break;
                }
                previous = q;
            }
            for a in 0..2 {
                for upper in [false, true] {
                    let inward = if upper { -1.0 } else { 1.0 };
                    let plane = if upper { g.dims[a] as f32 } else { 0.0 };
                    let ambient = a == 1 && upper && g.open_top;
                    for side in [-1.0, 1.0] {
                        let mut probe = p;
                        probe[1 - a] += side * 1e-4;
                        probe[a] = plane + inward * 1e-4;
                        let mut face = g.clamp_cell(probe.map(|v| v.floor() as i32));
                        if !upper {
                            face[a] -= 1;
                        }
                        if !ambient && !g.is_released(face, a) {
                            continue;
                        }
                        let travel = dt * inward * g.relative_face(face, a);
                        if travel > 1e-4 * g.h[a] {
                            value = value.max(travel - inward * (p[a] - plane) * g.h[a]);
                        }
                    }
                }
            }
            value = g.source_phi(p, value);
            if o.phi_agreement == "on" && value.abs() < 2.0 * h {
                let mut r = 0.0;
                let mut area = 0.0;
                for dy in -4..4 {
                    for dx in -4..4 {
                        if let Some(i) = g.index([x as i32 + dx, y as i32 + dy]) {
                            let packed = agreement[i];
                            if packed == 0.0 {
                                continue;
                            }
                            let cut = packed > 2.0;
                            let w = (1.0 - (dx as f32 + 0.5).abs() / 4.5)
                                * (1.0 - (dy as f32 + 0.5).abs() / 4.5);
                            r += w * (packed - if cut { 4.0 } else { 0.0 });
                            if cut {
                                area += w;
                            }
                        }
                    }
                }
                if area >= 1.0 && (r / area).abs() >= 0.02 {
                    value -= h
                        * (o.phi_agreement_gain * r / area)
                            .clamp(-o.phi_agreement_clamp, o.phi_agreement_clamp);
                }
            }
            if o.phi_seed_from_volume == "on" {
                let mut sum = 0.0;
                let mut count = 0.0;
                for k in 0..4 {
                    if let Some(i) = g
                        .index([x as i32 - 1 + (k & 1), y as i32 - 1 + ((k >> 1) & 1)])
                        .filter(|&i| g.capacity[i] >= 0.99999)
                    {
                        sum += g.volume[i];
                        count += 1.0;
                    }
                }
                let nearby = (-2..2).any(|dy| {
                    (-2..2).any(|dx| {
                        let q = [x as i32 + dx, y as i32 + dy];
                        g.index(q).is_some()
                            && g.phi_at([q[0] as f32 + 0.5, q[1] as f32 + 0.5]) < 0.0
                    })
                });
                if count >= 1.0 && sum / count > 0.25 && !nearby {
                    value = value.min(h * (0.5 - sum / count));
                }
            }
            next[x + (g.dims[0] + 1) * y] = value;
        }
    }
    g.phi = next;
}

/// CM11b closest-point pass, kept separate so scene probes can compare its input.
pub fn redistance(g: &mut Grid) {
    let h = g.h[0].min(g.h[1]);
    let next = &g.phi;
    let mut redistanced = next.clone();
    let band = 4.0 * g.h[0].max(g.h[1]);
    for y in 0..=g.dims[1] {
        for x in 0..=g.dims[0] {
            let i = x + (g.dims[0] + 1) * y;
            let initial = next[i];
            if initial.abs() <= 1e-8 || initial.abs() >= band {
                continue;
            }
            let p = [x as f32, y as f32];
            let mut q = p;
            for _ in 0..8 {
                let grad = g.gradient(&next, q);
                let norm = (grad[0] / g.h[0]).powi(2) + (grad[1] / g.h[1]).powi(2);
                if norm < 1e-16 {
                    break;
                }
                let phi = g.scalar(&next, q);
                for a in 0..2 {
                    q[a] = (q[a] - (phi * grad[a] / (g.h[a] * g.h[a] * norm)).clamp(-2.0, 2.0))
                        .clamp((p[a] - 4.0).max(0.0), (p[a] + 4.0).min(g.dims[a] as f32));
                }
            }
            if g.scalar(&next, q).abs() < 0.005 * h {
                redistanced[i] = initial.signum()
                    * (((p[0] - q[0]) * g.h[0]).powi(2) + ((p[1] - q[1]) * g.h[1]).powi(2)).sqrt();
            }
        }
    }
    g.phi = redistanced;
}
pub fn sharpen(g: &mut Grid, o: &UniformGeometricOptions) -> f64 {
    sharpen_rounds(g, o, 8)
}

/// Diagnostic iteration control; the production schedule remains eight rounds.
pub fn sharpen_rounds(g: &mut Grid, o: &UniformGeometricOptions, rounds: usize) -> f64 {
    if o.density_sharpening == "off" {
        return 0.0;
    }
    let n = g.volume.len();
    let h = g.h[0].min(g.h[1]);
    let phi: Vec<_> = (0..n)
        .map(|i| {
            let p = g.point(i);
            g.phi_at([p[0] as f32 + 0.5, p[1] as f32 + 0.5])
        })
        .collect();
    let target: Vec<_> = (0..n).map(|i| g.target(g.point(i))).collect();
    let compact = o.volume_compaction == "on";
    let dose = o.sharpening_strength;
    let mut dust = 0.0;
    let mut surplus = vec![0.0; n];
    let mut need = vec![0.0; n];
    let mut flux = vec![[0.0; 2]; n];
    let mut limits = vec![[0.0; 2]; n];
    for _ in 0..rounds {
        for i in 0..n {
            let admitted = g.capacity[i] > 0.99999
                && (if compact {
                    phi[i] < o.sharpening_distance * h
                } else {
                    phi[i].abs() < o.sharpening_distance * h
                });
            let relay = phi[i] > 0.0 && target[i] <= 1e-6;
            surplus[i] = if admitted {
                dose * if compact && phi[i] < 0.0 {
                    g.volume[i]
                } else {
                    (g.volume[i] - target[i]).max(0.0)
                }
            } else {
                0.0
            };
            need[i] = if admitted {
                dose * ((if relay { 1.0 } else { target[i] }) - g.volume[i]).max(0.0)
            } else {
                0.0
            };
        }
        for i in 0..n {
            let p = g.point(i);
            for a in 0..2 {
                flux[i][a] = 0.0;
                let mut q = p;
                q[a] += 1;
                let Some(j) = g.index(q) else {
                    continue;
                };
                if g.capacity[i] < 0.99999 || g.capacity[j] < 0.99999 {
                    continue;
                }
                let mut mid = [p[0] as f32 + 0.5, p[1] as f32 + 0.5];
                mid[a] += 0.5;
                let middle = g.phi_at(mid);
                let pa = phi[i];
                let pb = phi[j];
                let inward_a =
                    pa >= 0.0 && pb < pa - 1e-6 && middle <= pa + 1e-6 && middle >= pb - 1e-6;
                let inward_b =
                    pb >= 0.0 && pa < pb - 1e-6 && middle <= pb + 1e-6 && middle >= pa - 1e-6;
                let mut ca = surplus[i];
                let mut cb = surplus[j];
                if compact {
                    if !(pa < 0.0 && pb < pa - 1e-6) {
                        ca = ca.min(dose * (g.volume[i] - target[i]).max(0.0));
                    }
                    if !(pb < 0.0 && pa < pb - 1e-6) {
                        cb = cb.min(dose * (g.volume[j] - target[j]).max(0.0));
                    }
                }
                let ab = if middle <= 1e-6 && !(pb > 0.0 && target[j] <= 1e-6) || inward_a {
                    ca.min(need[j])
                } else {
                    0.0
                };
                let ba = if middle <= 1e-6 && !(pa > 0.0 && target[i] <= 1e-6) || inward_b {
                    cb.min(need[i])
                } else {
                    0.0
                };
                flux[i][a] = ab - ba;
            }
        }
        for i in 0..n {
            let p = g.point(i);
            let mut outgoing = 0.0;
            let mut incoming = 0.0;
            for a in 0..2 {
                let mut q = p;
                q[a] -= 1;
                let positive = flux[i][a];
                let negative = g.index(q).map_or(0.0, |j| flux[j][a]);
                outgoing += positive.max(0.0) + (-negative).max(0.0);
                incoming += (-positive).max(0.0) + negative.max(0.0);
            }
            limits[i] = [
                (surplus[i] / outgoing.max(1e-20)).min(1.0),
                (need[i] / incoming.max(1e-20)).min(1.0),
            ];
        }
        let limited = |i: usize, j: usize, a: usize| {
            let raw = flux[i][a];
            raw * if raw >= 0.0 {
                limits[i][0].min(limits[j][1])
            } else {
                limits[i][1].min(limits[j][0])
            }
        };
        for i in 0..n {
            let p = g.point(i);
            let mut terms = [0.0; 4];
            for a in 0..2 {
                let mut q = p;
                q[a] += 1;
                if let Some(j) = g.index(q) {
                    terms[2 * a] = -limited(i, j, a);
                }
                q[a] -= 2;
                if let Some(j) = g.index(q) {
                    terms[2 * a + 1] = limited(j, i, a);
                }
            }
            let value = g.volume[i] + ((terms[0] + terms[1]) + (terms[2] + terms[3]));
            g.volume[i] = if value.abs() < o.volume_dust_threshold {
                dust += value as f64;
                0.0
            } else {
                value
            };
        }
    }
    dust
}
