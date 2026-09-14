//! Source-owned conservative generation transfer, followed by a target gather.
//! Spatial indices accelerate discovery; canonical source ordering and f32
//! compensation remain the numerical contract.
use crate::{Fields, Graph, ValidationError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
const NEW_AIR: u32 = u32::MAX;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewAirCoverage {
    pub minimum_fine: [f32; 2],
    pub maximum_exclusive_fine: [f32; 2],
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransferPlan {
    pub cell_offsets: Vec<u32>,
    pub cell_sources: Vec<u32>,
    pub cell_areas: Vec<f32>,
    pub face_offsets: Vec<u32>,
    pub face_sources: Vec<u32>,
    pub face_areas: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransferResult {
    pub density: Vec<f32>,
    pub gamma: Vec<f32>,
    pub pressure: Vec<f32>,
    pub cell_velocity: Vec<f32>,
    pub face_velocity: Vec<f32>,
    pub capacity: Vec<f32>,
    pub interface_normal: Vec<f32>,
    pub interface_offset: Vec<f32>,
    pub plan: TransferPlan,
    pub source_amounts: Vec<f32>,
    pub source_capacities: Vec<f32>,
    pub target_amounts: Vec<f32>,
    pub target_capacities: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TransferError {
    pub fault: u32,
    pub owner: u32,
    pub amount: f32,
    pub capacity: f32,
    pub message: String,
}
impl std::fmt::Display for TransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "generation transfer: {} (fault {}, owner {}, amount {}, capacity {})",
            self.message, self.fault, self.owner, self.amount, self.capacity
        )
    }
}
impl std::error::Error for TransferError {}
impl From<ValidationError> for TransferError {
    fn from(e: ValidationError) -> Self {
        Self {
            fault: 0,
            owner: 0,
            amount: 0.0,
            capacity: 0.0,
            message: e.to_string(),
        }
    }
}
fn deferred(fault: u32, owner: usize, amount: f32, capacity: f32) -> TransferError {
    TransferError {
        fault,
        owner: owner as u32,
        amount,
        capacity,
        message: "candidate capacity cannot represent accepted volume".into(),
    }
}
fn tolerance(capacity: f32) -> f32 {
    9.536_743e-7 * capacity
}
fn add(total: [f32; 2], value: f32) -> [f32; 2] {
    let sum = total[0] + value;
    let error = if total[0].abs() >= value.abs() {
        (total[0] - sum) + value
    } else {
        (value - sum) + total[0]
    };
    let tail = total[1] + error;
    let result = sum + tail;
    [result, (sum - result) + tail]
}
fn valid(amount: f32, capacity: f32) -> bool {
    let tol = tolerance(capacity);
    capacity.is_finite()
        && capacity >= 0.0
        && amount.is_finite()
        && amount >= -tol
        && amount as f64 <= capacity as f64 + tol as f64
}
fn overlap(a: f64, b: f64, c: f64, d: f64) -> f64 {
    (b.min(d) - a.max(c)).max(0.0)
}

pub fn plan_transfer(
    source: &Graph,
    target: &Graph,
    new_air: &[NewAirCoverage],
) -> Result<TransferPlan, ValidationError> {
    source.validate()?;
    target.validate()?;
    if source.dimension != 2
        || target.dimension != 2
        || source.dimensions[..2] != target.dimensions[..2]
    {
        return Err(ValidationError(
            "slice transfer requires the same 2D physical domain".into(),
        ));
    }
    let mut spatial: HashMap<[i32; 2], Vec<u32>> = HashMap::new();
    for c in &source.cells {
        for y in c.minimum[1].floor() as i32..c.maximum[1].ceil() as i32 {
            for x in c.minimum[0].floor() as i32..c.maximum[0].ceil() as i32 {
                spatial.entry([x, y]).or_default().push(c.id);
            }
        }
    }
    let mut plan = TransferPlan::default();
    for c in &target.cells {
        plan.cell_offsets.push(plan.cell_sources.len() as u32);
        let mut candidates = BTreeSet::new();
        for y in c.minimum[1].floor() as i32..c.maximum[1].ceil() as i32 {
            for x in c.minimum[0].floor() as i32..c.maximum[0].ceil() as i32 {
                if let Some(ids) = spatial.get(&[x, y]) {
                    candidates.extend(ids)
                }
            }
        }
        let mut covered = 0.0_f64;
        for &id in &candidates {
            let old = &source.cells[id as usize];
            let area = overlap(
                c.minimum[0] as f64,
                c.maximum[0] as f64,
                old.minimum[0] as f64,
                old.maximum[0] as f64,
            ) * overlap(
                c.minimum[1] as f64,
                c.maximum[1] as f64,
                old.minimum[1] as f64,
                old.maximum[1] as f64,
            );
            if area <= 0.0 {
                continue;
            }
            plan.cell_sources.push(id);
            plan.cell_areas.push(area as f32);
            covered += area;
        }
        if covered > c.measure as f64 + 1e-6 {
            return Err(ValidationError(format!(
                "target cell {} has overlapping accepted coverage",
                c.id
            )));
        }
        if covered < c.measure as f64 - 1e-6 {
            if !new_air.iter().any(|b| {
                (0..2).all(|a| {
                    c.minimum[a] >= b.minimum_fine[a] && c.maximum[a] <= b.maximum_exclusive_fine[a]
                })
            }) {
                return Err(ValidationError(format!(
                    "target cell {} lacks complete accepted coverage",
                    c.id
                )));
            }
            plan.cell_sources.push(NEW_AIR);
            plan.cell_areas.push((c.measure as f64 - covered) as f32);
        }
    }
    plan.cell_offsets.push(plan.cell_sources.len() as u32);
    let mut faces: HashMap<(u8, u32, i32), Vec<u32>> = HashMap::new();
    for row in &source.rows {
        let a = row.axis as usize;
        let t = 1 - a;
        let lo = row.center[t] as f64 - 0.5 * row.measure as f64;
        let hi = row.center[t] as f64 + 0.5 * row.measure as f64;
        for q in lo.floor() as i32..hi.ceil() as i32 {
            faces
                .entry((row.axis, row.center[a].to_bits(), q))
                .or_default()
                .push(row.id);
        }
    }
    for row in &target.rows {
        plan.face_offsets.push(plan.face_sources.len() as u32);
        let a = row.axis as usize;
        let t = 1 - a;
        let lo = row.center[t] as f64 - 0.5 * row.measure as f64;
        let hi = row.center[t] as f64 + 0.5 * row.measure as f64;
        let mut candidates = BTreeSet::new();
        for q in lo.floor() as i32..hi.ceil() as i32 {
            if let Some(ids) = faces.get(&(row.axis, row.center[a].to_bits(), q)) {
                candidates.extend(ids)
            }
        }
        for &id in &candidates {
            let old = &source.rows[id as usize];
            let area = overlap(
                lo,
                hi,
                old.center[t] as f64 - 0.5 * old.measure as f64,
                old.center[t] as f64 + 0.5 * old.measure as f64,
            );
            if area > 0.0 {
                plan.face_sources.push(id);
                plan.face_areas.push(area as f32);
            }
        }
    }
    plan.face_offsets.push(plan.face_sources.len() as u32);
    Ok(plan)
}

fn interface_plane(fill: f32, g: [f32; 2], widths: [f32; 2]) -> ([f32; 2], f32, bool) {
    let maximum = g[0].abs().max(g[1].abs());
    if maximum <= 1e-20 {
        return ([0.0; 2], 0.0, false);
    }
    let gx = g[0] / maximum;
    let gy = g[1] / maximum;
    let length = (gx * gx + gy * gy).sqrt();
    let n = [gx / length, gy / length];
    let projected = [n[0].abs() * widths[0], n[1].abs() * widths[1]];
    let dominant = projected[0].max(projected[1]);
    let radius = 0.5 * (projected[0] + projected[1]);
    let mut offset = 0.0;
    if fill <= 0.0 {
        offset = -radius
    } else if fill >= 1.0 {
        offset = radius
    } else if fill != 0.5 && dominant > 1e-20 {
        let spans: Vec<_> = projected
            .iter()
            .map(|v| v / dominant)
            .filter(|v| *v >= 1e-6)
            .collect();
        if spans.len() == 1 {
            offset = (fill - 0.5) * (spans[0] * dominant)
        } else {
            let complement = fill > 0.5;
            let target = if complement { 1.0 - fill } else { fill };
            let a = spans[0].min(spans[1]);
            let b = spans[0].max(spans[1]);
            let mut shifted = target * b + 0.5 * a;
            if target < (0.5 * a) / b {
                shifted = (((2.0 * target) * a) * b).sqrt()
            }
            let lower = (shifted - 0.5 * (a + b)) * dominant;
            offset = if complement { -lower } else { lower };
        }
    }
    (
        n,
        offset,
        ((n[0] as f64 * n[0] as f64 + n[1] as f64 * n[1] as f64) as f32) > 0.5,
    )
}

pub fn transfer_fields(
    source: &Graph,
    target: &Graph,
    fields: &Fields,
    target_capacity: &[f32],
    new_air: &[NewAirCoverage],
) -> Result<TransferResult, TransferError> {
    transfer_fields_impl(source, target, fields, target_capacity, new_air, false)
}

/// Level-set-volume generation transfer. It preserves finite non-negative
/// excess volume instead of treating cell capacity as a hard storage limit.
pub fn transfer_fields_allow_overcapacity(
    source: &Graph,
    target: &Graph,
    fields: &Fields,
    target_capacity: &[f32],
    new_air: &[NewAirCoverage],
) -> Result<TransferResult, TransferError> {
    transfer_fields_impl(source, target, fields, target_capacity, new_air, true)
}

fn transfer_fields_impl(
    source: &Graph,
    target: &Graph,
    fields: &Fields,
    target_capacity: &[f32],
    new_air: &[NewAirCoverage],
    allow_overcapacity: bool,
) -> Result<TransferResult, TransferError> {
    fields.validate_for(source)?;
    if target_capacity.len() != target.cells.len() {
        return Err(ValidationError("target capacity length differs".into()).into());
    }
    let plan = plan_transfer(source, target, new_air)?;
    let count = plan.cell_sources.len();
    let mut entry_target = vec![0usize; count];
    let mut centers = vec![[0.0; 2]; count];
    let mut widths = vec![[0.0; 2]; count];
    let mut groups = vec![Vec::new(); source.cells.len()];
    for next in &target.cells {
        for entry in plan.cell_offsets[next.id as usize] as usize
            ..plan.cell_offsets[next.id as usize + 1] as usize
        {
            entry_target[entry] = next.id as usize;
            let id = plan.cell_sources[entry];
            if id == NEW_AIR {
                continue;
            }
            let before = &source.cells[id as usize];
            for a in 0..2 {
                let lo = next.minimum[a].max(before.minimum[a]);
                let hi = next.maximum[a].min(before.maximum[a]);
                widths[entry][a] = hi - lo;
                centers[entry][a] = lo + 0.5 * widths[entry][a];
            }
            groups[id as usize].push(entry);
        }
    }
    for group in &mut groups {
        group.sort_by(|&a, &b| {
            let a_cell = &target.cells[entry_target[a]];
            let b_cell = &target.cells[entry_target[b]];
            a_cell.minimum[0]
                .total_cmp(&b_cell.minimum[0])
                .then(a_cell.minimum[1].total_cmp(&b_cell.minimum[1]))
                .then(a.cmp(&b))
        })
    }
    let mut contributions = vec![0.0; count];
    let mut source_amounts = vec![0.0; source.cells.len()];
    let mut source_capacities = source_amounts.clone();
    for before in &source.cells {
        let id = before.id as usize;
        let amount = fields.density[id] * before.measure;
        let capacity = fields.capacity[id] * before.measure;
        source_amounts[id] = amount;
        source_capacities[id] = capacity;
        if if allow_overcapacity {
            !(capacity.is_finite() && capacity >= 0.0 && amount.is_finite() && amount >= -tolerance(capacity)
                && !(amount > 0.0 && capacity <= 0.0))
        } else {
            !valid(amount, capacity)
        } {
            return Err(deferred(1, id, amount, capacity));
        }
        let group = &groups[id];
        let (mut covered, mut capacities) = ([0.0; 2], [0.0; 2]);
        for &entry in group {
            covered = add(covered, plan.cell_areas[entry]);
            let child = target_capacity[entry_target[entry]] * plan.cell_areas[entry];
            if !child.is_finite() || child < 0.0 {
                return Err(deferred(64, id, amount, child));
            }
            capacities = add(capacities, child);
        }
        let coverage = covered[0] + covered[1];
        if (coverage - before.measure).abs() > tolerance(before.measure) && amount != 0.0 {
            return Err(deferred(128, id, amount, coverage));
        }
        if group.is_empty() {
            continue;
        }
        if group.len() == 1 {
            contributions[group[0]] = amount;
            continue;
        }
        let available = capacities[0] + capacities[1];
        if !allow_overcapacity && amount.abs() > available + tolerance(available) {
            return Err(deferred(64, id, amount, available));
        }
        let base_amount = if allow_overcapacity { amount.min(capacity) } else { amount };
        let supplied_normal = [
            fields.interface_normal[2 * id],
            fields.interface_normal[2 * id + 1],
        ];
        let (normal, offset, plane_valid) = if allow_overcapacity {
            let length = supplied_normal[0].hypot(supplied_normal[1]);
            (
                if length > 1e-20 {
                    [supplied_normal[0] / length, supplied_normal[1] / length]
                } else {
                    [0.0; 2]
                },
                fields.interface_offset[id],
                length.is_finite() && length > 1e-20 && fields.interface_offset[id].is_finite(),
            )
        } else {
            interface_plane(
                (base_amount / before.measure).clamp(0.0, 1.0),
                supplied_normal,
                [before.widths[0], before.widths[1]],
            )
        };
        let use_plane = if allow_overcapacity {
            capacity > 0.0 && plane_valid && amount >= 0.0
        } else {
            capacity == before.measure && plane_valid && amount >= 0.0
        };
        let mut remaining = [amount, 0.0];
        for &entry in group {
            let area = plan.cell_areas[entry];
            let child = target_capacity[entry_target[entry]] * area;
            let mut proposed = if available > 0.0 {
                base_amount * (child / available)
            } else {
                0.0
            };
            if use_plane && child == area {
                let rel = [
                    centers[entry][0] - before.center[0],
                    centers[entry][1] - before.center[1],
                ];
                proposed = area
                    * crate::numerics::plic_box_fraction(
                        normal[0],
                        normal[1],
                        offset - (normal[0] * rel[0] + normal[1] * rel[1]),
                        widths[entry][0],
                        widths[entry][1],
                    );
            }
            if amount >= 0.0 {
                let excess = if allow_overcapacity {
                    (amount - base_amount).max(0.0)
                } else {
                    (amount - available).max(0.0)
                };
                if excess > 0.0 && available > 0.0 {
                    if allow_overcapacity {
                        proposed += excess * (child / available)
                    } else {
                        proposed = child + excess * (child / available)
                    }
                }
                proposed = if allow_overcapacity {
                    proposed.max(0.0)
                } else {
                    proposed.clamp(0.0, child + tolerance(child))
                }
            }
            contributions[entry] = proposed;
            remaining = add(remaining, -proposed);
        }
        for _ in 0..2 {
            for &entry in group {
                let child = target_capacity[entry_target[entry]] * plan.cell_areas[entry];
                let previous = contributions[entry];
                let residual = remaining[0] + remaining[1];
                let lower = if amount < 0.0 { -tolerance(child) } else { 0.0 };
                let upper = if amount < 0.0 || child <= 0.0 {
                    0.0
                } else if allow_overcapacity {
                    f32::INFINITY
                } else {
                    child + tolerance(child)
                };
                let proposed = (previous + residual).clamp(lower, upper);
                contributions[entry] = proposed;
                remaining = add(remaining, previous);
                remaining = add(remaining, -proposed);
            }
        }
        let remainder = remaining[0] + remaining[1];
        if remainder.abs() > tolerance(amount.abs()) {
            return Err(deferred(256, id, remainder, amount.abs()));
        }
    }
    let n = target.cells.len();
    let mut result = TransferResult {
        density: vec![0.0; n],
        gamma: vec![0.0; n],
        pressure: vec![0.0; n],
        cell_velocity: vec![0.0; 2 * n],
        face_velocity: vec![0.0; target.rows.len()],
        capacity: target_capacity.to_vec(),
        interface_normal: vec![0.0; 2 * n],
        interface_offset: vec![0.0; n],
        source_amounts,
        source_capacities,
        target_amounts: vec![0.0; n],
        target_capacities: vec![0.0; n],
        plan,
    };
    let plan = &result.plan;
    for next in &target.cells {
        let id = next.id as usize;
        let (mut mass, mut observed) = ([0.0; 2], [0.0; 2]);
        let (mut g, mut p, mut mx, mut my, mut dx, mut dy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for entry in plan.cell_offsets[id] as usize..plan.cell_offsets[id + 1] as usize {
            let old = plan.cell_sources[entry];
            let area = plan.cell_areas[entry];
            if old == NEW_AIR {
                g += area;
                continue;
            }
            let old = old as usize;
            let contribution = contributions[entry];
            mass = add(mass, contribution);
            g += fields.gamma[old] * area;
            p += fields.pressure[old] * area;
            let weight = contribution.max(0.0);
            observed = add(observed, weight);
            mx += weight * fields.cell_velocity[2 * old];
            my += weight * fields.cell_velocity[2 * old + 1];
            dx += area * fields.cell_velocity[2 * old];
            dy += area * fields.cell_velocity[2 * old + 1];
        }
        let area = next.measure;
        let amount = mass[0] + mass[1];
        result.density[id] = amount / area;
        result.gamma[id] = g / area;
        result.pressure[id] = p / area;
        let cap = target_capacity[id] * area;
        result.target_amounts[id] = amount;
        result.target_capacities[id] = cap;
        if if allow_overcapacity {
            !(cap.is_finite() && cap >= 0.0 && amount.is_finite() && amount >= -tolerance(cap)
                && !(amount > 0.0 && cap <= 0.0))
        } else {
            !valid(amount, cap)
        } {
            return Err(deferred(64, id, amount, cap));
        }
        let weight = observed[0] + observed[1];
        result.cell_velocity[2 * id] = if weight > 0.0 { mx / weight } else { dx / area };
        result.cell_velocity[2 * id + 1] = if weight > 0.0 { my / weight } else { dy / area };
    }
    for row in &target.rows {
        let id = row.id as usize;
        let (mut flux, mut covered) = (0.0, 0.0);
        for entry in plan.face_offsets[id] as usize..plan.face_offsets[id + 1] as usize {
            let area = plan.face_areas[entry];
            flux += area * fields.face_velocity[plan.face_sources[entry] as usize];
            covered += area;
        }
        let (mut velocity, mut weight) = (0.0_f32, 0.0_f32);
        for term in &row.terms {
            let w = term.coefficient.abs();
            velocity += w * result.cell_velocity[2 * term.cell_id as usize + row.axis as usize];
            weight += w;
        }
        let fill = (row.measure - covered).max(0.0) * (velocity / weight.max(1e-20));
        result.face_velocity[id] = (flux + fill) / row.measure;
    }
    Ok(result)
}
