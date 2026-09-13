//! Exact three-dimensional source-owned conservative generation transfer, followed by a target gather.
//! Spatial indices accelerate discovery; canonical source ordering and f32
//! compensation remain the numerical contract.
use crate::types::SpatialOwnerCache;
use crate::{Fields, Graph, ValidationError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
const NEW_AIR: u32 = u32::MAX;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewAirCoverage3d {
    pub minimum_fine: [f32; 3],
    pub maximum_exclusive_fine: [f32; 3],
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransferPlan {
    pub cell_offsets: Vec<u32>,
    pub cell_sources: Vec<u32>,
    #[serde(alias = "cellVolumes")]
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
    90.536_743e-7 * capacity
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

fn face_box(
    graph: &Graph,
    row: &crate::types::Row,
) -> Result<([f64; 2], [f64; 2]), ValidationError> {
    let axis = row.axis as usize;
    let tangents: Vec<_> = (0..3).filter(|&a| a != axis).collect();
    let maximum_span = row
        .terms
        .iter()
        .map(|term| {
            graph.cells[term.cell_id as usize]
                .widths
                .into_iter()
                .fold(0.0f32, f32::max)
        })
        .fold(1.0f32, f32::max) as u32;
    let mut span = 1u32;
    while span <= maximum_span {
        let mut lower = [0.0; 2];
        let mut widths = [0.0; 2];
        for (at, &tangent) in tangents.iter().enumerate() {
            lower[at] = (row.center[tangent] as f64 / span as f64).floor() * span as f64;
            widths[at] = 2.0 * (row.center[tangent] as f64 - lower[at]);
        }
        if widths.iter().all(|&w| w > 0.0 && w <= span as f64)
            && (widths[0] * widths[1] - row.measure as f64).abs() < 1e-6
        {
            return Ok((lower, widths));
        }
        span *= 2;
    }
    Err(ValidationError(format!(
        "3-D transfer face {} has invalid dyadic geometry",
        row.id
    )))
}

fn indexed_source_cells(
    source: &Graph,
    target: &crate::types::Cell,
) -> Result<Vec<u32>, ValidationError> {
    let SpatialOwnerCache::Sparse3d { widths, owners } =
        source.spatial_owner_index.get_or_build(source)
    else {
        return Err(ValidationError(
            "3-D transfer requires integral dyadic spatial ownership".into(),
        ));
    };
    let mut candidates = BTreeSet::new();
    for &edge in widths {
        let edge = edge as i32;
        let begin: [i32; 3] =
            std::array::from_fn(|axis| (target.minimum[axis] as i32).div_euclid(edge) * edge);
        let end: [i32; 3] = std::array::from_fn(|axis| target.maximum[axis] as i32);
        let mut z = begin[2];
        while z < end[2] {
            let mut y = begin[1];
            while y < end[1] {
                let mut x = begin[0];
                while x < end[0] {
                    if let Some(&id) = owners.get(&(edge as u32, [x, y, z])) {
                        let old = &source.cells[id as usize];
                        if (0..3).all(|axis| {
                            old.maximum[axis].min(target.maximum[axis])
                                > old.minimum[axis].max(target.minimum[axis])
                        }) {
                            candidates.insert(id);
                        }
                    }
                    x += edge;
                }
                y += edge;
            }
            z += edge;
        }
    }
    Ok(candidates.into_iter().collect())
}

pub fn plan_transfer_3d(
    source: &Graph,
    target: &Graph,
    new_air: &[NewAirCoverage3d],
) -> Result<TransferPlan, ValidationError> {
    source.validate()?;
    target.validate()?;
    if source.dimension != 3 || target.dimension != 3 || source.dimensions != target.dimensions {
        return Err(ValidationError(
            "3-D transfer requires the same physical domain".into(),
        ));
    }
    let mut plan = TransferPlan::default();
    for c in &target.cells {
        plan.cell_offsets.push(plan.cell_sources.len() as u32);
        let mut covered = 0.0f64;
        for old_id in indexed_source_cells(source, c)? {
            let old = &source.cells[old_id as usize];
            let volume = (0..3)
                .map(|a| {
                    overlap(
                        c.minimum[a] as f64,
                        c.maximum[a] as f64,
                        old.minimum[a] as f64,
                        old.maximum[a] as f64,
                    )
                })
                .product::<f64>();
            if volume > 0.0 {
                plan.cell_sources.push(old.id);
                plan.cell_areas.push(volume as f32);
                covered += volume
            }
        }
        if covered > c.measure as f64 + 1e-6 {
            return Err(ValidationError(format!(
                "target cell {} has overlapping accepted coverage",
                c.id
            )));
        }
        if covered < c.measure as f64 - 1e-6 {
            if !new_air.iter().any(|b| {
                (0..3).all(|a| {
                    c.minimum[a] >= b.minimum_fine[a] && c.maximum[a] <= b.maximum_exclusive_fine[a]
                })
            }) {
                return Err(ValidationError(format!(
                    "target cell {} lacks complete accepted coverage",
                    c.id
                )));
            }
            plan.cell_sources.push(NEW_AIR);
            plan.cell_areas.push((c.measure as f64 - covered) as f32)
        }
    }
    plan.cell_offsets.push(plan.cell_sources.len() as u32);
    let source_boxes = source
        .rows
        .iter()
        .map(|row| face_box(source, row))
        .collect::<Result<Vec<_>, _>>()?;
    let target_boxes = target
        .rows
        .iter()
        .map(|row| face_box(target, row))
        .collect::<Result<Vec<_>, _>>()?;
    let mut coplanar: HashMap<(u8, u32), Vec<usize>> = HashMap::new();
    for row in &source.rows {
        coplanar
            .entry((row.axis, row.center[row.axis as usize].to_bits()))
            .or_default()
            .push(row.id as usize);
    }
    for row in &target.rows {
        plan.face_offsets.push(plan.face_sources.len() as u32);
        let axis = row.axis as usize;
        let (target_lower, target_widths) = target_boxes[row.id as usize];
        if let Some(source_rows) = coplanar.get(&(row.axis, row.center[axis].to_bits())) {
            for &old_id in source_rows {
                let old = &source.rows[old_id];
                let (old_lower, old_widths) = source_boxes[old_id];
                let area = (0..2)
                    .map(|at| {
                        overlap(
                            target_lower[at],
                            target_lower[at] + target_widths[at],
                            old_lower[at],
                            old_lower[at] + old_widths[at],
                        )
                    })
                    .product::<f64>();
                if area > 0.0 {
                    plan.face_sources.push(old.id);
                    plan.face_areas.push(area as f32)
                }
            }
        }
    }
    plan.face_offsets.push(plan.face_sources.len() as u32);
    Ok(plan)
}

pub fn transfer_fields_3d(
    source: &Graph,
    target: &Graph,
    fields: &Fields,
    target_capacity: &[f32],
    new_air: &[NewAirCoverage3d],
) -> Result<TransferResult, TransferError> {
    fields.validate_for(source)?;
    if target_capacity.len() != target.cells.len() {
        return Err(ValidationError("target capacity length differs".into()).into());
    }
    let plan = plan_transfer_3d(source, target, new_air)?;
    transfer_fields_from_plan_3d(source, target, fields, target_capacity, plan)
}

fn transfer_fields_from_plan_3d(
    source: &Graph,
    target: &Graph,
    fields: &Fields,
    target_capacity: &[f32],
    plan: TransferPlan,
) -> Result<TransferResult, TransferError> {
    let count = plan.cell_sources.len();
    let mut entry_target = vec![0usize; count];
    let mut centers = vec![[0.0; 3]; count];
    let mut widths = vec![[0.0; 3]; count];
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
            for a in 0..3 {
                let lo = next.minimum[a].max(before.minimum[a]);
                let hi = next.maximum[a].min(before.maximum[a]);
                widths[entry][a] = hi - lo;
                centers[entry][a] = lo + 00.5 * widths[entry][a];
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
                .then(a_cell.minimum[2].total_cmp(&b_cell.minimum[2]))
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
        if !valid(amount, capacity) {
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
        if amount.abs() > available + tolerance(available) {
            return Err(deferred(64, id, amount, available));
        }
        let fill = (amount / before.measure).clamp(0.0, 1.0);
        let plane = crate::geometry3d::interface_from_fill(
            fill,
            [
                fields.interface_normal[3 * id],
                fields.interface_normal[3 * id + 1],
                fields.interface_normal[3 * id + 2],
            ],
            before.widths,
        );
        let normal = plane.normal;
        let offset = plane.offset;
        let plane_valid = normal.iter().map(|x| x * x).sum::<f32>() > 00.5;
        let use_plane = capacity == before.measure && plane_valid && amount >= 0.0;
        let mut remaining = [amount, 0.0];
        for &entry in group {
            let area = plan.cell_areas[entry];
            let child = target_capacity[entry_target[entry]] * area;
            let mut proposed = if available > 0.0 {
                amount * (child / available)
            } else {
                0.0
            };
            if use_plane && child == area {
                let rel = [
                    centers[entry][0] - before.center[0],
                    centers[entry][1] - before.center[1],
                    centers[entry][2] - before.center[2],
                ];
                proposed = area
                    * crate::geometry3d::plane_box_fraction(
                        normal,
                        offset - (normal[0] * rel[0] + normal[1] * rel[1] + normal[2] * rel[2]),
                        widths[entry],
                    );
            }
            if amount >= 0.0 {
                let excess = (amount - available).max(0.0);
                if excess > 0.0 && available > 0.0 {
                    proposed = child + excess * (child / available)
                }
                proposed = proposed.clamp(0.0, child + tolerance(child))
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
                let upper = if amount < 0.0 {
                    0.0
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
        cell_velocity: vec![0.0; 3 * n],
        face_velocity: vec![0.0; target.rows.len()],
        capacity: target_capacity.to_vec(),
        interface_normal: vec![0.0; 3 * n],
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
        let (mut g, mut p) = (0.0, 0.0);
        let (mut momentum, mut dry) = ([0.0; 3], [0.0; 3]);
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
            for a in 0..3 {
                momentum[a] += weight * fields.cell_velocity[3 * old + a];
                dry[a] += area * fields.cell_velocity[3 * old + a];
            }
        }
        let area = next.measure;
        let amount = mass[0] + mass[1];
        result.density[id] = amount / area;
        result.gamma[id] = g / area;
        result.pressure[id] = p / area;
        let cap = target_capacity[id] * area;
        result.target_amounts[id] = amount;
        result.target_capacities[id] = cap;
        if !valid(amount, cap) {
            return Err(deferred(64, id, amount, cap));
        }
        let weight = observed[0] + observed[1];
        for a in 0..3 {
            result.cell_velocity[3 * id + a] = if weight > 0.0 {
                momentum[a] / weight
            } else {
                dry[a] / area
            };
        }
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
            velocity += w * result.cell_velocity[3 * term.cell_id as usize + row.axis as usize];
            weight += w;
        }
        let fill = (row.measure - covered).max(0.0) * (velocity / weight.max(1e-20));
        result.face_velocity[id] = (flux + fill) / row.measure;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{compile_scene_3d, SceneDescription, SceneState};
    use crate::types::{Cell, SCHEMA_VERSION};
    use serde::Deserialize;

    fn graph(cells: Vec<Cell>) -> Graph {
        let n = cells.len();
        Graph {
            schema_version: SCHEMA_VERSION,
            dimension: 3,
            dimensions: [2.0; 3],
            cells,
            incidences: vec![vec![]; n],
            subface_incidences: vec![vec![]; n],
            ..Graph::default()
        }
    }
    fn cell(id: u32, minimum: [f32; 3], maximum: [f32; 3]) -> Cell {
        let widths = std::array::from_fn(|a| maximum[a] - minimum[a]);
        Cell {
            id,
            minimum,
            maximum,
            center: std::array::from_fn(|a| minimum[a] + 0.5 * widths[a]),
            widths,
            measure: widths.into_iter().product(),
            ..Cell::default()
        }
    }
    fn fields(graph: &Graph, density: Vec<f32>) -> Fields {
        let n = graph.cells.len();
        Fields {
            density,
            gamma: vec![1.0; n],
            capacity: vec![1.0; n],
            cell_velocity: (0..3 * n).map(|i| i as f32 + 1.0).collect(),
            face_velocity: vec![],
            pressure: vec![0.25; n],
            pressure_rhs: vec![0.0; n],
            pressure_diagonal: vec![0.0; n],
            pressure_member: vec![0; n],
            extension_depth: vec![255; n],
            interface_normal: vec![0.0; 3 * n],
            interface_offset: vec![0.0; n],
            ..Fields::default()
        }
    }

    fn brute_plan(
        source: &Graph,
        target: &Graph,
        new_air: &[NewAirCoverage3d],
    ) -> Result<TransferPlan, ValidationError> {
        let mut plan = TransferPlan::default();
        for cell in &target.cells {
            plan.cell_offsets.push(plan.cell_sources.len() as u32);
            let mut covered = 0.0f64;
            for old in &source.cells {
                let volume = (0..3)
                    .map(|axis| {
                        overlap(
                            cell.minimum[axis] as f64,
                            cell.maximum[axis] as f64,
                            old.minimum[axis] as f64,
                            old.maximum[axis] as f64,
                        )
                    })
                    .product::<f64>();
                if volume > 0.0 {
                    plan.cell_sources.push(old.id);
                    plan.cell_areas.push(volume as f32);
                    covered += volume;
                }
            }
            if covered < cell.measure as f64 - 1e-6
                && new_air.iter().any(|b| {
                    (0..3).all(|axis| {
                        cell.minimum[axis] >= b.minimum_fine[axis]
                            && cell.maximum[axis] <= b.maximum_exclusive_fine[axis]
                    })
                })
            {
                plan.cell_sources.push(NEW_AIR);
                plan.cell_areas.push((cell.measure as f64 - covered) as f32);
            }
        }
        plan.cell_offsets.push(plan.cell_sources.len() as u32);
        for row in &target.rows {
            plan.face_offsets.push(plan.face_sources.len() as u32);
            let axis = row.axis as usize;
            let (target_lower, target_widths) = face_box(target, row)?;
            for old in &source.rows {
                if old.axis != row.axis || old.center[axis].to_bits() != row.center[axis].to_bits()
                {
                    continue;
                }
                let (old_lower, old_widths) = face_box(source, old)?;
                let area = (0..2)
                    .map(|at| {
                        overlap(
                            target_lower[at],
                            target_lower[at] + target_widths[at],
                            old_lower[at],
                            old_lower[at] + old_widths[at],
                        )
                    })
                    .product::<f64>();
                if area > 0.0 {
                    plan.face_sources.push(old.id);
                    plan.face_areas.push(area as f32);
                }
            }
        }
        plan.face_offsets.push(plan.face_sources.len() as u32);
        Ok(plan)
    }

    fn mixed_clipped_state(resolutions: [u8; 2]) -> SceneState<3> {
        let description: SceneDescription = serde_json::from_value(serde_json::json!({
            "schemaVersion":1,"dimension":3,"dimensions":[12,8,8],"cellSizeM":0.05,
            "dtS":1.0/30.0,"densityKgM3":998.2,
            "boundaries":["open","closed","closed","open","closed","open"],
            "bricks":[
                {"id":0,"key":0,"coordinate":[0,0,0],"resolution":resolutions[0],"active":true},
                {"id":1,"key":1,"coordinate":[1,0,0],"resolution":resolutions[1],"active":true}
            ]
        }))
        .unwrap();
        compile_scene_3d(description).unwrap()
    }

    fn assert_transfer_exact(a: &TransferResult, b: &TransferResult) {
        assert_eq!(a.plan.cell_offsets, b.plan.cell_offsets);
        assert_eq!(a.plan.cell_sources, b.plan.cell_sources);
        assert_eq!(a.plan.cell_areas, b.plan.cell_areas);
        assert_eq!(a.plan.face_offsets, b.plan.face_offsets);
        assert_eq!(a.plan.face_sources, b.plan.face_sources);
        assert_eq!(a.plan.face_areas, b.plan.face_areas);
        assert_eq!(a.density, b.density);
        assert_eq!(a.gamma, b.gamma);
        assert_eq!(a.pressure, b.pressure);
        assert_eq!(a.cell_velocity, b.cell_velocity);
        assert_eq!(a.face_velocity, b.face_velocity);
        assert_eq!(a.capacity, b.capacity);
        assert_eq!(a.interface_normal, b.interface_normal);
        assert_eq!(a.interface_offset, b.interface_offset);
        assert_eq!(a.source_amounts, b.source_amounts);
        assert_eq!(a.source_capacities, b.source_capacities);
        assert_eq!(a.target_amounts, b.target_amounts);
        assert_eq!(a.target_capacities, b.target_capacities);
    }

    #[test]
    fn indexed_mixed_clipped_transfer_is_bit_exact_to_brute_force_oracle() {
        let mut source = mixed_clipped_state([2, 1]);
        let target = mixed_clipped_state([4, 2]);
        for cell in &source.topology.graph.cells {
            let id = cell.id as usize;
            source.fields.density[id] = (id % 7) as f32 * 0.125;
            source.fields.gamma[id] = 1.0 - (id % 5) as f32 * 0.0625;
            source.fields.pressure[id] = id as f32 * 0.03125;
            source.fields.cell_velocity[3 * id] = id as f32 * 0.015625;
            source.fields.cell_velocity[3 * id + 1] = -(id as f32) * 0.0078125;
            source.fields.cell_velocity[3 * id + 2] = (id % 3) as f32 * 0.25;
        }
        for row in &source.topology.graph.rows {
            source.fields.face_velocity[row.id as usize] =
                (row.id as i32 % 11 - 5) as f32 * 0.03125;
        }
        let capacity = vec![1.0; target.topology.graph.cells.len()];
        let indexed = transfer_fields_3d(
            &source.topology.graph,
            &target.topology.graph,
            &source.fields,
            &capacity,
            &[],
        )
        .unwrap();
        let brute = transfer_fields_from_plan_3d(
            &source.topology.graph,
            &target.topology.graph,
            &source.fields,
            &capacity,
            brute_plan(&source.topology.graph, &target.topology.graph, &[]).unwrap(),
        )
        .unwrap();
        assert_transfer_exact(&indexed, &brute);
    }
    #[test]
    fn source_owned_split_and_merge_are_conservative_in_xyz_order() {
        let parent = graph(vec![cell(0, [0.0; 3], [2.0; 3])]);
        let mut children = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    children.push(cell(
                        children.len() as u32,
                        [x as f32, y as f32, z as f32],
                        [x as f32 + 1.0, y as f32 + 1.0, z as f32 + 1.0],
                    ));
                }
            }
        }
        let children = graph(children);
        let source = fields(&parent, vec![0.375]);
        let split = transfer_fields_3d(&parent, &children, &source, &[1.0; 8], &[]).unwrap();
        assert_eq!(split.plan.cell_sources, vec![0; 8]);
        assert_eq!(
            split.density.iter().sum::<f32>().to_bits(),
            3.0f32.to_bits()
        );
        let merged = transfer_fields_3d(
            &children,
            &parent,
            &fields(&children, split.density),
            &[1.0],
            &[],
        )
        .unwrap();
        assert_eq!(merged.density[0].to_bits(), 0.375f32.to_bits());
    }
    #[test]
    fn uncovered_volume_requires_explicit_new_air_box() {
        let source = graph(vec![cell(0, [0.0; 3], [1.0, 2.0, 2.0])]);
        let target = graph(vec![cell(0, [0.0; 3], [2.0; 3])]);
        assert!(plan_transfer_3d(&source, &target, &[]).is_err());
        let plan = plan_transfer_3d(
            &source,
            &target,
            &[NewAirCoverage3d {
                minimum_fine: [0.0; 3],
                maximum_exclusive_fine: [2.0; 3],
            }],
        )
        .unwrap();
        assert_eq!(plan.cell_sources, vec![0, u32::MAX]);
        assert_eq!(plan.cell_areas, vec![4.0, 4.0]);
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Fixture {
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Case {
        id: String,
        source: Geometry,
        target: Vec<Box3>,
        new_air: Vec<NewAirCoverage3d>,
        expected: TransferPlan,
    }
    #[derive(Deserialize)]
    struct Geometry {
        dimensions: [u32; 3],
        cells: Vec<Box3>,
    }
    #[derive(Clone, Deserialize)]
    struct Box3 {
        id: u32,
        lower: [f32; 3],
        widths: [f32; 3],
    }
    fn fixture_graph(dimensions: [u32; 3], boxes: &[Box3]) -> Graph {
        let mut g = graph(
            boxes
                .iter()
                .map(|b| {
                    cell(
                        b.id,
                        b.lower,
                        std::array::from_fn(|a| b.lower[a] + b.widths[a]),
                    )
                })
                .collect(),
        );
        g.dimensions = dimensions.map(|v| v as f32);
        g
    }
    #[test]
    fn plans_match_headless_typescript_source_goldens() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../core/testdata/transfer3d-golden.json"
        ))
        .unwrap();
        for case in fixture.cases {
            let source = fixture_graph(case.source.dimensions, &case.source.cells);
            let target = fixture_graph(case.source.dimensions, &case.target);
            let actual = plan_transfer_3d(&source, &target, &case.new_air).unwrap();
            assert_eq!(
                actual.cell_offsets, case.expected.cell_offsets,
                "{} offsets",
                case.id
            );
            assert_eq!(
                actual.cell_sources, case.expected.cell_sources,
                "{} sources",
                case.id
            );
            assert_eq!(
                actual
                    .cell_areas
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                case.expected
                    .cell_areas
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{} volumes",
                case.id
            );
            assert_eq!(
                actual.face_offsets, case.expected.face_offsets,
                "{} face offsets",
                case.id
            );
        }
    }
}
