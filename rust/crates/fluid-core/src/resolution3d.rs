//! CPU authority for production three-dimensional sparse-CM12 adaptivity.
//!
//! This is intentionally a separate policy from the Slice (2-D) implementation:
//! activity owns a complete 3x3x3 support stencil and all measurements and
//! closure operate in XYZ. Floating point reductions preserve cell/source order.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::resolution::{activity_reason, resolution_fault, ActivityPolicy, SurfaceProofState};
use crate::topology::{BrickSeed, CompiledTopology, BRICK_FINE_RESOLUTION};
use crate::types::{Fields, RowKind};

const B: i32 = BRICK_FINE_RESOLUTION;
const ACTIVITY_FIXED: f64 = 65_536.0;
const VOLUME_ROUNDOFF_RATIO: f64 = 9.536_743_164_062_5e-7;
const CENTER_BIT: i32 = 13;
const SUPPORT_MASK_3D: u32 = 0x07ff_ffff;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolutionRegion3d {
    pub minimum_fine: [f64; 3],
    pub maximum_fine: [f64; 3],
    pub minimum_cell_width: u8,
    pub maximum_cell_width: Option<u8>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrickActivityHistory3d {
    pub score_byte: u8,
    pub reasons: u32,
    pub hot_epochs: u8,
    pub quiet_epochs: u8,
    pub proof_epochs: u8,
    pub mean_density: f32,
    pub density_moments: [f32; 3],
    pub mean_velocity: [f32; 3],
    pub velocity_travel: f32,
    /// Bits are `x + 3*y + 9*z`, with each coordinate in -1..=1.
    pub support_mask: u32,
    pub swept_support_mask: u32,
    pub last_transition_step: u32,
    pub surface_proof: Option<SurfaceProofState>,
}

impl Default for BrickActivityHistory3d {
    fn default() -> Self {
        Self {
            score_byte: 0,
            reasons: activity_reason::FIRST_STEP,
            hot_epochs: 0,
            quiet_epochs: 0,
            proof_epochs: 0,
            mean_density: 0.0,
            density_moments: [0.0; 3],
            mean_velocity: [0.0; 3],
            velocity_travel: 0.0,
            support_mask: 0,
            swept_support_mask: 0,
            last_transition_step: 0,
            surface_proof: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolutionPolicyState3d {
    pub accepted_steps: u32,
    pub accepted_generation: u32,
    pub scheduling_cursor: usize,
    pub scheduling_credits: usize,
    pub history: BTreeMap<u32, BrickActivityHistory3d>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(default)]
pub struct ResolutionPolicyOptions3d {
    pub policy: ActivityPolicy,
    pub refinement_regions: Vec<ResolutionRegion3d>,
    pub static_boundary_floor_by_brick: BTreeMap<u32, u8>,
    pub moving_rigid_bodies: bool,
    pub injection_demanded_brick_keys: BTreeSet<u32>,
    /// Inclusive fine-coordinate bounds of this frame's liquid source demand.
    /// Every intersected sparse-world page is made resident before transport.
    pub injection_bounds_fine: Option<([f64; 3], [f64; 3])>,
    /// Conservative physical-source receiver boxes in finest-cell units.
    /// Unlike editor injection, these persist for every active source frame.
    pub source_demand_bounds_fine: Vec<([f64; 3], [f64; 3])>,
    pub frozen_brick_keys: BTreeSet<u32>,
    pub maximum_leaves: Option<usize>,
    pub maximum_cells: Option<usize>,
    pub allocate_missing_pages: bool,
    pub free_leaf_ids: Vec<u32>,
}

impl Default for ResolutionPolicyOptions3d {
    fn default() -> Self {
        Self {
            policy: ActivityPolicy::default(),
            refinement_regions: vec![],
            static_boundary_floor_by_brick: BTreeMap::new(),
            moving_rigid_bodies: false,
            injection_demanded_brick_keys: BTreeSet::new(),
            injection_bounds_fine: None,
            source_demand_bounds_fine: vec![],
            frozen_brick_keys: BTreeSet::new(),
            maximum_leaves: None,
            maximum_cells: None,
            allocate_missing_pages: true,
            free_leaf_ids: vec![],
        }
    }
}

impl ResolutionPolicyOptions3d {
    pub fn production() -> Self {
        Self::default()
    }
}

/// Exact resident-WGSL continuous-inflow support box. `velocity_fine` must
/// already include the frame's averaged source strength. The two-step reach
/// and radius-plus-one padding match `geometricTransportMaterialDemand`.
pub fn continuous_inflow_demand_bounds_3d(
    outlet_fine: [f32; 3],
    velocity_fine: [f32; 3],
    radius_fine: f32,
    dt: f32,
) -> Result<Option<([f64; 3], [f64; 3])>, ResolutionError3d> {
    if !outlet_fine.into_iter().all(f32::is_finite)
        || !velocity_fine.into_iter().all(f32::is_finite)
        || !radius_fine.is_finite()
        || !dt.is_finite()
        || radius_fine < 0.0
        || dt <= 0.0
    {
        return Err(ResolutionError3d::InvalidStep);
    }
    let speed2 = velocity_fine
        .into_iter()
        .fold(0.0_f32, |sum, v| sum + v * v);
    if radius_fine <= 0.0 || speed2 <= 1e-12 {
        return Ok(None);
    }
    let endpoint: [f32; 3] =
        std::array::from_fn(|a| outlet_fine[a] + (2.0_f32 * velocity_fine[a]) * dt);
    let padding = radius_fine + 1.0;
    Ok(Some((
        std::array::from_fn(|a| (outlet_fine[a].min(endpoint[a]) - padding) as f64),
        std::array::from_fn(|a| (outlet_fine[a].max(endpoint[a]) + padding) as f64),
    )))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrickResolutionReceipt3d {
    pub brick_key: u32,
    pub accepted_resolution: u8,
    pub requested_resolution: u8,
    pub scheduled_resolution: u8,
    pub accepted_active: bool,
    pub candidate_active: bool,
    pub score_byte: u8,
    pub reasons: u32,
    pub plan_reasons: u32,
    pub support_mask: u32,
    pub swept_support_mask: u32,
    pub fault_bits: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolutionPolicyReceipt3d {
    pub topology_epoch: bool,
    pub accepted_generation: u32,
    pub candidate_generation: u32,
    pub measured_brick_count: usize,
    pub surface_brick_count: usize,
    pub occupied_brick_count: usize,
    pub activated_brick_count: usize,
    pub allocated_brick_count: usize,
    pub claimed_leaf_ids: Vec<u32>,
    pub retired_brick_count: usize,
    pub promoted_brick_count: usize,
    pub demoted_brick_count: usize,
    pub deferred_demotion_count: usize,
    pub maximum_score_byte: u8,
    pub fault_bits: u32,
    pub bricks: Vec<BrickResolutionReceipt3d>,
}

#[derive(Clone, Debug)]
pub struct ResolutionPolicyDecision3d {
    pub candidate_bricks: Vec<BrickSeed>,
    pub state: ResolutionPolicyState3d,
    pub receipt: ResolutionPolicyReceipt3d,
}

#[derive(Clone, Debug)]
pub struct ProjectedTransportSupportDecision3d {
    pub candidate_bricks: Vec<BrickSeed>,
    pub demanded_brick_keys: BTreeSet<u32>,
    pub claimed_leaf_ids: Vec<u32>,
    pub fault_bits: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolutionError3d {
    FieldShape,
    InvalidStep,
    InvalidResolution(u8),
    InvalidOrDuplicateFreeLeaf(u32),
    ActiveFreeLeaf(u32),
    IdOverflow,
    StaticSolidShape,
}

impl std::fmt::Display for ResolutionError3d {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "{self:?}")
    }
}
impl std::error::Error for ResolutionError3d {}

/// Reproduce the resident shader's static-solid restriction evidence for each
/// authored page. The returned rung is the first of B8/B4/B2 whose volume and
/// oriented-face standard deviation exceeds `tolerance`.
pub fn static_boundary_floors_3d(
    topology: &CompiledTopology<3>,
    tolerance: f32,
) -> Result<BTreeMap<u32, u8>, ResolutionError3d> {
    let dimensions = topology.graph.dimensions.map(|v| v as i32);
    let expected = dimensions
        .into_iter()
        .try_fold(1_usize, |n, v| n.checked_mul(v.max(0) as usize))
        .ok_or(ResolutionError3d::StaticSolidShape)?;
    let voxels = &topology.graph.solid_voxel_fraction;
    if voxels.is_empty() {
        return Ok(BTreeMap::new());
    }
    if voxels.len() != expected || !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ResolutionError3d::StaticSolidShape);
    }
    let sample = |q: [i32; 3]| -> f32 {
        for axis in 0..3 {
            if q[axis] < 0 {
                return if topology.boundaries[2 * axis] == crate::geometry::BoundaryMode::Open {
                    0.0
                } else {
                    1.0
                };
            }
            if q[axis] >= dimensions[axis] {
                return if topology.boundaries[2 * axis + 1] == crate::geometry::BoundaryMode::Open {
                    0.0
                } else {
                    1.0
                };
            }
        }
        let i = q[0] as usize
            + dimensions[0] as usize * (q[1] as usize + dimensions[1] as usize * q[2] as usize);
        ((voxels[i].clamp(0.0, 1.0) * 255.0).round() / 255.0) as f32
    };
    let restriction_error = |origin: [i32; 3], rung: u8| -> f32 {
        let cell_span = B / rung as i32;
        let mut maximum = 0.0_f32;
        for z in 0..rung as i32 {
            for y in 0..rung as i32 {
                for x in 0..rung as i32 {
                    let lower = [
                        origin[0] + x * cell_span,
                        origin[1] + y * cell_span,
                        origin[2] + z * cell_span,
                    ];
                    let (mut sum, mut square_sum, mut count) = (0.0_f32, 0.0_f32, 0.0_f32);
                    for dz in 0..cell_span {
                        for dy in 0..cell_span {
                            for dx in 0..cell_span {
                                let value = sample([lower[0] + dx, lower[1] + dy, lower[2] + dz]);
                                sum += value;
                                square_sum += value * value;
                                count += 1.0;
                            }
                        }
                    }
                    let mean = sum / count;
                    maximum = maximum.max((square_sum / count - mean * mean).max(0.0).sqrt());
                }
            }
        }
        for axis in 0..3 {
            let u_axis = (axis + 1) % 3;
            let v_axis = (axis + 2) % 3;
            for face in 0..=rung as i32 {
                for macro_v in 0..rung as i32 {
                    for macro_u in 0..rung as i32 {
                        let (mut sum, mut square_sum, mut count) = (0.0_f32, 0.0_f32, 0.0_f32);
                        for dv in 0..cell_span {
                            for du in 0..cell_span {
                                let mut positive = origin;
                                positive[axis] += face * cell_span;
                                positive[u_axis] += macro_u * cell_span + du;
                                positive[v_axis] += macro_v * cell_span + dv;
                                let mut negative = positive;
                                negative[axis] -= 1;
                                let value = sample(negative).max(sample(positive));
                                sum += value;
                                square_sum += value * value;
                                count += 1.0;
                            }
                        }
                        let mean = sum / count;
                        maximum = maximum.max((square_sum / count - mean * mean).max(0.0).sqrt());
                    }
                }
            }
        }
        maximum
    };
    let mut floors = BTreeMap::new();
    for brick in &topology.bricks {
        let origin = brick.seed.coordinate.map(|v| v * B);
        let floor = [(4, 8), (2, 4), (1, 2)]
            .into_iter()
            .find_map(|(rung, floor)| {
                (restriction_error(origin, rung) > tolerance).then_some(floor)
            })
            .unwrap_or(1);
        floors.insert(brick.seed.key, floor);
    }
    Ok(floors)
}

pub fn initialize_resolution_policy_3d(topology: &CompiledTopology<3>) -> ResolutionPolicyState3d {
    ResolutionPolicyState3d {
        accepted_generation: topology.graph.topology_generation,
        history: topology
            .bricks
            .iter()
            .map(|b| (b.seed.key, BrickActivityHistory3d::default()))
            .collect(),
        ..ResolutionPolicyState3d::default()
    }
}

#[derive(Clone, Debug)]
struct Measurement {
    history: BrickActivityHistory3d,
    surface: bool,
    thin: bool,
    occupied: bool,
    deeply_enclosed: bool,
    curvature_floor: u8,
}

fn f(v: f64) -> f32 {
    v as f32
}
fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}
fn clamp_byte(v: f64) -> u8 {
    js_round(255.0 * v.clamp(0.0, 1.0)) as u8
}
fn valid_resolution(r: u8) -> Result<(), ResolutionError3d> {
    if matches!(r, 1 | 2 | 4 | 8) {
        Ok(())
    } else {
        Err(ResolutionError3d::InvalidResolution(r))
    }
}
fn span(b: &BrickSeed) -> i32 {
    b.span_bricks as i32
}
fn width(b: &BrickSeed, rung: u8) -> f64 {
    (B * span(b)) as f64 / rung as f64
}
fn bounds(b: &BrickSeed) -> ([i32; 3], [i32; 3]) {
    let lo = b.coordinate.map(|v| B * v);
    let size = B * span(b);
    (lo, lo.map(|v| v + size))
}
fn overlaps(a0: i32, a1: i32, b0: i32, b1: i32) -> bool {
    a1.min(b1) > a0.max(b0)
}

fn face_neighbors(bricks: &[BrickSeed], i: usize) -> Vec<usize> {
    let (al, ah) = bounds(&bricks[i]);
    bricks
        .iter()
        .enumerate()
        .filter_map(|(j, b)| {
            if i == j {
                return None;
            }
            let (bl, bh) = bounds(b);
            (0..3)
                .any(|axis| {
                    (ah[axis] == bl[axis] || bh[axis] == al[axis])
                        && (0..3)
                            .filter(|&a| a != axis)
                            .all(|a| overlaps(al[a], ah[a], bl[a], bh[a]))
                })
                .then_some(j)
        })
        .collect()
}

fn owner_at(bricks: &[BrickSeed], p: [f64; 3]) -> Option<usize> {
    bricks.iter().position(|b| {
        let (lo, hi) = bounds(b);
        (0..3).all(|a| p[a] >= lo[a] as f64 && p[a] < hi[a] as f64)
    })
}

fn thresholds(policy: &ActivityPolicy, dt: f64, cell_size: f64) -> [f32; 4] {
    if policy.coarse_first {
        let finest = f((2.0 * policy.energy_threshold).sqrt() * dt / cell_size);
        [
            f(finest as f64 / 8.0),
            f(finest as f64 / 4.0),
            f(finest as f64 / 2.0),
            finest,
        ]
    } else {
        [
            f(policy.two_travel_cells / 2.0),
            f(policy.two_travel_cells),
            f(policy.four_travel_cells),
            f(policy.finest_travel_cells),
        ]
    }
}
fn velocity_floor(travel: f32, ts: [f32; 4], enabled: bool) -> u8 {
    if !enabled {
        return 1;
    }
    for i in (0..4).rev() {
        if travel >= ts[i] {
            return [1, 2, 4, 8][i];
        }
    }
    1
}

fn region_bounds(b: &BrickSeed, regions: &[ResolutionRegion3d]) -> (u8, u8) {
    let (lo, hi) = bounds(b);
    let nominal = (B * span(b)) as f64;
    let (mut floor, mut ceiling) = (1_u8, 0_u8);
    for r in regions {
        if (0..3).all(|a| hi[a] as f64 > r.minimum_fine[a] && r.maximum_fine[a] > lo[a] as f64) {
            floor = floor.max(r.minimum_cell_width);
        }
        if (0..3).all(|a| lo[a] as f64 >= r.minimum_fine[a] && hi[a] as f64 <= r.maximum_fine[a]) {
            if let Some(c) = r.maximum_cell_width {
                ceiling = if ceiling == 0 { c } else { ceiling.min(c) };
            }
        }
    }
    (
        (nominal / floor as f64).clamp(1.0, 8.0) as u8,
        if ceiling == 0 {
            1
        } else {
            (nominal / ceiling as f64).clamp(1.0, 8.0) as u8
        },
    )
}
fn apply_regions(r: u8, b: &BrickSeed, regions: &[ResolutionRegion3d]) -> u8 {
    let (cap, floor) = region_bounds(b, regions);
    r.min(cap).max(floor)
}

fn offset_bit(dx: i32, dy: i32, dz: i32) -> u32 {
    (dx + 1 + 3 * (dy + 1) + 9 * (dz + 1)) as u32
}
fn decode_offset(bit: i32) -> [i32; 3] {
    [bit % 3 - 1, (bit / 3) % 3 - 1, bit / 9 - 1]
}
fn neighbor_probe(source: &BrickSeed, d: [i32; 3]) -> [f64; 3] {
    let s = span(source);
    std::array::from_fn(|a| {
        (B * (source.coordinate[a]
            + if d[a] < 0 {
                -1
            } else if d[a] > 0 {
                s
            } else {
                0
            })) as f64
            + 0.5
    })
}

fn measure(
    topology: &CompiledTopology<3>,
    fields: &Fields,
    bi: usize,
    old: Option<&BrickActivityHistory3d>,
    policy: &ActivityPolicy,
    topology_epoch: bool,
    dt: f64,
    ts: [f32; 4],
) -> Measurement {
    let brick = &topology.bricks[bi];
    if !brick.seed.active {
        let mut h = old.cloned().unwrap_or_default();
        h.score_byte = 0;
        h.reasons &= 0x3c00;
        h.mean_density = 0.0;
        h.density_moments = [0.0; 3];
        h.mean_velocity = [0.0; 3];
        h.velocity_travel = 0.0;
        h.support_mask = 0;
        h.swept_support_mask = 0;
        return Measurement {
            history: h,
            surface: false,
            thin: false,
            occupied: false,
            deeply_enclosed: false,
            curvature_floor: 1,
        };
    }
    let cells =
        &topology.graph.cells[brick.cell_range.start as usize..brick.cell_range.end as usize];
    let mut density_sum = 0.0_f64;
    let mut moments = [0.0_f64; 3];
    let mut momentum = [0.0_f32; 3];
    let mut momentum_mass = 0.0_f32;
    let mut deformation = 0.0_f32;
    let mut predicted = 0.0_f32;
    let mut detail = 0.0_f32;
    let mut travel = 0.0_f32;
    let mut axes = 0_u8;
    let mut occupied_cell = false;
    let mut substantial = false;
    let mut thin = false;
    let mut cut = false;
    let mut density_surface = false;
    let mut support = 0_u32;
    let mut swept = 0_u32;
    let (mut normal_min, mut normal_max) = ([1.0_f64; 3], [-1.0_f64; 3]);
    let feature_density = policy.residency_density.max(policy.thin_feature_density);
    let (blo, bhi) = bounds(&brick.seed);
    let r = brick.seed.resolution as usize;
    for (local_index, cell) in cells.iter().enumerate() {
        let id = cell.id as usize;
        let rho = fields.density[id] as f64;
        let cap = (fields.capacity[id] as f64).max(1e-6);
        let fill = rho / cap;
        density_sum += js_round(rho * ACTIVITY_FIXED);
        let local = [
            local_index % r,
            (local_index / r) % r,
            local_index / (r * r),
        ];
        for a in 0..3 {
            moments[a] += js_round(
                rho * (2 * local[a] as i32 + 1 - r as i32) as f64 / r as f64 * ACTIVITY_FIXED,
            );
        }
        cut |= fields.capacity[id] < 0.999;
        occupied_cell |= rho > policy.residency_density;
        substantial |= fill > policy.surface_density_minimum;
        let wet = fill >= 0.5;
        let v = [
            fields.cell_velocity[3 * id],
            fields.cell_velocity[3 * id + 1],
            fields.cell_velocity[3 * id + 2],
        ];
        if policy.coarse_first && wet {
            for a in 0..3 {
                momentum[a] = f(momentum[a] as f64 + f(v[a] as f64 * rho) as f64);
            }
            momentum_mass = f(momentum_mass as f64 + rho);
        }
        let mut sweep_min = [0.0_f64; 3];
        let mut sweep_max = [0.0_f64; 3];
        let roundoff = VOLUME_ROUNDOFF_RATIO * cap * cell.measure as f64;
        for &row_id in &topology.graph.incidences[id] {
            let row = &topology.graph.rows[row_id as usize];
            let a = row.axis as usize;
            let open = row.open_fraction as f64;
            let wall = row.solid_velocity as f64;
            let stored = fields
                .face_velocity
                .get(row.id as usize)
                .copied()
                .unwrap_or(v[a]) as f64;
            let fluid = if open > 1e-6 {
                (stored - (1.0 - open) * wall) / open
            } else {
                wall
            };
            let endpoint = fluid + dt * fields.acceleration_fine[a] as f64;
            for candidate in [fluid, endpoint] {
                if dt * row.measure as f64 * candidate.abs() > roundoff {
                    sweep_min[a] = sweep_min[a].min(candidate);
                    sweep_max[a] = sweep_max[a].max(candidate);
                }
            }
        }
        let mut interface = false;
        let mut exposed = 0_u8;
        for &row_id in &topology.graph.incidences[id] {
            let row = &topology.graph.rows[row_id as usize];
            let a = row.axis as usize;
            let own = row
                .terms
                .iter()
                .find(|t| t.cell_id == cell.id)
                .expect("incidence term");
            let mut side_has_fluid = false;
            let mut has_other = false;
            for term in row
                .terms
                .iter()
                .filter(|t| t.cell_id != cell.id && own.coefficient * t.coefficient < 0.0)
            {
                has_other = true;
                let nid = term.cell_id as usize;
                let nfill = fields.density[nid] as f64 / (fields.capacity[nid] as f64).max(1e-6);
                side_has_fluid |= nfill > feature_density;
                let crosses = (nfill >= 0.5) != wet;
                let same_brick = topology.graph.cells[nid].brick_key == Some(brick.seed.key);
                if crosses && (!wet || same_brick || policy.coarse_first) {
                    interface = true;
                    density_surface = true;
                    axes |= 1 << a;
                }
                if crosses {
                    let liquid = if wet { id } else { nid };
                    predicted = predicted
                        .max(f(dt * (fields.cell_velocity[3 * liquid + a] as f64).abs()
                            / (0.25 * row.distance as f64).max(1e-12)));
                }
                if wet && nfill >= 0.5 {
                    let mut dv = 0.0_f64;
                    for q in 0..3 {
                        dv = dv.max((v[q] as f64 - fields.cell_velocity[3 * nid + q] as f64).abs());
                    }
                    deformation =
                        deformation.max(f(dt * dv / (0.15 * row.distance as f64).max(1e-12)));
                }
            }
            if row.kind == RowKind::SparseAir && !has_other && wet {
                interface = true;
                axes |= 1 << a;
                predicted = predicted.max(f(
                    dt * (v[a] as f64).abs() / (0.25 * row.distance as f64).max(1e-12)
                ));
            }
            if rho > feature_density && !side_has_fluid {
                let side = usize::from(row.center[a] > cell.center[a]);
                exposed |= 1 << (2 * a + side);
            }
        }
        let thickness =
            rho.clamp(0.0, 1.0) * cell.widths.into_iter().fold(f32::INFINITY, f32::min) as f64;
        let cell_thin = fill > feature_density
            && thickness < policy.thin_feature_cells
            && (0..3).any(|a| exposed & (3 << (2 * a)) == (3 << (2 * a)));
        thin |= cell_thin;
        if policy.coarse_first
            && (interface || cell_thin)
            && fields.interface_normal.len() >= 3 * id + 3
        {
            let n = [
                fields.interface_normal[3 * id] as f64,
                fields.interface_normal[3 * id + 1] as f64,
                fields.interface_normal[3 * id + 2] as f64,
            ];
            if n.iter().map(|x| x * x).sum::<f64>() > 0.5 {
                for a in 0..3 {
                    normal_min[a] = normal_min[a].min(n[a]);
                    normal_max[a] = normal_max[a].max(n[a]);
                }
            }
        }
        if (interface && wet) || cell_thin || (policy.coarse_first && wet) {
            travel = travel.max(f(dt * (v[0] as f64).hypot(v[1] as f64).hypot(v[2] as f64)));
        }
        if interface || cell_thin || rho != 0.0 {
            let mut ranges = [[0_i32; 2]; 3];
            for a in 0..3 {
                let lower = cell.minimum[a] as f64 <= blo[a] as f64;
                let upper = cell.maximum[a] as f64 >= bhi[a] as f64;
                ranges[a] = match (lower, upper) {
                    (true, true) => [-1, 1],
                    (true, false) => [-1, 0],
                    (false, true) => [0, 1],
                    (false, false) => [0, 0],
                };
            }
            for dz in ranges[2][0]..=ranges[2][1] {
                for dy in ranges[1][0]..=ranges[1][1] {
                    for dx in ranges[0][0]..=ranges[0][1] {
                        if dx != 0 || dy != 0 || dz != 0 {
                            if interface || cell_thin {
                                support |= 1 << offset_bit(dx, dy, dz);
                            }
                        }
                    }
                }
            }
            if rho != 0.0 {
                swept |= 1 << CENTER_BIT;
                let mut lo = [0; 3];
                let mut hi = [0; 3];
                for a in 0..3 {
                    lo[a] = if cell.minimum[a] as f64 + dt * sweep_min[a] < blo[a] as f64 {
                        -1
                    } else {
                        0
                    };
                    hi[a] = if cell.maximum[a] as f64 + dt * sweep_max[a] > bhi[a] as f64 {
                        1
                    } else {
                        0
                    };
                }
                for dz in lo[2]..=hi[2] {
                    for dy in lo[1]..=hi[1] {
                        for dx in lo[0]..=hi[0] {
                            if dx != 0 || dy != 0 || dz != 0 {
                                let bit = 1 << offset_bit(dx, dy, dz);
                                support |= bit;
                                swept |= bit;
                            }
                        }
                    }
                }
            }
        }
    }
    if r > 1 {
        let base = brick.cell_range.start as usize;
        for z in (0..r).step_by(2) {
            for y in (0..r).step_by(2) {
                for x in (0..r).step_by(2) {
                    let mut values = [0.0_f64; 8];
                    for dz in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let lane = dx + 2 * dy + 4 * dz;
                                values[lane] = js_round(
                                    fields.density[base + x + dx + r * (y + dy + r * (z + dz))]
                                        as f64
                                        * ACTIVITY_FIXED,
                                );
                            }
                        }
                    }
                    let sum: f64 = values.iter().sum();
                    for value in values {
                        detail = detail.max(f((8.0 * value - sum).abs() / (8.0 * ACTIVITY_FIXED)));
                    }
                }
            }
        }
    }
    let count = cells.len().max(1) as f64;
    let mean_density = f(density_sum / (count * ACTIVITY_FIXED));
    let density_moments = moments.map(|v| f(v / (count * ACTIVITY_FIXED)));
    let mass_fine =
        f(density_sum / ACTIVITY_FIXED * cells.first().map_or(0.0, |c| c.measure as f64));
    let occupied = occupied_cell
        && (substantial || thin)
        && mass_fine as f64 >= policy.residency_mass_fine_cells;
    let surface = occupied && axes != 0;
    let shape = if axes.count_ones() >= 2 { 1.0 } else { 0.0 };
    let temporal = if policy.coarse_first {
        0.0
    } else {
        old.map_or(0.0, |o| {
            let mut t = (mean_density as f64 - o.mean_density as f64).abs() / 0.05;
            for a in 0..3 {
                t = t.max((density_moments[a] as f64 - o.density_moments[a] as f64).abs() / 0.02);
            }
            t
        })
    };
    let scored_detail = if surface && !thin && shape == 0.0 {
        0.0
    } else {
        detail as f64
    };
    let dynamic = if surface || thin {
        (deformation as f64).max(temporal)
    } else {
        0.0
    };
    let feature = dynamic
        .max(predicted as f64)
        .max(shape)
        .max(if thin { 1.0 } else { 0.0 })
        .max((scored_detail / policy.detail_tolerance - 1.0).max(0.0));
    let normal_diameter = if (0..3).all(|a| normal_max[a] >= normal_min[a]) {
        (0..3)
            .map(|a| (normal_max[a] - normal_min[a]).max(0.0).powi(2))
            .sum::<f64>()
            .sqrt()
    } else {
        0.0
    };
    let mut curvature_floor = 1;
    while curvature_floor < 8
        && normal_diameter / curvature_floor as f64 > policy.curvature_tolerance
    {
        curvature_floor *= 2;
    }
    let coarse_score = (normal_diameter
        / (brick.seed.resolution as f64 * policy.curvature_tolerance.max(0.02)))
    .max(travel as f64 / (ts[3] as f64).max(1e-6));
    let score_byte = clamp_byte(if policy.coarse_first {
        coarse_score
    } else {
        feature.max(if surface || thin {
            travel as f64 / (ts[3] as f64).max(1e-6)
        } else {
            0.0
        })
    });
    let mut reasons = 0;
    if surface {
        reasons |= activity_reason::SURFACE;
    }
    if deformation as f64 >= policy.emergency_score {
        reasons |= activity_reason::DEFORMATION;
    }
    if temporal > 0.0 {
        reasons |= activity_reason::TEMPORAL;
    }
    if detail as f64 > policy.detail_tolerance {
        reasons |= activity_reason::FINE_DETAIL;
    }
    if predicted as f64 >= policy.emergency_score {
        reasons |= activity_reason::PREDICTED_FACE;
    }
    if old.is_none() {
        reasons |= activity_reason::FIRST_STEP;
    }
    if occupied {
        reasons |= activity_reason::OCCUPIED;
    }
    if velocity_floor(travel, ts, policy.activity_signals) > 1 {
        reasons |= activity_reason::VELOCITY_FLOOR;
    }
    if thin {
        reasons |= activity_reason::THIN_FLUID;
    }
    if cut {
        reasons |= activity_reason::CUT_BOUNDARY;
    }
    if density_surface {
        reasons |= activity_reason::DENSITY_SURFACE;
    }
    if policy.coarse_first {
        reasons |= (curvature_floor as u32) << 16;
    }
    let hot = policy.activity_signals && feature >= policy.promote_score;
    let quiet = !thin
        && score_byte as f64 / 255.0 <= policy.demote_score
        && scored_detail <= policy.detail_tolerance;
    let hot_epochs = if topology_epoch {
        if hot {
            old.map_or(0, |o| o.hot_epochs).saturating_add(1)
        } else {
            0
        }
    } else {
        old.map_or(0, |o| o.hot_epochs)
    };
    let quiet_epochs = if topology_epoch {
        if quiet {
            old.map_or(0, |o| o.quiet_epochs).saturating_add(1)
        } else {
            0
        }
    } else {
        old.map_or(0, |o| o.quiet_epochs)
    };
    let neighbors = face_neighbors(
        &topology
            .bricks
            .iter()
            .map(|r| r.seed.clone())
            .collect::<Vec<_>>(),
        bi,
    );
    let mut enclosed_sides = [false; 6];
    for &ni in &neighbors {
        let (lo, hi) = bounds(&topology.bricks[ni].seed);
        for a in 0..3 {
            if hi[a] == blo[a] {
                enclosed_sides[2 * a] = true;
            }
            if lo[a] == bhi[a] {
                enclosed_sides[2 * a + 1] = true;
            }
        }
    }
    let deeply_enclosed = occupied
        && enclosed_sides.iter().all(|&x| x)
        && neighbors.iter().all(|&ni| {
            let n = &topology.bricks[ni];
            n.seed.active
                && n.cell_range.clone().all(|id| {
                    fields.density[id as usize] as f64
                        / (fields.capacity[id as usize] as f64).max(1e-6)
                        >= 0.5
                })
        });
    let mean_velocity = if momentum_mass > 1e-8 {
        momentum.map(|v| f(v as f64 / momentum_mass as f64))
    } else {
        [0.0; 3]
    };
    Measurement {
        history: BrickActivityHistory3d {
            score_byte,
            reasons,
            hot_epochs,
            quiet_epochs,
            proof_epochs: old.map_or(0, |o| o.proof_epochs),
            mean_density,
            density_moments,
            mean_velocity,
            velocity_travel: travel,
            support_mask: support & SUPPORT_MASK_3D,
            swept_support_mask: swept & SUPPORT_MASK_3D,
            last_transition_step: old.map_or(0, |o| o.last_transition_step),
            surface_proof: old.and_then(|o| o.surface_proof.clone()),
        },
        surface,
        thin,
        occupied,
        deeply_enclosed,
        curvature_floor,
    }
}

fn approach_travel(delta: [f64; 3], sweep: [f64; 3], extent: f64) -> f64 {
    let (mut enter, mut leave, mut approach) = (0.0_f64, 1.0_f64, 0.0_f64);
    for axis in 0..3 {
        let distance = delta[axis].abs();
        let motion = sweep[axis];
        if distance >= extent {
            let closing = delta[axis].signum() * motion;
            if closing <= 0.0 {
                return 0.0;
            }
            let axis_enter = (distance - extent) / closing;
            if axis_enter > enter || approach == 0.0 {
                approach = closing;
            } else if axis_enter == enter {
                approach = approach.min(closing);
            }
            enter = enter.max(axis_enter);
            leave = leave.min((distance + extent) / closing);
        } else if motion.abs() > 1e-8 {
            leave = leave.min((extent + motion.signum() * delta[axis]) / motion.abs());
        }
    }
    if enter < leave {
        approach
    } else {
        0.0
    }
}

/// Retains already represented detail when an interface is moving toward this
/// page. Prediction cannot create a finer rung than the receiver already owns.
fn incoming_floor(
    topology: &CompiledTopology<3>,
    bi: usize,
    measurements: &BTreeMap<u32, Measurement>,
    policy: &ActivityPolicy,
) -> u8 {
    if policy.anticipation_seconds <= 0.0 {
        return 1;
    }
    let brick = &topology.bricks[bi].seed;
    let (lo, hi) = bounds(brick);
    let center: [f64; 3] = std::array::from_fn(|a| 0.5 * (lo[a] + hi[a]) as f64);
    let receiver = &measurements[&brick.key];
    let mut required = 1;
    for source_record in &topology.bricks {
        let source = &source_record.seed;
        if !source.active
            || source.key == brick.key
            || (0..3).any(|a| {
                (source.coordinate[a] - brick.coordinate[a]).abs()
                    > policy.anticipation_radius_bricks
            })
        {
            continue;
        }
        let m = &measurements[&source.key];
        if !m.occupied || !(m.surface || m.thin) {
            continue;
        }
        let (sl, sh) = bounds(source);
        let source_center: [f64; 3] = std::array::from_fn(|a| 0.5 * (sl[a] + sh[a]) as f64);
        let sweep = std::array::from_fn(|a| {
            policy.anticipation_seconds
                * (m.history.mean_velocity[a] - receiver.history.mean_velocity[a]) as f64
        });
        if sweep.iter().map(|v| v * v).sum::<f64>().sqrt() <= 1.0 {
            continue;
        }
        let delta: [f64; 3] = std::array::from_fn(|a| center[a] - source_center[a]);
        let extent = 0.5 * B as f64 * (span(brick) + span(source)) as f64;
        let approach = approach_travel(delta, sweep, extent);
        if approach <= 1.0 {
            continue;
        }
        let gap = (0..3)
            .map(|a| (delta[a].abs() - extent).max(0.0).powi(2))
            .sum::<f64>()
            .sqrt();
        let demand = B as f64 * (approach / (B as f64).max(gap + B as f64)).min(1.0);
        let mut rung = 1;
        while rung < 8 && (rung as f64) < demand {
            rung *= 2;
        }
        required = required.max(rung);
    }
    required
}

fn validate_inputs(
    topology: &CompiledTopology<3>,
    fields: &Fields,
    dt: f64,
    cell_size: f64,
) -> Result<(), ResolutionError3d> {
    let n = topology.graph.cells.len();
    if fields.density.len() != n
        || fields.capacity.len() != n
        || fields.cell_velocity.len() != 3 * n
    {
        return Err(ResolutionError3d::FieldShape);
    }
    if !(dt > 0.0 && dt.is_finite() && cell_size > 0.0 && cell_size.is_finite()) {
        return Err(ResolutionError3d::InvalidStep);
    }
    Ok(())
}
fn validate_free(bricks: &[BrickSeed], free: &[u32]) -> Result<BTreeSet<u32>, ResolutionError3d> {
    let mut set = BTreeSet::new();
    for &id in free {
        if !set.insert(id) {
            return Err(ResolutionError3d::InvalidOrDuplicateFreeLeaf(id));
        }
        if bricks.iter().any(|b| b.id == id && b.active) {
            return Err(ResolutionError3d::ActiveFreeLeaf(id));
        }
    }
    Ok(set)
}

fn close_region_caps(
    bricks: &[BrickSeed],
    targets: &mut BTreeMap<u32, u8>,
    regions: &[ResolutionRegion3d],
) {
    if regions.is_empty() {
        return;
    }
    let mut caps: BTreeMap<_, _> = bricks
        .iter()
        .map(|b| (b.key, region_bounds(b, regions).0))
        .collect();
    loop {
        let mut changed = false;
        for i in 0..bricks.len() {
            for j in face_neighbors(bricks, i) {
                let cap = (2 * caps[&bricks[j].key] as i32 * span(&bricks[i]) / span(&bricks[j]))
                    .clamp(1, 8) as u8;
                if cap < caps[&bricks[i].key] {
                    caps.insert(bricks[i].key, cap);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for b in bricks {
        targets.insert(b.key, targets[&b.key].min(caps[&b.key]));
    }
}
fn close_two_to_one(
    bricks: &[BrickSeed],
    _active: &BTreeMap<u32, bool>,
    targets: &mut BTreeMap<u32, u8>,
    regions: &[ResolutionRegion3d],
) {
    loop {
        let mut changed = false;
        for i in 0..bricks.len() {
            for j in face_neighbors(bricks, i) {
                if j <= i {
                    continue;
                }
                let aw = width(&bricks[i], targets[&bricks[i].key]);
                let bw = width(&bricks[j], targets[&bricks[j].key]);
                if aw.max(bw) > 2.0 * aw.min(bw) {
                    let ci = if aw > bw { i } else { j };
                    let cur = targets[&bricks[ci].key];
                    let raised = apply_regions((2 * cur).min(8), &bricks[ci], regions);
                    if raised != cur {
                        targets.insert(bricks[ci].key, raised);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}

fn allocate_frontier(
    topology: &CompiledTopology<3>,
    accepted: &[BrickSeed],
    measurements: &BTreeMap<u32, Measurement>,
    options: &ResolutionPolicyOptions3d,
) -> Result<(Vec<BrickSeed>, BTreeSet<u32>, Vec<u32>, BTreeSet<u32>), ResolutionError3d> {
    let free = validate_free(accepted, &options.free_leaf_ids)?;
    let mut working: Vec<_> = accepted
        .iter()
        .filter(|b| !free.contains(&b.id))
        .cloned()
        .collect();
    let mut demand = options.injection_demanded_brick_keys.clone();
    let mut allocated = BTreeSet::new();
    let mut claimed = Vec::new();
    let mut next_id = accepted.iter().map(|b| b.id).max().map_or(Ok(0), |v| {
        v.checked_add(1).ok_or(ResolutionError3d::IdOverflow)
    })?;
    let mut next_key = accepted.iter().map(|b| b.key).max().map_or(Ok(0), |v| {
        v.checked_add(1).ok_or(ResolutionError3d::IdOverflow)
    })?;
    let mut free_stack = options.free_leaf_ids.clone();
    for (minimum, maximum) in options
        .injection_bounds_fine
        .into_iter()
        .chain(options.source_demand_bounds_fine.iter().copied())
    {
        let lower = minimum.map(|v| (v / B as f64).floor() as i32);
        let upper = maximum.map(|v| (v / B as f64).floor() as i32);
        for z in lower[2]..=upper[2] {
            for y in lower[1]..=upper[1] {
                for x in lower[0]..=upper[0] {
                    let coordinate = [x, y, z];
                    let p = coordinate.map(|v| (v * B) as f64 + 0.5);
                    if (0..3).any(|a| p[a] < 0.0 || p[a] >= topology.graph.dimensions[a] as f64) {
                        continue;
                    }
                    let ri = if let Some(i) = owner_at(&working, p) {
                        i
                    } else {
                        if !options.allocate_missing_pages {
                            continue;
                        }
                        let id = if let Some(id) = free_stack.pop() {
                            id
                        } else {
                            let id = next_id;
                            next_id = next_id
                                .checked_add(1)
                                .ok_or(ResolutionError3d::IdOverflow)?;
                            id
                        };
                        let key = next_key;
                        next_key = next_key
                            .checked_add(1)
                            .ok_or(ResolutionError3d::IdOverflow)?;
                        working.push(BrickSeed {
                            id,
                            key,
                            coordinate,
                            span_bricks: 1,
                            resolution: 8,
                            active: true,
                            density: vec![],
                            gamma: vec![],
                            refinement_region_scale: None,
                        });
                        claimed.push(id);
                        allocated.insert(key);
                        working.len() - 1
                    };
                    demand.insert(working[ri].key);
                }
            }
        }
    }
    let mut sources: Vec<_> = accepted.iter().filter(|b| b.active).collect();
    sources.sort_by_key(|b| b.key);
    for source in sources {
        let mask = measurements[&source.key].history.swept_support_mask;
        for bit in 0..27 {
            if bit == CENTER_BIT || mask & (1 << bit) == 0 {
                continue;
            }
            let d = decode_offset(bit);
            let p = neighbor_probe(source, d);
            if (0..3).any(|a| p[a] < 0.0 || p[a] >= topology.graph.dimensions[a] as f64) {
                continue;
            }
            let ri = if let Some(i) = owner_at(&working, p) {
                i
            } else {
                if !options.allocate_missing_pages {
                    continue;
                }
                let id = if let Some(id) = free_stack.pop() {
                    id
                } else {
                    let id = next_id;
                    next_id = next_id
                        .checked_add(1)
                        .ok_or(ResolutionError3d::IdOverflow)?;
                    id
                };
                let coordinate = std::array::from_fn(|a| {
                    source.coordinate[a]
                        + if d[a] < 0 {
                            -1
                        } else if d[a] > 0 {
                            span(source)
                        } else {
                            0
                        }
                });
                let key = next_key;
                next_key = next_key
                    .checked_add(1)
                    .ok_or(ResolutionError3d::IdOverflow)?;
                working.push(BrickSeed {
                    id,
                    key,
                    coordinate,
                    span_bricks: 1,
                    resolution: 8,
                    active: true,
                    density: vec![],
                    gamma: vec![],
                    refinement_region_scale: None,
                });
                claimed.push(id);
                allocated.insert(key);
                working.len() - 1
            };
            if working[ri].key != source.key {
                demand.insert(working[ri].key);
            }
        }
    }
    Ok((working, demand, claimed, allocated))
}

pub fn plan_projected_transport_support_3d(
    topology: &CompiledTopology<3>,
    fields: &Fields,
    dt: f64,
    cell_size: f64,
    options: &ResolutionPolicyOptions3d,
) -> Result<ProjectedTransportSupportDecision3d, ResolutionError3d> {
    validate_inputs(topology, fields, dt, cell_size)?;
    let ts = thresholds(&options.policy, dt, cell_size);
    let mut measurements = BTreeMap::new();
    for i in 0..topology.bricks.len() {
        measurements.insert(
            topology.bricks[i].seed.key,
            measure(topology, fields, i, None, &options.policy, false, dt, ts),
        );
    }
    let accepted: Vec<_> = topology.bricks.iter().map(|b| b.seed.clone()).collect();
    let (working, demand, claimed, _) =
        allocate_frontier(topology, &accepted, &measurements, options)?;
    let mut targets: BTreeMap<_, _> = working.iter().map(|b| (b.key, b.resolution)).collect();
    let active: BTreeMap<_, _> = working
        .iter()
        .map(|b| (b.key, b.active || demand.contains(&b.key)))
        .collect();
    for b in &working {
        if demand.contains(&b.key) {
            targets.insert(b.key, b.resolution.max(8));
        }
    }
    close_two_to_one(&working, &active, &mut targets, &[]);
    let mut candidate = working.clone();
    for b in &mut candidate {
        b.resolution = targets[&b.key];
        b.active = active[&b.key];
    }
    let leaves = candidate.iter().filter(|b| b.active).count();
    let cells: usize = candidate
        .iter()
        .filter(|b| b.active)
        .map(|b| (b.resolution as usize).pow(3))
        .sum();
    let mut faults = 0;
    if options.maximum_leaves.is_some_and(|v| leaves > v) {
        faults |= resolution_fault::LEAF_CAPACITY;
    }
    if options.maximum_cells.is_some_and(|v| cells > v) {
        faults |= resolution_fault::CELL_CAPACITY;
    }
    Ok(ProjectedTransportSupportDecision3d {
        candidate_bricks: if faults == 0 { candidate } else { accepted },
        demanded_brick_keys: demand,
        claimed_leaf_ids: claimed,
        fault_bits: faults,
    })
}

pub fn plan_resolution_3d(
    topology: &CompiledTopology<3>,
    fields: &Fields,
    previous: &ResolutionPolicyState3d,
    dt: f64,
    cell_size: f64,
    options: &ResolutionPolicyOptions3d,
) -> Result<ResolutionPolicyDecision3d, ResolutionError3d> {
    validate_inputs(topology, fields, dt, cell_size)?;
    let policy = &options.policy;
    let accepted_steps = previous.accepted_steps + 1;
    let epoch =
        policy.topology_cadence_steps != 0 && accepted_steps % policy.topology_cadence_steps == 0;
    let ts = thresholds(policy, dt, cell_size);
    let accepted: Vec<_> = topology.bricks.iter().map(|b| b.seed.clone()).collect();
    for b in &accepted {
        valid_resolution(b.resolution)?;
    }
    let mut measurements = BTreeMap::new();
    for i in 0..accepted.len() {
        measurements.insert(
            accepted[i].key,
            measure(
                topology,
                fields,
                i,
                previous.history.get(&accepted[i].key),
                policy,
                epoch,
                dt,
                ts,
            ),
        );
    }
    let (working, material_demand, claimed, allocated) =
        allocate_frontier(topology, &accepted, &measurements, options)?;
    for b in working.iter().filter(|b| allocated.contains(&b.key)) {
        measurements.insert(
            b.key,
            Measurement {
                history: BrickActivityHistory3d {
                    reasons: 0,
                    last_transition_step: accepted_steps,
                    ..Default::default()
                },
                surface: false,
                thin: false,
                occupied: false,
                deeply_enclosed: false,
                curvature_floor: 1,
            },
        );
    }
    let mut targets = BTreeMap::new();
    let mut active = BTreeMap::new();
    let mut plan_reasons = BTreeMap::new();
    for (bi, brick) in accepted.iter().enumerate() {
        let m = &measurements[&brick.key];
        let current = brick.resolution;
        let frozen = policy.freeze_topology || options.frozen_brick_keys.contains(&brick.key);
        let mut requested = current;
        let mut reason = 32;
        if !brick.active || frozen {
            active.insert(brick.key, brick.active);
            targets.insert(brick.key, current);
            plan_reasons.insert(brick.key, if frozen { 32 } else { 128 });
            continue;
        }
        let measured = velocity_floor(m.history.velocity_travel, ts, policy.activity_signals);
        let static_floor = *options
            .static_boundary_floor_by_brick
            .get(&brick.key)
            .unwrap_or(&1);
        let moving_floor = if options.moving_rigid_bodies
            && m.history.reasons & activity_reason::CUT_BOUNDARY != 0
        {
            4
        } else {
            1
        };
        let enclosed = policy.activity_signals && m.deeply_enclosed && measured == 1;
        let surface = m.surface && !enclosed;
        let slow_surface = surface && !m.thin && measured == 1;
        let touches_liquid = face_neighbors(&accepted, bi)
            .into_iter()
            .any(|j| accepted[j].active && measurements[&accepted[j].key].occupied);
        let injection = options.injection_demanded_brick_keys.contains(&brick.key);
        let page_demand = injection
            || (touches_liquid
                && (!m.occupied
                    || (m.history.mean_density as f64) < policy.surface_density_minimum));
        let mut required = if enclosed {
            static_floor.max(moving_floor)
        } else {
            measured
                .max(if surface { (current / 2).max(1) } else { 1 })
                .max(if m.thin || page_demand { 8 } else { 1 })
                .max(static_floor)
                .max(moving_floor)
        };
        let mut proof_epochs = m.history.proof_epochs;
        if policy.coarse_first {
            let frontier = (0..27).filter(|&bit| bit != CENTER_BIT).any(|bit| {
                let mask = m.history.support_mask | m.history.swept_support_mask;
                mask & (1 << bit) != 0
                    && owner_at(&accepted, neighbor_probe(brick, decode_offset(bit)))
                        .is_none_or(|j| !accepted[j].active)
            });
            let moving_frontier = (frontier || page_demand) && measured > 1;
            let incoming = if m
                .curvature_floor
                .max(static_floor)
                .max(moving_floor)
                .max(measured)
                < 8
                && !m.thin
                && !injection
                && (surface || page_demand)
            {
                current.min(incoming_floor(topology, bi, &measurements, policy))
            } else {
                1
            };
            required = m
                .curvature_floor
                .max(static_floor)
                .max(moving_floor)
                .max(measured)
                .max(incoming)
                .max(if m.thin || injection || moving_frontier {
                    8
                } else {
                    1
                });
        }
        if policy.coarse_first {
            if required > current {
                requested = required;
                reason = if page_demand {
                    2
                } else if m.thin {
                    256
                } else {
                    4
                };
                proof_epochs = 0;
            } else if required < current {
                let next = (current / 2).max(1);
                let receipt_fresh = !surface
                    || (policy.surface_coarsening_enabled
                        && m.history
                            .surface_proof
                            .as_ref()
                            .and_then(|proof| proof.generation_by_target_resolution.get(&next))
                            .copied()
                            == Some(topology.graph.topology_generation));
                if receipt_fresh && epoch {
                    proof_epochs = proof_epochs.saturating_add(1);
                    if proof_epochs >= policy.surface_quiet_epochs {
                        requested = required.max(next);
                        reason = 16;
                    }
                } else if !receipt_fresh {
                    proof_epochs = 0;
                }
            } else {
                proof_epochs = 0;
            }
        } else if required > current
            || (!surface
                && !enclosed
                && !slow_surface
                && m.history.score_byte >= clamp_byte(policy.emergency_score))
        {
            let urgent = m.thin || page_demand;
            requested = if urgent {
                required
            } else {
                required.max(2 * current).min(8)
            };
            reason = if page_demand || m.history.reasons & activity_reason::PREDICTED_FACE != 0 {
                2
            } else if m.thin {
                256
            } else if measured > current {
                64
            } else {
                4
            };
        } else if epoch {
            if !enclosed && !slow_surface && m.history.hot_epochs >= policy.promote_epochs {
                requested = (2 * current).min(8);
                reason = 8;
            } else if current > required
                && (enclosed || slow_surface || m.history.quiet_epochs >= policy.demote_epochs)
                && m.history.reasons & activity_reason::FINE_DETAIL == 0
            {
                requested = if enclosed {
                    required
                } else {
                    required.max(current / 2)
                };
                reason = if enclosed { 2048 } else { 16 };
            }
        }
        measurements
            .get_mut(&brick.key)
            .unwrap()
            .history
            .proof_epochs = proof_epochs;
        requested = apply_regions(requested, brick, &options.refinement_regions);
        targets.insert(brick.key, requested);
        active.insert(brick.key, true);
        plan_reasons.insert(brick.key, reason);
    }
    for b in working.iter().filter(|b| allocated.contains(&b.key)) {
        targets.insert(b.key, 8);
        active.insert(b.key, true);
        plan_reasons.insert(b.key, 0x8000_0001);
    }
    for b in accepted.iter().filter(|b| !b.active) {
        if material_demand.contains(&b.key) {
            active.insert(b.key, true);
            targets.insert(b.key, apply_regions(8, b, &options.refinement_regions));
            plan_reasons.insert(b.key, 0x8000_0001);
        }
    }
    for (bi, b) in accepted.iter().enumerate().filter(|(_, b)| b.active) {
        let m = &measurements[&b.key];
        let exact_empty = topology.bricks[bi]
            .cell_range
            .clone()
            .all(|id| fields.density[id as usize] == 0.0);
        let moving =
            options.moving_rigid_bodies && m.history.reasons & activity_reason::CUT_BOUNDARY != 0;
        if exact_empty
            && !m.occupied
            && !material_demand.contains(&b.key)
            && !moving
            && !options.injection_demanded_brick_keys.contains(&b.key)
        {
            active.insert(b.key, false);
            plan_reasons.insert(b.key, 0x8000_0000);
        }
    }
    close_region_caps(&working, &mut targets, &options.refinement_regions);
    close_two_to_one(&working, &active, &mut targets, &options.refinement_regions);
    let changed: BTreeSet<_> = (0..working.len())
        .filter(|&i| {
            allocated.contains(&working[i].key)
                || targets[&working[i].key] != working[i].resolution
                || active[&working[i].key] != working[i].active
        })
        .collect();
    let urgent: BTreeSet<_> = changed
        .iter()
        .copied()
        .filter(|&i| {
            targets[&working[i].key] > working[i].resolution
                || active[&working[i].key] != working[i].active
                || region_bounds(&working[i], &options.refinement_regions).0 < working[i].resolution
        })
        .collect();
    let ordinary: BTreeSet<_> = changed
        .difference(&urgent)
        .copied()
        .filter(|&i| targets[&working[i].key] < working[i].resolution)
        .collect();
    let mut order: Vec<_> = (0..working.len()).collect();
    order.sort_by_key(|&i| working[i].key);
    let rotate = if order.is_empty() {
        0
    } else {
        previous.scheduling_cursor % order.len()
    };
    order.rotate_left(rotate);
    let budget = policy.prepare_bricks_per_frame;
    let credit_budget = (budget + 1).min(previous.scheduling_credits + budget);
    let admitted: BTreeSet<_> = order
        .iter()
        .copied()
        .filter(|i| ordinary.contains(i))
        .take(credit_budget)
        .collect();
    for &i in ordinary.difference(&admitted) {
        targets.insert(working[i].key, working[i].resolution);
    }
    let mut candidate = working.clone();
    for b in &mut candidate {
        b.resolution = targets[&b.key];
        b.active = active[&b.key];
    }
    let leaves = candidate.iter().filter(|b| b.active).count();
    let cells: usize = candidate
        .iter()
        .filter(|b| b.active)
        .map(|b| (b.resolution as usize).pow(3))
        .sum();
    let mut faults = 0;
    for i in 0..candidate.len() {
        for j in face_neighbors(&candidate, i) {
            if j > i {
                let a = width(&candidate[i], candidate[i].resolution);
                let b = width(&candidate[j], candidate[j].resolution);
                if a.max(b) > 2.0 * a.min(b) {
                    faults |= resolution_fault::TWO_TO_ONE;
                }
            }
        }
    }
    if options.maximum_leaves.is_some_and(|v| leaves > v) {
        faults |= resolution_fault::LEAF_CAPACITY;
    }
    if options.maximum_cells.is_some_and(|v| cells > v) {
        faults |= resolution_fault::CELL_CAPACITY;
    }
    if faults != 0 {
        candidate = accepted.clone();
    }
    let published = faults == 0
        && (candidate.len() != accepted.len()
            || candidate.iter().any(|b| {
                accepted
                    .iter()
                    .find(|a| a.key == b.key)
                    .is_none_or(|a| a.active != b.active || a.resolution != b.resolution)
            }));
    let mut history = BTreeMap::new();
    let mut records = Vec::new();
    let (mut promoted, mut demoted, mut activated, mut retired, mut maximum_score) =
        (0, 0, 0, 0, 0);
    for b in &working {
        let scheduled = candidate.iter().find(|c| c.key == b.key).unwrap_or(b);
        let mut h = measurements[&b.key].history.clone();
        let was_active = !allocated.contains(&b.key) && b.active;
        let did_change = allocated.contains(&b.key)
            || scheduled.resolution != b.resolution
            || scheduled.active != b.active;
        if did_change {
            h.hot_epochs = 0;
            h.quiet_epochs = 0;
            h.proof_epochs = 0;
            h.last_transition_step = accepted_steps;
        }
        maximum_score = maximum_score.max(h.score_byte);
        promoted += usize::from(scheduled.resolution > b.resolution);
        demoted += usize::from(scheduled.resolution < b.resolution);
        activated += usize::from(!was_active && scheduled.active);
        retired += usize::from(was_active && !scheduled.active);
        records.push(BrickResolutionReceipt3d {
            brick_key: b.key,
            accepted_resolution: b.resolution,
            requested_resolution: targets[&b.key],
            scheduled_resolution: scheduled.resolution,
            accepted_active: was_active,
            candidate_active: scheduled.active,
            score_byte: h.score_byte,
            reasons: h.reasons,
            plan_reasons: plan_reasons[&b.key],
            support_mask: h.support_mask,
            swept_support_mask: h.swept_support_mask,
            fault_bits: faults,
        });
        history.insert(b.key, h);
    }
    let ordinary_len = ordinary.len();
    let order_len = order.len();
    Ok(ResolutionPolicyDecision3d {
        candidate_bricks: candidate,
        state: ResolutionPolicyState3d {
            accepted_steps,
            accepted_generation: topology.graph.topology_generation,
            scheduling_cursor: if order_len == 0 {
                0
            } else {
                (previous.scheduling_cursor + budget.min(ordinary_len)) % order_len
            },
            scheduling_credits: credit_budget - admitted.len(),
            history,
        },
        receipt: ResolutionPolicyReceipt3d {
            topology_epoch: epoch,
            accepted_generation: topology.graph.topology_generation,
            candidate_generation: topology.graph.topology_generation + u32::from(published),
            measured_brick_count: accepted.len(),
            surface_brick_count: measurements.values().filter(|m| m.surface).count(),
            occupied_brick_count: measurements.values().filter(|m| m.occupied).count(),
            activated_brick_count: activated,
            allocated_brick_count: allocated.len(),
            claimed_leaf_ids: claimed,
            retired_brick_count: retired,
            promoted_brick_count: promoted,
            demoted_brick_count: demoted,
            deferred_demotion_count: ordinary_len - admitted.len(),
            maximum_score_byte: maximum_score,
            fault_bits: faults,
            bricks: records,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::BoundaryMode;
    use crate::topology::{compile_topology, TopologySeed};

    fn brick(id: u32, coordinate: [i32; 3], resolution: u8, active: bool) -> BrickSeed {
        BrickSeed {
            id,
            key: id,
            coordinate,
            span_bricks: 1,
            resolution,
            active,
            density: vec![],
            gamma: vec![],
            refinement_region_scale: None,
        }
    }
    fn setup(bricks: Vec<BrickSeed>, dimensions: [u32; 3]) -> (CompiledTopology<3>, Fields) {
        let topology = compile_topology(TopologySeed {
            dimensions,
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks,
        })
        .unwrap();
        let n = topology.graph.cells.len();
        (
            topology,
            Fields {
                density: vec![0.0; n],
                capacity: vec![1.0; n],
                cell_velocity: vec![0.0; 3 * n],
                face_velocity: vec![],
                acceleration_fine: [0.0; 3],
                interface_normal: vec![0.0; 3 * n],
                ..Fields::default()
            },
        )
    }

    #[test]
    fn xyz_sweep_demands_corner_support() {
        let (topology, mut fields) = setup(vec![brick(0, [1, 1, 1], 1, true)], [24, 24, 24]);
        fields.density[0] = 1.0;
        fields.cell_velocity[..3].copy_from_slice(&[20.0, 20.0, 20.0]);
        fields.acceleration_fine = [-4_000.0; 3];
        fields.cell_velocity[..3].copy_from_slice(&[20.0, 20.0, 20.0]);
        let decision = plan_projected_transport_support_3d(
            &topology,
            &fields,
            1.0,
            1.0,
            &ResolutionPolicyOptions3d::default(),
        )
        .unwrap();
        assert!(decision
            .candidate_bricks
            .iter()
            .any(|b| b.coordinate == [2, 2, 2]));
        assert!(decision.claimed_leaf_ids.len() >= 7);
    }

    #[test]
    fn wet_sparse_page_publishes_all_twenty_six_support_directions() {
        let mut bricks = Vec::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    let key = (x + 3 * y + 9 * z) as u32;
                    bricks.push(brick(key, [x, y, z], 1, true));
                }
            }
        }
        let (topology, mut fields) = setup(bricks, [24, 24, 24]);
        let central = topology.bricks.iter().find(|b| b.seed.key == 13).unwrap();
        fields.density[central.cell_range.start as usize] = 1.0;
        let state = initialize_resolution_policy_3d(&topology);
        let decision = plan_resolution_3d(
            &topology,
            &fields,
            &state,
            0.01,
            1.0,
            &ResolutionPolicyOptions3d::default(),
        )
        .unwrap();
        let expected = SUPPORT_MASK_3D & !(1 << CENTER_BIT);
        let receipt = decision
            .receipt
            .bricks
            .iter()
            .find(|b| b.brick_key == 13)
            .unwrap();
        assert_eq!(receipt.support_mask, expected);
        assert_eq!(receipt.swept_support_mask, 1 << CENTER_BIT);
    }

    #[test]
    fn opposing_sources_elect_one_frontier_page_owner() {
        let (topology, mut fields) = setup(
            vec![brick(7, [0, 0, 0], 1, true), brick(3, [2, 0, 0], 1, true)],
            [24, 8, 8],
        );
        fields.density.fill(1.0);
        for record in &topology.bricks {
            let velocity = if record.seed.coordinate[0] == 0 {
                20.0
            } else {
                -20.0
            };
            fields.cell_velocity[3 * record.cell_range.start as usize] = velocity;
        }
        let decision = plan_projected_transport_support_3d(
            &topology,
            &fields,
            1.0,
            1.0,
            &ResolutionPolicyOptions3d::default(),
        )
        .unwrap();
        let middle: Vec<_> = decision
            .candidate_bricks
            .iter()
            .filter(|b| b.coordinate == [1, 0, 0])
            .collect();
        assert_eq!(middle.len(), 1);
        assert_eq!(decision.claimed_leaf_ids.len(), 1);
    }

    #[test]
    fn closure_refines_coarse_face_neighbor() {
        let (topology, mut fields) = setup(
            vec![
                brick(0, [0, 0, 0], 4, true),
                brick(1, [1, 0, 0], 2, true),
                brick(2, [2, 0, 0], 1, true),
            ],
            [24, 8, 8],
        );
        for id in topology.bricks[0].cell_range.clone() {
            fields.density[id as usize] = 1.0;
            fields.cell_velocity[3 * id as usize] = 20.0;
        }
        let d = plan_projected_transport_support_3d(
            &topology,
            &fields,
            1.0,
            1.0,
            &ResolutionPolicyOptions3d::default(),
        )
        .unwrap();
        assert_eq!(
            d.candidate_bricks
                .iter()
                .find(|b| b.key == 1)
                .unwrap()
                .resolution,
            8
        );
        assert_eq!(
            d.candidate_bricks
                .iter()
                .find(|b| b.key == 2)
                .unwrap()
                .resolution,
            4
        );
    }

    #[test]
    fn moving_cut_page_cannot_retire() {
        let (topology, mut fields) = setup(vec![brick(0, [0, 0, 0], 1, true)], [8, 8, 8]);
        fields.capacity[0] = 0.5;
        let mut options = ResolutionPolicyOptions3d::default();
        options.moving_rigid_bodies = true;
        let state = initialize_resolution_policy_3d(&topology);
        let d = plan_resolution_3d(&topology, &fields, &state, 0.01, 1.0, &options).unwrap();
        assert!(d.candidate_bricks[0].active);
        assert_eq!(d.candidate_bricks[0].resolution, 4);
    }

    #[test]
    fn exact_zero_retires_but_residue_remains() {
        let (topology, fields) = setup(vec![brick(0, [0, 0, 0], 1, true)], [8, 8, 8]);
        let state = initialize_resolution_policy_3d(&topology);
        let zero = plan_resolution_3d(
            &topology,
            &fields,
            &state,
            0.01,
            1.0,
            &ResolutionPolicyOptions3d::default(),
        )
        .unwrap();
        assert!(!zero.candidate_bricks[0].active);
        let mut residue = fields;
        residue.density[0] = f32::MIN_POSITIVE;
        let nonzero = plan_resolution_3d(
            &topology,
            &residue,
            &state,
            0.01,
            1.0,
            &ResolutionPolicyOptions3d::default(),
        )
        .unwrap();
        assert!(nonzero.candidate_bricks[0].active);
    }

    #[test]
    fn inclusive_injection_bounds_allocate_edge_page_without_a_donor() {
        let (topology, fields) = setup(vec![brick(0, [0, 0, 0], 1, true)], [24, 8, 8]);
        let mut options = ResolutionPolicyOptions3d::default();
        options.injection_bounds_fine = Some(([15.0, 1.0, 1.0], [16.0, 2.0, 2.0]));
        let d =
            plan_projected_transport_support_3d(&topology, &fields, 0.01, 1.0, &options).unwrap();
        assert!(d
            .candidate_bricks
            .iter()
            .any(|b| b.coordinate == [1, 0, 0] && b.active));
        assert!(d
            .candidate_bricks
            .iter()
            .any(|b| b.coordinate == [2, 0, 0] && b.active));
    }

    #[test]
    fn capacity_fault_keeps_the_accepted_catalogue() {
        let (topology, mut fields) = setup(vec![brick(0, [1, 1, 1], 1, true)], [24, 24, 24]);
        fields.density[0] = 1.0;
        fields.cell_velocity[..3].copy_from_slice(&[20.0, 20.0, 20.0]);
        let mut options = ResolutionPolicyOptions3d::default();
        options.maximum_leaves = Some(1);
        let decision =
            plan_projected_transport_support_3d(&topology, &fields, 1.0, 1.0, &options).unwrap();
        assert_ne!(decision.fault_bits & resolution_fault::LEAF_CAPACITY, 0);
        assert_eq!(decision.candidate_bricks.len(), 1);
        assert_eq!(decision.candidate_bricks[0].key, 0);
    }

    #[test]
    fn three_axis_region_bounds_force_the_requested_rung() {
        let (topology, mut fields) = setup(vec![brick(0, [0, 0, 0], 1, true)], [8, 8, 8]);
        fields.density[0] = 1.0;
        let mut options = ResolutionPolicyOptions3d::default();
        options.refinement_regions.push(ResolutionRegion3d {
            minimum_fine: [0.0, 0.0, 0.0],
            maximum_fine: [8.0, 8.0, 8.0],
            minimum_cell_width: 1,
            maximum_cell_width: Some(1),
        });
        let decision = plan_resolution_3d(
            &topology,
            &fields,
            &initialize_resolution_policy_3d(&topology),
            0.01,
            1.0,
            &options,
        )
        .unwrap();
        assert_eq!(decision.candidate_bricks[0].resolution, 8);
    }

    fn x_half_space(topology: &CompiledTopology<3>, fields: &mut Fields, curved: bool) {
        let r = topology.bricks[0].seed.resolution as usize;
        for local in 0..r.pow(3) {
            let x = local % r;
            let y = (local / r) % r;
            if x < r / 2 {
                fields.density[local] = 1.0;
            }
            let normal = if curved && y >= r / 2 {
                [0.0, 1.0, 0.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            fields.interface_normal[3 * local..3 * local + 3].copy_from_slice(&normal);
        }
    }

    #[test]
    fn flat_surface_demotes_only_with_a_fresh_accepted_proof() {
        let (topology, mut fields) = setup(vec![brick(0, [0, 0, 0], 4, true)], [8, 8, 8]);
        x_half_space(&topology, &mut fields, false);
        let mut options = ResolutionPolicyOptions3d::default();
        options.policy.surface_quiet_epochs = 2;
        let mut state = initialize_resolution_policy_3d(&topology);
        state.history.get_mut(&0).unwrap().proof_epochs = 1;
        state.history.get_mut(&0).unwrap().surface_proof = Some(SurfaceProofState {
            generation_by_target_resolution: BTreeMap::from([(2, 1)]),
        });
        let proved = plan_resolution_3d(&topology, &fields, &state, 0.01, 1.0, &options).unwrap();
        assert_eq!(proved.candidate_bricks[0].resolution, 2);

        state.history.get_mut(&0).unwrap().surface_proof = None;
        let unproved = plan_resolution_3d(&topology, &fields, &state, 0.01, 1.0, &options).unwrap();
        assert_eq!(unproved.candidate_bricks[0].resolution, 4);
    }

    #[test]
    fn curved_xyz_normals_raise_the_geometric_floor() {
        let (topology, mut fields) = setup(vec![brick(0, [0, 0, 0], 4, true)], [8, 8, 8]);
        x_half_space(&topology, &mut fields, true);
        let decision = plan_resolution_3d(
            &topology,
            &fields,
            &initialize_resolution_policy_3d(&topology),
            0.01,
            1.0,
            &ResolutionPolicyOptions3d::default(),
        )
        .unwrap();
        assert_eq!(decision.candidate_bricks[0].resolution, 8);
    }

    #[test]
    fn captured_gpu_policy_abi_matches_the_rust_3d_contract() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/resolution3d-gpu-policy.json"
        ))
        .unwrap();
        let activity = &fixture["activityMask"];
        assert_eq!(activity["activityFlagValues"], 256);
        assert_eq!(activity["directionValues"], 32);
        assert_eq!(activity["directionBits"], 27);
        assert_eq!(activity["fullSupportMask"], SUPPORT_MASK_3D);
        let mut reconstructed = 0_u32;
        for bit in 0..27 {
            let d = decode_offset(bit);
            assert_eq!(offset_bit(d[0], d[1], d[2]), bit as u32);
            reconstructed |= 1 << bit;
        }
        assert_eq!(reconstructed, SUPPORT_MASK_3D);

        let curvature = &fixture["coarseFirstCurvature"];
        assert_eq!(curvature["cellWidths"], serde_json::json!([8, 4, 2, 1]));
        assert_eq!(curvature["axes"], 3);
        assert_eq!(curvature["cutVariants"], 2);
        assert_eq!(curvature["maximumRestrictionError"], 0.0);
        assert!(curvature["maximumNormalError"].as_f64().unwrap() <= f32::EPSILON as f64);
    }

    #[test]
    fn continuous_inflow_uses_the_gpu_two_step_receiver_box() {
        let bounds =
            continuous_inflow_demand_bounds_3d([8.0, 8.0, 8.0], [4.0, -2.0, 1.0], 2.0, 0.5)
                .unwrap()
                .unwrap();
        assert_eq!(bounds, ([5.0, 3.0, 5.0], [15.0, 11.0, 12.0]));

        let (topology, fields) = setup(vec![brick(0, [0, 0, 0], 1, false)], [24, 16, 16]);
        let mut options = ResolutionPolicyOptions3d::default();
        options.source_demand_bounds_fine.push(bounds);
        let decision =
            plan_projected_transport_support_3d(&topology, &fields, 0.5, 1.0, &options).unwrap();
        assert!(decision
            .candidate_bricks
            .iter()
            .any(|b| b.coordinate == [1, 0, 0] && b.active));
        assert!(decision
            .candidate_bricks
            .iter()
            .any(|b| b.coordinate == [1, 1, 1] && b.active));
    }

    #[test]
    fn static_solid_evidence_accepts_planes_and_refines_unresolved_voxels() {
        let (mut topology, _) = setup(vec![brick(0, [0, 0, 0], 1, true)], [8, 8, 8]);
        topology.graph.solid_voxel_fraction = (0..8)
            .flat_map(|_| (0..8).flat_map(|_| (0..8).map(|x| f32::from(x < 4))))
            .collect();
        assert_eq!(static_boundary_floors_3d(&topology, 0.01).unwrap()[&0], 2);

        topology.graph.solid_voxel_fraction.fill(0.0);
        topology.graph.solid_voxel_fraction[1 + 8 * (1 + 8)] = 1.0;
        assert_eq!(static_boundary_floors_3d(&topology, 0.01).unwrap()[&0], 8);
    }
}
