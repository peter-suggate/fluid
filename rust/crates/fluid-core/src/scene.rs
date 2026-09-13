//! Physical scene ingestion and authoritative initial-state construction.

use serde::{Deserialize, Serialize};

use crate::geometry::{face_geometry, open_fraction, BoundaryMode, Solid, SolidShape};
use crate::topology::{compile_topology, BrickSeed, CompiledTopology, TopologyError, TopologySeed};
use crate::types::Fields;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitialLiquid {
    pub shape: SolidShape,
    #[serde(default = "one")]
    pub density: f32,
    #[serde(default = "one")]
    pub gamma: f32,
    #[serde(default)]
    pub velocity: [f32; 3],
}
fn one() -> f32 {
    1.0
}

/// Serialized physical input. The frontend may author this document, but all
/// sampling and numerical initialization occur here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneDescription {
    pub schema_version: u32,
    pub dimension: u8,
    pub dimensions: [u32; 3],
    pub cell_size_m: f32,
    #[serde(default)]
    pub origin_m: [f32; 3],
    pub dt_s: f32,
    pub density_kg_m3: f32,
    #[serde(default)]
    pub gravity_m_s2: [f32; 3],
    pub boundaries: [BoundaryMode; 6],
    pub bricks: Vec<BrickSeed>,
    #[serde(default)]
    pub solids: Vec<Solid>,
    #[serde(default)]
    pub liquids: Vec<InitialLiquid>,
    /// Optional finest-lattice scalar planes for exact SliceSceneSeed ingestion.
    #[serde(default)]
    pub raster_capacity: Vec<f32>,
    #[serde(default)]
    pub raster_density: Vec<f32>,
    #[serde(default)]
    pub raster_gamma: Vec<f32>,
    /// Finest-lattice cell-centered velocity, flattened by dimension.
    #[serde(default)]
    pub raster_velocity: Vec<f32>,
    #[serde(default)]
    pub velocity_x: Vec<f32>,
    #[serde(default)]
    pub velocity_y: Vec<f32>,
    #[serde(default)]
    pub aperture_x: Vec<f32>,
    #[serde(default)]
    pub aperture_y: Vec<f32>,
    #[serde(default)]
    pub solid_velocity_x: Vec<f32>,
    #[serde(default)]
    pub solid_velocity_y: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct SceneState<const D: usize> {
    pub description: SceneDescription,
    pub topology: CompiledTopology<D>,
    pub fields: Fields,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SceneError {
    Invalid(&'static str),
    Topology(TopologyError),
}
impl From<TopologyError> for SceneError {
    fn from(e: TopologyError) -> Self {
        Self::Topology(e)
    }
}
impl std::fmt::Display for SceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SceneError {}

pub fn compile_scene<const D: usize>(
    description: SceneDescription,
) -> Result<SceneState<D>, SceneError> {
    validate_scene::<D>(&description)?;
    let mut topology = compile_topology::<D>(TopologySeed {
        dimensions: description.dimensions,
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: description.boundaries,
        bricks: description.bricks.clone(),
    })?;
    let cell_count = topology.graph.cells.len();
    let row_count = topology.graph.rows.len();
    let mut density = vec![0.0_f32; cell_count];
    let mut gamma = vec![1.0_f32; cell_count];
    let mut capacity = vec![1.0_f32; cell_count];
    let mut cell_velocity = vec![0.0; D * cell_count];
    for cell in &topology.graph.cells {
        let i = cell.id as usize;
        capacity[i] = capacity_for_cell::<D>(&description, cell.minimum, cell.maximum);
        let capacity_per_brick = if D == 2 { 64 } else { 512 };
        let local = cell.stable_id.unwrap_or(0) as usize % capacity_per_brick;
        let brick = topology
            .bricks
            .iter()
            .find(|b| Some(b.seed.key) == cell.brick_key)
            .unwrap();
        if let Some(&value) = brick.seed.density.get(local) {
            density[i] = value;
        } else if !description.raster_density.is_empty() {
            density[i] = average_raster::<D>(
                &description.raster_density,
                description.dimensions,
                cell.minimum,
                cell.maximum,
            );
        }
        if let Some(&value) = brick.seed.gamma.get(local) {
            gamma[i] = value;
        } else if !description.raster_gamma.is_empty() {
            gamma[i] = average_raster::<D>(
                &description.raster_gamma,
                description.dimensions,
                cell.minimum,
                cell.maximum,
            );
        }
        if !description.raster_velocity.is_empty() {
            for axis in 0..D {
                cell_velocity[D * i + axis] = average_raster_component::<D>(
                    &description.raster_velocity,
                    description.dimensions,
                    cell.minimum,
                    cell.maximum,
                    axis,
                );
            }
        }
        for liquid in &description.liquids {
            let coverage = shape_coverage::<D>(&liquid.shape, cell.minimum, cell.maximum, 4);
            if coverage > 0.0 {
                density[i] = density[i].max(coverage * liquid.density * capacity[i]);
                gamma[i] = liquid.gamma;
                for axis in 0..D {
                    cell_velocity[D * i + axis] = liquid.velocity[axis] / description.cell_size_m;
                }
            }
        }
    }
    let mut face_velocity = vec![0.0; row_count];
    for row in &mut topology.graph.rows {
        let analytic_geometry = || {
            face_geometry::<D>(
                row.axis as usize,
                row.center,
                face_widths::<D>(row.axis as usize, row.measure),
                &description.solids,
                2,
            )
        };
        let (analytic_aperture, analytic_solid_velocity) = analytic_geometry();
        let mut aperture = if D == 2 && has_slice_apertures(&description) {
            slice_face_average(
                &description,
                row.axis as usize,
                row.center,
                row.measure,
                SliceFacePlane::Aperture,
            )
        } else {
            analytic_aperture
        };
        if row.kind == crate::types::RowKind::ClosedWorld
            && !(D == 2 && has_slice_apertures(&description))
        {
            aperture = 0.0;
        }
        let solid_velocity_m_s = if D == 2 && has_slice_faces(&description) {
            slice_face_average(
                &description,
                row.axis as usize,
                row.center,
                row.measure,
                SliceFacePlane::SolidVelocity,
            )
        } else {
            analytic_solid_velocity
        };
        row.open_fraction = aperture;
        row.static_open_fraction = Some(aperture);
        row.open_fraction_before = Some(aperture);
        row.open_fraction_after = Some(aperture);
        if D == 2 && has_slice_apertures(&description) {
            row.dual_weight = row.static_dual_weight.unwrap_or(row.dual_weight) * aperture;
        }
        row.solid_velocity = solid_velocity_m_s / description.cell_size_m;
        let mut total = 0.0;
        let mut weight = 0.0;
        for term in &row.terms {
            let w = term.coefficient.abs();
            total += w * cell_velocity[D * term.cell_id as usize + row.axis as usize];
            weight += w;
        }
        face_velocity[row.id as usize] = if D == 2 && has_slice_faces(&description) {
            slice_face_average(
                &description,
                row.axis as usize,
                row.center,
                row.measure,
                SliceFacePlane::Velocity,
            )
        } else if weight > 0.0 {
            total / weight
        } else {
            row.solid_velocity
        };
    }
    for face in &mut topology.graph.subfaces {
        let row = &topology.graph.rows[face.row_id as usize];
        face.aperture = row.open_fraction;
        face.solid_velocity = row.solid_velocity;
    }
    let acceleration_fine = description
        .gravity_m_s2
        .map(|v| v / description.cell_size_m);
    let fields = Fields {
        density,
        gamma,
        capacity: capacity.clone(),
        capacity_before: capacity.clone(),
        capacity_after: capacity,
        solid_motion_active: description
            .solids
            .iter()
            .any(|s| s.linear_velocity != [0.0; 3] || s.angular_velocity != [0.0; 3]),
        frame_dt: description.dt_s,
        acceleration_fine,
        capacity_rate: vec![0.0; cell_count],
        source_rate: vec![0.0; cell_count],
        inflow_coverage: vec![0.0; row_count],
        cell_velocity,
        face_velocity,
        subface_velocity_correction: vec![0.0; topology.graph.subfaces.len()],
        subface_compatibility_rate: Vec::new(),
        pressure: vec![0.0; cell_count],
        pressure_rhs: vec![0.0; cell_count],
        pressure_diagonal: vec![0.0; cell_count],
        pressure_member: vec![0; cell_count],
        pressure_row_member: vec![0; row_count],
        extension_depth: vec![0; cell_count],
        interface_normal: vec![0.0; D * cell_count],
        interface_offset: vec![0.0; cell_count],
        low_flux: vec![0.0; topology.graph.subfaces.len()],
        high_flux: vec![0.0; topology.graph.subfaces.len()],
        limited_flux: vec![0.0; topology.graph.subfaces.len()],
        characteristic_clearance: vec![0.0; row_count],
        fault: None,
    };
    Ok(SceneState {
        description,
        topology,
        fields,
    })
}

pub fn compile_scene_2d(description: SceneDescription) -> Result<SceneState<2>, SceneError> {
    compile_scene(description)
}
pub fn compile_scene_3d(description: SceneDescription) -> Result<SceneState<3>, SceneError> {
    compile_scene(description)
}

fn validate_scene<const D: usize>(s: &SceneDescription) -> Result<(), SceneError> {
    if s.schema_version != 1 {
        return Err(SceneError::Invalid("schema version"));
    }
    if s.dimension as usize != D || (D != 2 && D != 3) {
        return Err(SceneError::Invalid("dimension"));
    }
    if !(s.cell_size_m.is_finite()
        && s.cell_size_m > 0.0
        && s.dt_s.is_finite()
        && s.dt_s > 0.0
        && s.density_kg_m3.is_finite()
        && s.density_kg_m3 > 0.0)
    {
        return Err(SceneError::Invalid("physical units"));
    }
    let samples = sample_count::<D>(s.dimensions)?;
    for raster in [&s.raster_capacity, &s.raster_density, &s.raster_gamma] {
        if !raster.is_empty() && raster.len() != samples {
            return Err(SceneError::Invalid("raster length"));
        }
    }
    if !s.raster_velocity.is_empty() && s.raster_velocity.len() != D * samples {
        return Err(SceneError::Invalid("velocity raster length"));
    }
    if D == 2 {
        let nx = s.dimensions[0] as usize;
        let ny = s.dimensions[1] as usize;
        for (plane, n) in [
            (&s.velocity_x, (nx + 1) * ny),
            (&s.velocity_y, nx * (ny + 1)),
            (&s.aperture_x, (nx + 1) * ny),
            (&s.aperture_y, nx * (ny + 1)),
            (&s.solid_velocity_x, (nx + 1) * ny),
            (&s.solid_velocity_y, nx * (ny + 1)),
        ] {
            if !plane.is_empty() && plane.len() != n {
                return Err(SceneError::Invalid("staggered slice plane length"));
            }
        }
    }
    Ok(())
}

fn sample_count<const D: usize>(dims: [u32; 3]) -> Result<usize, SceneError> {
    (0..D).try_fold(1usize, |n, a| {
        n.checked_mul(dims[a] as usize)
            .ok_or(SceneError::Invalid("scene too large"))
    })
}
fn raster_index<const D: usize>(q: [i32; 3], dims: [u32; 3]) -> usize {
    q[0] as usize
        + dims[0] as usize
            * (q[1] as usize
                + if D == 3 {
                    dims[1] as usize * q[2] as usize
                } else {
                    0
                })
}
fn average_raster<const D: usize>(r: &[f32], dims: [u32; 3], min: [f32; 3], max: [f32; 3]) -> f32 {
    let mut sum = 0.0;
    let mut n = 0;
    for_points::<D>(min, max, |q| {
        sum += r[raster_index::<D>(q, dims)];
        n += 1
    });
    if n == 0 {
        0.0
    } else {
        sum / n as f32
    }
}
pub(crate) fn capacity_for_cell<const D: usize>(
    s: &SceneDescription,
    min: [f32; 3],
    max: [f32; 3],
) -> f32 {
    if s.raster_capacity.is_empty() {
        open_fraction::<D>(min, max, &s.solids, 4)
    } else {
        average_raster::<D>(&s.raster_capacity, s.dimensions, min, max)
    }
}
fn average_raster_component<const D: usize>(
    r: &[f32],
    dims: [u32; 3],
    min: [f32; 3],
    max: [f32; 3],
    axis: usize,
) -> f32 {
    let mut sum = 0.0;
    let mut n = 0;
    for_points::<D>(min, max, |q| {
        sum += r[D * raster_index::<D>(q, dims) + axis];
        n += 1
    });
    if n == 0 {
        0.0
    } else {
        sum / n as f32
    }
}
fn for_points<const D: usize>(min: [f32; 3], max: [f32; 3], mut f: impl FnMut([i32; 3])) {
    let lo = min.map(|v| v.floor() as i32);
    let hi = max.map(|v| v.ceil() as i32);
    for z in lo[2]..if D == 3 { hi[2] } else { lo[2] + 1 } {
        for y in lo[1]..hi[1] {
            for x in lo[0]..hi[0] {
                f([x, y, z]);
            }
        }
    }
}
fn shape_coverage<const D: usize>(
    shape: &SolidShape,
    min: [f32; 3],
    max: [f32; 3],
    samples: usize,
) -> f32 {
    let mut inside = 0;
    let count = samples.pow(D as u32);
    for linear in 0..count {
        let mut q = linear;
        let mut p = [0.0; 3];
        for a in 0..D {
            let lane = q % samples;
            q /= samples;
            p[a] = min[a] + (lane as f32 + 0.5) / samples as f32 * (max[a] - min[a]);
        }
        if D == 2 {
            p[2] = 0.5 * (min[2] + max[2]);
        }
        if shape.contains(p) {
            inside += 1;
        }
    }
    inside as f32 / count as f32
}
fn face_widths<const D: usize>(axis: usize, measure: f32) -> [f32; 3] {
    let mut w = [1.0; 3];
    if D == 2 {
        w[1 - axis] = measure;
    } else {
        let side = measure.sqrt();
        for (a, item) in w.iter_mut().enumerate().take(D) {
            if a != axis {
                *item = side;
            }
        }
    }
    w
}
fn has_slice_faces(s: &SceneDescription) -> bool {
    !s.velocity_x.is_empty() && !s.velocity_y.is_empty()
}
fn has_slice_apertures(s: &SceneDescription) -> bool {
    !s.aperture_x.is_empty() && !s.aperture_y.is_empty()
}
#[derive(Clone, Copy)]
enum SliceFacePlane {
    Velocity,
    Aperture,
    SolidVelocity,
}
fn slice_face_sample(s: &SceneDescription, axis: usize, q: [f32; 3], kind: SliceFacePlane) -> f32 {
    let nx = s.dimensions[0] as usize;
    let ny = s.dimensions[1] as usize;
    if axis == 0 {
        let x = (q[0].round() as isize).clamp(0, nx as isize) as usize;
        let cy = ny - 1 - (q[1].floor() as isize).clamp(0, ny as isize - 1) as usize;
        let i = cy * (nx + 1) + x;
        match kind {
            SliceFacePlane::Velocity => s.velocity_x[i] / s.cell_size_m,
            SliceFacePlane::Aperture => s.aperture_x.get(i).copied().unwrap_or(1.0),
            SliceFacePlane::SolidVelocity => s.solid_velocity_x.get(i).copied().unwrap_or(0.0),
        }
    } else {
        let x = (q[0].floor() as isize).clamp(0, nx as isize - 1) as usize;
        let cy = ny - (q[1].round() as isize).clamp(0, ny as isize) as usize;
        let i = cy * nx + x;
        match kind {
            SliceFacePlane::Velocity => -s.velocity_y[i] / s.cell_size_m,
            SliceFacePlane::Aperture => s.aperture_y.get(i).copied().unwrap_or(1.0),
            SliceFacePlane::SolidVelocity => {
                -s.solid_velocity_y.get(i).copied().unwrap_or(0.0) / s.cell_size_m
            }
        }
    }
}
fn slice_face_average(
    s: &SceneDescription,
    axis: usize,
    q: [f32; 3],
    length: f32,
    kind: SliceFacePlane,
) -> f32 {
    let tangent = 1 - axis;
    let lower = q[tangent] - 0.5 * length;
    let upper = q[tangent] + 0.5 * length;
    let mut sum = 0.0_f32;
    let mut weight = 0.0_f32;
    for cell in lower.floor() as i32..upper.ceil() as i32 {
        let overlap = (upper.min(cell as f32 + 1.0) - lower.max(cell as f32)).max(0.0);
        if overlap > 0.0 {
            let mut point = q;
            point[tangent] = cell as f32 + 0.5;
            sum = (sum + (overlap * slice_face_sample(s, axis, point, kind))) as f32;
            weight = (weight + overlap) as f32;
        }
    }
    if weight > 0.0 {
        (sum / weight) as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initializes_authored_liquid_and_closed_boundaries() {
        let scene = SceneDescription {
            schema_version: 1,
            dimension: 2,
            dimensions: [8, 8, 1],
            cell_size_m: 0.1,
            origin_m: [0.0; 3],
            dt_s: 1.0 / 60.0,
            density_kg_m3: 998.2,
            gravity_m_s2: [0.0, -9.81, 0.0],
            boundaries: [BoundaryMode::Closed; 6],
            bricks: vec![BrickSeed {
                id: 0,
                key: 0,
                coordinate: [0, 0, 0],
                span_bricks: 1,
                resolution: 2,
                active: true,
                density: vec![],
                gamma: vec![],
                refinement_region_scale: None,
            }],
            solids: vec![],
            liquids: vec![InitialLiquid {
                shape: SolidShape::Box {
                    center: [2.0, 4.0, 0.0],
                    half_extents: [2.0, 4.0, 1.0],
                },
                density: 1.0,
                gamma: 1.0,
                velocity: [0.0; 3],
            }],
            raster_capacity: vec![],
            raster_density: vec![],
            raster_gamma: vec![],
            raster_velocity: vec![],
            velocity_x: vec![0.0; 9 * 8],
            velocity_y: vec![0.0; 8 * 9],
            aperture_x: vec![],
            aperture_y: vec![],
            solid_velocity_x: vec![],
            solid_velocity_y: vec![],
        };
        let state = compile_scene_2d(scene).unwrap();
        assert!(state.fields.density.iter().any(|&v| v > 0.0));
        assert!(state
            .topology
            .graph
            .rows
            .iter()
            .filter(|r| r.kind == crate::types::RowKind::ClosedWorld)
            .all(|r| r.open_fraction == 0.0));
    }
}
