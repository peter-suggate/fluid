use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

pub const SCHEMA_VERSION: u32 = 1;
pub const PRESSURE_REDUCTION_LANES: usize = 64;

fn one() -> f32 {
    1.0
}
fn minus_one() -> i32 {
    -1
}

fn deserialize_coordinate<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<[f32; 3], D::Error> {
    let values = Vec::<f32>::deserialize(deserializer)?;
    match values.as_slice() {
        [x, y] => Ok([*x, *y, 0.0]),
        [x, y, z] => Ok([*x, *y, *z]),
        _ => Err(serde::de::Error::custom(
            "coordinate must contain 2 or 3 values",
        )),
    }
}

fn deserialize_widths<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<[f32; 3], D::Error> {
    let values = Vec::<f32>::deserialize(deserializer)?;
    match values.as_slice() {
        [x, y] => Ok([*x, *y, 1.0]),
        [x, y, z] => Ok([*x, *y, *z]),
        _ => Err(serde::de::Error::custom(
            "widths must contain 2 or 3 values",
        )),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RowKind {
    #[default]
    IntraBrick,
    BrickFace,
    MixedSeam,
    SparseAir,
    ClosedWorld,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cell {
    pub id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_id: Option<u32>,
    #[serde(deserialize_with = "deserialize_coordinate")]
    pub minimum: [f32; 3],
    #[serde(deserialize_with = "deserialize_coordinate")]
    pub maximum: [f32; 3],
    #[serde(deserialize_with = "deserialize_coordinate")]
    pub center: [f32; 3],
    #[serde(deserialize_with = "deserialize_widths")]
    pub widths: [f32; 3],
    /// Area in 2-D, volume in 3-D.
    #[serde(alias = "area")]
    pub measure: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brick_key: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refinement_region_scale: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RowTerm {
    pub cell_id: u32,
    pub coefficient: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Row {
    pub id: u32,
    pub kind: RowKind,
    pub axis: u8,
    #[serde(deserialize_with = "deserialize_coordinate")]
    pub center: [f32; 3],
    /// Face measure: length in 2-D, area in 3-D.
    #[serde(alias = "area")]
    pub measure: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(alias = "staticArea")]
    pub static_measure: Option<f32>,
    pub distance: f32,
    pub dual_weight: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub static_dual_weight: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub static_open_fraction: Option<f32>,
    pub terms: Vec<RowTerm>,
    #[serde(default = "one")]
    pub open_fraction: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_fraction_before: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_fraction_after: Option<f32>,
    #[serde(default)]
    pub solid_velocity: f32,
    #[serde(default)]
    pub separating: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Subface {
    pub id: u32,
    pub row_id: u32,
    pub axis: u8,
    #[serde(deserialize_with = "deserialize_coordinate")]
    pub center: [f32; 3],
    #[serde(alias = "area")]
    pub measure: f32,
    /// `-1` denotes a physical boundary endpoint.
    #[serde(default = "minus_one")]
    pub negative_cell: i32,
    /// `-1` denotes a physical boundary endpoint.
    #[serde(default = "minus_one")]
    pub positive_cell: i32,
    #[serde(default = "one")]
    pub aperture: f32,
    #[serde(default)]
    pub solid_velocity: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubfaceIncidence {
    pub subface_id: u32,
    /// -1 means the cell is the negative endpoint; +1 means positive.
    pub orientation: i8,
}

#[derive(Clone, Debug)]
pub(crate) enum SpatialOwnerCache {
    Dense2d {
        dimensions: [usize; 2],
        owners: Vec<i32>,
    },
    Sparse3d {
        widths: Vec<u32>,
        owners: HashMap<(u32, [i32; 3]), u32>,
    },
    LinearFallback,
}

#[derive(Clone, Debug, Default)]
pub struct SpatialOwnerIndex(OnceLock<SpatialOwnerCache>);
// Runtime cache population is not part of graph identity.
impl PartialEq for SpatialOwnerIndex {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl SpatialOwnerIndex {
    pub(crate) fn get_or_build<'a>(&'a self, graph: &Graph) -> &'a SpatialOwnerCache {
        self.0.get_or_init(|| SpatialOwnerCache::build(graph))
    }
    pub(crate) fn initialize(&self, graph: &Graph) {
        let _ = self.get_or_build(graph);
    }
}

impl SpatialOwnerCache {
    fn build(graph: &Graph) -> Self {
        let d = graph.dimension as usize;
        let integral =
            |v: f32| v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v <= i32::MAX as f32;
        if !(d == 2 || d == 3)
            || !(0..d).all(|a| integral(graph.dimensions[a]))
            || graph.cells.iter().any(|c| {
                (0..d).any(|a| {
                    !integral(c.minimum[a])
                        || !integral(c.maximum[a])
                        || c.minimum[a] > c.maximum[a]
                        || c.maximum[a] > graph.dimensions[a]
                })
            })
        {
            return Self::LinearFallback;
        }
        if d == 2 {
            let dimensions = [graph.dimensions[0] as usize, graph.dimensions[1] as usize];
            let Some(count) = dimensions[0].checked_mul(dimensions[1]) else {
                return Self::LinearFallback;
            };
            let mut owners = vec![-1; count];
            for cell in &graph.cells {
                for y in cell.minimum[1] as usize..cell.maximum[1] as usize {
                    for x in cell.minimum[0] as usize..cell.maximum[0] as usize {
                        let at = x + dimensions[0] * y;
                        if owners[at] < 0 {
                            owners[at] = cell.id as i32
                        }
                    }
                }
            }
            Self::Dense2d { dimensions, owners }
        } else {
            let mut owners = HashMap::new();
            let mut widths = Vec::new();
            for cell in &graph.cells {
                // A clipped boundary cell retains its enclosing dyadic tile.
                // Index represented cells, never all the finest voxels inside
                // a coarse bulk cell (which can cover millions of voxels).
                let edge = (0..3)
                    .map(|a| (cell.maximum[a] - cell.minimum[a]) as u32)
                    .max()
                    .unwrap_or(1)
                    .max(1)
                    .next_power_of_two();
                if (0..3).any(|a| (cell.minimum[a] as u32) % edge != 0) {
                    return Self::LinearFallback;
                }
                let key = cell.minimum.map(|v| v as i32);
                if !widths.contains(&edge) {
                    widths.push(edge);
                }
                owners.entry((edge, key)).or_insert(cell.id);
            }
            widths.sort_unstable();
            Self::Sparse3d { widths, owners }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Graph {
    #[serde(default = "schema_version")]
    pub schema_version: u32,
    pub dimension: u8,
    #[serde(deserialize_with = "deserialize_coordinate")]
    pub dimensions: [f32; 3],
    #[serde(default)]
    pub topology_generation: u32,
    pub cells: Vec<Cell>,
    pub rows: Vec<Row>,
    #[serde(default)]
    pub subfaces: Vec<Subface>,
    pub incidences: Vec<Vec<u32>>,
    #[serde(default)]
    pub subface_incidences: Vec<Vec<SubfaceIncidence>>,
    /// Immutable finest-lattice solid Q8 fractions in x-major order.
    #[serde(default)]
    pub solid_voxel_fraction: Vec<f32>,
    #[serde(skip)]
    pub spatial_owner_index: SpatialOwnerIndex,
}

impl Graph {
    pub fn initialize_spatial_owner_cache(&self) {
        self.spatial_owner_index.initialize(self);
    }
}

#[cfg(test)]
mod spatial_index_tests {
    use super::*;
    #[test]
    fn coarse_3d_lookup_storage_scales_with_represented_cells() {
        let graph = Graph {
            dimension: 3,
            dimensions: [1024.0; 3],
            cells: vec![Cell {
                id: 0,
                minimum: [0.0; 3],
                maximum: [1024.0; 3],
                widths: [1024.0; 3],
                ..Cell::default()
            }],
            ..Graph::default()
        };
        graph.initialize_spatial_owner_cache();
        match graph.spatial_owner_index.get_or_build(&graph) {
            SpatialOwnerCache::Sparse3d { owners, widths } => {
                assert_eq!(owners.len(), 1);
                assert_eq!(widths, &[1024]);
            }
            _ => panic!("compiled dyadic cell must use sparse index"),
        }
        assert_eq!(crate::numerics::owner_at(&graph, [1023.5; 3]), Some(0));
        assert_eq!(crate::numerics::owner_at(&graph, [1024.0; 3]), None);
    }
}

const fn schema_version() -> u32 {
    SCHEMA_VERSION
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NumericalFault {
    pub stage: String,
    pub index: u32,
    pub observed: f32,
    pub expected: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Fields {
    pub density: Vec<f32>,
    pub gamma: Vec<f32>,
    pub capacity: Vec<f32>,
    #[serde(default)]
    pub capacity_before: Vec<f32>,
    #[serde(default)]
    pub capacity_after: Vec<f32>,
    #[serde(default)]
    pub solid_motion_active: bool,
    #[serde(default)]
    pub frame_dt: f32,
    #[serde(default)]
    pub acceleration_fine: [f32; 3],
    #[serde(default)]
    pub capacity_rate: Vec<f32>,
    #[serde(default)]
    pub source_rate: Vec<f32>,
    #[serde(default)]
    pub inflow_coverage: Vec<f32>,
    pub cell_velocity: Vec<f32>,
    pub face_velocity: Vec<f32>,
    /// Per-physical-subface fluid-velocity correction produced by the
    /// geometric transport compatibility solve. A row may own several
    /// mixed-resolution subfaces with different centre distances, so this
    /// cannot be represented by changing `face_velocity` once per row.
    #[serde(default)]
    pub subface_velocity_correction: Vec<f32>,
    /// Authoritative compatible physical bulk rate per subface. Empty means
    /// callers derive the rate from face velocity and the diagnostic
    /// correction; a populated plane must match the current graph exactly.
    #[serde(default)]
    pub subface_compatibility_rate: Vec<f64>,
    pub pressure: Vec<f32>,
    pub pressure_rhs: Vec<f32>,
    pub pressure_diagonal: Vec<f32>,
    pub pressure_member: Vec<u8>,
    #[serde(default)]
    pub pressure_row_member: Vec<u8>,
    pub extension_depth: Vec<u8>,
    pub interface_normal: Vec<f32>,
    pub interface_offset: Vec<f32>,
    #[serde(default)]
    pub low_flux: Vec<f32>,
    #[serde(default)]
    pub high_flux: Vec<f32>,
    #[serde(default)]
    pub limited_flux: Vec<f32>,
    #[serde(default)]
    pub characteristic_clearance: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fault: Option<NumericalFault>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationError(pub String);

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ValidationError {}

impl Graph {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ValidationError("unsupported schemaVersion".into()));
        }
        let d = self.dimension as usize;
        if !(d == 2 || d == 3) {
            return Err(ValidationError("dimension must be 2 or 3".into()));
        }
        if self.incidences.len() != self.cells.len() {
            return Err(ValidationError("incidence/cell count differs".into()));
        }
        if !self.subface_incidences.is_empty() && self.subface_incidences.len() != self.cells.len()
        {
            return Err(ValidationError(
                "subface incidence/cell count differs".into(),
            ));
        }
        for (i, c) in self.cells.iter().enumerate() {
            if c.id as usize != i || !(c.measure.is_finite() && c.measure > 0.0) {
                return Err(ValidationError(format!("invalid cell {i}")));
            }
        }
        for (i, row) in self.rows.iter().enumerate() {
            if row.id as usize != i || row.axis as usize >= d || row.terms.is_empty() {
                return Err(ValidationError(format!("invalid row {i}")));
            }
            for term in &row.terms {
                if term.cell_id as usize >= self.cells.len() || !term.coefficient.is_finite() {
                    return Err(ValidationError(format!("invalid term in row {i}")));
                }
            }
        }
        for (cell, rows) in self.incidences.iter().enumerate() {
            for &row_id in rows {
                let Some(row) = self.rows.get(row_id as usize) else {
                    return Err(ValidationError(format!(
                        "invalid incidence for cell {cell}"
                    )));
                };
                if !row.terms.iter().any(|t| t.cell_id as usize == cell) {
                    return Err(ValidationError(format!(
                        "incidence missing term for cell {cell}"
                    )));
                }
            }
        }
        for (i, face) in self.subfaces.iter().enumerate() {
            if face.id as usize != i
                || face.row_id as usize >= self.rows.len()
                || face.axis as usize >= d
            {
                return Err(ValidationError(format!("invalid subface {i}")));
            }
        }
        if !self.solid_voxel_fraction.is_empty() {
            let expected = (0..d)
                .try_fold(1usize, |n, axis| {
                    n.checked_mul(self.dimensions[axis] as usize)
                })
                .ok_or_else(|| ValidationError("solid voxel plane size overflow".into()))?;
            if self.solid_voxel_fraction.len() != expected {
                return Err(ValidationError("solidVoxelFraction length differs".into()));
            }
        }
        Ok(())
    }
}

impl Fields {
    pub fn validate_for(&self, graph: &Graph) -> Result<(), ValidationError> {
        graph.validate()?;
        let n = graph.cells.len();
        let r = graph.rows.len();
        let d = graph.dimension as usize;
        for (name, len) in [
            ("density", self.density.len()),
            ("gamma", self.gamma.len()),
            ("capacity", self.capacity.len()),
            ("pressure", self.pressure.len()),
            ("pressureRhs", self.pressure_rhs.len()),
            ("pressureDiagonal", self.pressure_diagonal.len()),
            ("pressureMember", self.pressure_member.len()),
            ("extensionDepth", self.extension_depth.len()),
            ("interfaceOffset", self.interface_offset.len()),
        ] {
            if len != n {
                return Err(ValidationError(format!(
                    "{name} length {len}, expected {n}"
                )));
            }
        }
        if self.cell_velocity.len() != d * n || self.interface_normal.len() != d * n {
            return Err(ValidationError(
                "dimensioned cell plane length differs".into(),
            ));
        }
        if self.face_velocity.len() != r {
            return Err(ValidationError("faceVelocity/row count differs".into()));
        }
        if !self.subface_velocity_correction.is_empty()
            && self.subface_velocity_correction.len() != graph.subfaces.len()
        {
            return Err(ValidationError(
                "subfaceVelocityCorrection/subface count differs".into(),
            ));
        }
        if !self.subface_compatibility_rate.is_empty()
            && self.subface_compatibility_rate.len() != graph.subfaces.len()
        {
            return Err(ValidationError(
                "subface compatibility rate size differs".into(),
            ));
        }
        Ok(())
    }

    #[inline]
    pub fn optional_cell(plane: &[f32], id: usize, fallback: f32) -> f32 {
        plane.get(id).copied().unwrap_or(fallback)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schema_is_camel_case_and_dimensioned() {
        let graph = Graph {
            schema_version: 1,
            dimension: 3,
            dimensions: [2.0, 2.0, 2.0],
            ..Graph::default()
        };
        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("\"schemaVersion\":1"));
        assert!(json.contains("\"topologyGeneration\":0"));
    }

    #[test]
    fn slice_json_expands_vec2_and_area_aliases() {
        let graph: Graph = serde_json::from_str(
            r#"{
          "schemaVersion":1,"dimension":2,"dimensions":[8,4],"cells":[{
            "id":0,"minimum":[0,0],"maximum":[8,4],"center":[4,2],
            "widths":[8,4],"area":32
          }],"rows":[],"subfaces":[],"incidences":[[]]
        }"#,
        )
        .unwrap();
        assert_eq!(graph.dimensions, [8.0, 4.0, 0.0]);
        assert_eq!(graph.cells[0].widths, [8.0, 4.0, 1.0]);
        assert_eq!(graph.cells[0].measure, 32.0);
        graph.validate().unwrap();
    }
}
