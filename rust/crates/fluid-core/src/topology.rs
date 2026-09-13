//! Sparse accepted topology compiler.
//!
//! The emitted [`Graph`] is the sole stage authority.  Spatial maps and
//! transfer groups are construction accelerators only and never reorder a
//! row's terms or a cell's incidences.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::geometry::{box_overlap_measure, BoundaryMode};
use crate::types::{Cell, Graph, Row, RowKind, RowTerm, Subface, SubfaceIncidence, SCHEMA_VERSION};

pub const BRICK_FINE_RESOLUTION: i32 = 8;
pub const INVALID_CELL: i32 = -1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrickSeed {
    pub id: u32,
    pub key: u32,
    pub coordinate: [i32; 3],
    #[serde(default = "one_u32")]
    pub span_bricks: u32,
    pub resolution: u8,
    #[serde(default = "yes")]
    pub active: bool,
    #[serde(default)]
    pub density: Vec<f32>,
    #[serde(default)]
    pub gamma: Vec<f32>,
    #[serde(default)]
    pub refinement_region_scale: Option<f32>,
}

const fn one_u32() -> u32 {
    1
}
const fn yes() -> bool {
    true
}

#[derive(Clone, Debug)]
pub struct TopologySeed<const D: usize> {
    pub dimensions: [u32; 3],
    pub generation: u32,
    pub sparse_air_phi: f32,
    /// Axis-major: negative, positive.
    pub boundaries: [BoundaryMode; 6],
    pub bricks: Vec<BrickSeed>,
}

#[derive(Clone, Debug)]
pub struct BrickRecord {
    pub seed: BrickSeed,
    pub cell_range: std::ops::Range<u32>,
}

#[derive(Clone, Debug)]
pub struct CompiledTopology<const D: usize> {
    pub graph: Graph,
    pub boundaries: [BoundaryMode; 6],
    pub sparse_air_phi: f32,
    pub bricks: Vec<BrickRecord>,
    pub cell_by_stable_id: HashMap<u32, u32>,
    /// One entry per target cell; sources stay in accepted source cell order.
    pub transfer_groups: Vec<Vec<TransferSource>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferSource {
    pub source_cell: u32,
    pub overlap_measure: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TopologyError {
    UnsupportedDimension,
    InvalidDimensions,
    InvalidGeneration,
    InvalidSparseAirPhi,
    InvalidBrick(u32, &'static str),
    DuplicateBrickId(u32),
    DuplicateBrickKey(u32),
    OverlappingBricks(u32, u32),
    GradingViolation(u32, u32),
    IdOverflow,
    IncompleteTransfer(u32),
}

impl std::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for TopologyError {}

#[derive(Clone, Debug)]
struct AtomicFace {
    axis: usize,
    coordinate: i32,
    tangent: [i32; 2],
    negative: i32,
    positive: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum GroupKey {
    Boundary(usize, i32, i32, i32, [i32; 2]),
    Pair(usize, i32, i32),
    Mixed(usize, i32, i32),
}

pub fn compile_topology<const D: usize>(
    mut seed: TopologySeed<D>,
) -> Result<CompiledTopology<D>, TopologyError> {
    if D != 2 && D != 3 {
        return Err(TopologyError::UnsupportedDimension);
    }
    if seed.generation == 0 {
        return Err(TopologyError::InvalidGeneration);
    }
    if !seed.sparse_air_phi.is_finite() || seed.sparse_air_phi <= 0.0 {
        return Err(TopologyError::InvalidSparseAirPhi);
    }
    if (0..D).any(|a| seed.dimensions[a] == 0 || seed.dimensions[a] > i32::MAX as u32) {
        return Err(TopologyError::InvalidDimensions);
    }
    validate_bricks::<D>(&seed.bricks, seed.dimensions)?;
    seed.bricks.sort_by_key(|brick| brick.key);

    let capacity = if D == 2 { 64_u32 } else { 512_u32 };
    let mut cells = Vec::new();
    let mut records = Vec::with_capacity(seed.bricks.len());
    let mut cell_by_stable_id = HashMap::new();
    for brick in &seed.bricks {
        let begin = cells.len() as u32;
        if brick.active {
            let r = brick.resolution as usize;
            let width = BRICK_FINE_RESOLUTION * brick.span_bricks as i32 / r as i32;
            let count = r.pow(D as u32);
            for local_index in 0..count {
                let local = decode_index::<D>(local_index, r);
                let mut minimum = [0.0; 3];
                let mut maximum = [1.0; 3];
                let mut widths = [1.0; 3];
                for axis in 0..D {
                    let min =
                        brick.coordinate[axis] * BRICK_FINE_RESOLUTION + local[axis] as i32 * width;
                    let max = (min + width).min(seed.dimensions[axis] as i32);
                    minimum[axis] = min as f32;
                    maximum[axis] = max as f32;
                    widths[axis] = (max - min) as f32;
                }
                if (0..D).any(|a| widths[a] <= 0.0) {
                    continue;
                }
                let id = u32::try_from(cells.len()).map_err(|_| TopologyError::IdOverflow)?;
                let stable_id = brick
                    .key
                    .checked_mul(capacity)
                    .and_then(|v| v.checked_add(local_index as u32))
                    .ok_or(TopologyError::IdOverflow)?;
                let mut center = [0.0; 3];
                for a in 0..3 {
                    center[a] = 0.5 * (minimum[a] + maximum[a]);
                }
                let measure = (0..D).fold(1.0_f32, |v, a| v * widths[a]);
                cells.push(Cell {
                    id,
                    stable_id: Some(stable_id),
                    minimum,
                    maximum,
                    center,
                    widths,
                    measure,
                    brick_key: Some(brick.key),
                    refinement_region_scale: brick.refinement_region_scale,
                });
                cell_by_stable_id.insert(stable_id, id);
            }
        }
        records.push(BrickRecord {
            seed: brick.clone(),
            cell_range: begin..cells.len() as u32,
        });
    }

    let brick_order: HashMap<u32, usize> = seed
        .bricks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.key, i))
        .collect();
    let faces = atomic_faces::<D>(&cells)?;
    let mut groups: BTreeMap<GroupKey, Vec<AtomicFace>> = BTreeMap::new();
    for face in faces {
        let negative = (face.negative >= 0).then(|| &cells[face.negative as usize]);
        let positive = (face.positive >= 0).then(|| &cells[face.positive as usize]);
        let key = match (negative, positive) {
            (Some(n), Some(p)) if n.widths[face.axis] == p.widths[face.axis] => {
                GroupKey::Pair(face.axis, face.negative, face.positive)
            }
            (Some(n), Some(p)) => {
                let coarse = if n.widths[face.axis] > p.widths[face.axis] {
                    face.negative
                } else {
                    face.positive
                };
                GroupKey::Mixed(face.axis, face.coordinate, coarse)
            }
            _ => {
                let own = if face.negative >= 0 {
                    face.negative
                } else {
                    face.positive
                };
                let side = if face.negative >= 0 { 1 } else { -1 };
                let cell = &cells[own as usize];
                let tangents: Vec<_> = (0..D).filter(|&a| a != face.axis).collect();
                let mut lower = [0; 2];
                for (i, &axis) in tangents.iter().enumerate() {
                    lower[i] = cell.minimum[axis] as i32;
                }
                GroupKey::Boundary(face.axis, face.coordinate, own, side, lower)
            }
        };
        groups.entry(key).or_default().push(face);
    }
    let mut ordered: Vec<_> = groups.into_values().collect();
    // Canonical keys allocate and consult brick ownership. Compute each once
    // per generation instead of repeating that work inside sort comparisons.
    ordered.sort_by_cached_key(|pieces| canonical_group_order::<D>(pieces, &cells, &brick_order));

    let mut rows = Vec::with_capacity(ordered.len());
    let mut subfaces = Vec::new();
    for mut pieces in ordered {
        pieces.sort_by_key(|p| p.tangent);
        let first = &pieces[0];
        let negative: BTreeSet<u32> = pieces
            .iter()
            .filter_map(|p| (p.negative >= 0).then_some(p.negative as u32))
            .collect();
        let positive: BTreeSet<u32> = pieces
            .iter()
            .filter_map(|p| (p.positive >= 0).then_some(p.positive as u32))
            .collect();
        let measure = pieces.len() as f32;
        let neg_center = weighted_axis_center(&pieces, &cells, true, first.axis);
        let pos_center = weighted_axis_center(&pieces, &cells, false, first.axis);
        let only = negative
            .iter()
            .next()
            .or_else(|| positive.iter().next())
            .copied()
            .unwrap();
        let distance = match (neg_center, pos_center) {
            (Some(n), Some(p)) => p - n,
            _ => cells[only as usize].widths[first.axis],
        };
        let mut terms = Vec::new();
        append_terms(&mut terms, &pieces, &negative, true, measure, distance);
        append_terms(&mut terms, &pieces, &positive, false, measure, distance);
        let mut kind = if negative.is_empty() || positive.is_empty() {
            RowKind::SparseAir
        } else {
            let n = &cells[*negative.iter().next().unwrap() as usize];
            let p = &cells[*positive.iter().next().unwrap() as usize];
            if n.widths[first.axis] != p.widths[first.axis] {
                RowKind::MixedSeam
            } else if n.brick_key == p.brick_key {
                RowKind::IntraBrick
            } else {
                RowKind::BrickFace
            }
        };
        if kind == RowKind::SparseAir
            && boundary_for(first, seed.dimensions, seed.boundaries) == Some(BoundaryMode::Closed)
        {
            kind = RowKind::ClosedWorld;
        }
        let id = rows.len() as u32;
        let center = face_center::<D>(&pieces);
        rows.push(Row {
            id,
            kind,
            axis: first.axis as u8,
            center,
            measure,
            static_measure: Some(measure),
            distance,
            dual_weight: measure * distance,
            static_dual_weight: Some(measure * distance),
            static_open_fraction: Some(1.0),
            terms,
            open_fraction: 1.0,
            open_fraction_before: Some(1.0),
            open_fraction_after: Some(1.0),
            solid_velocity: 0.0,
            separating: false,
        });
        if D == 2 {
            let mut at = 0;
            while at < pieces.len() {
                let first_piece = &pieces[at];
                let mut end = at + 1;
                while end < pieces.len()
                    && pieces[end].negative == first_piece.negative
                    && pieces[end].positive == first_piece.positive
                    && pieces[end].tangent[0] == pieces[end - 1].tangent[0] + 1
                {
                    end += 1;
                }
                let mut center = atomic_center::<D>(first_piece);
                let tangent = 1 - first_piece.axis;
                center[tangent] =
                    0.5 * (first_piece.tangent[0] + pieces[end - 1].tangent[0] + 1) as f32;
                subfaces.push(Subface {
                    id: subfaces.len() as u32,
                    row_id: id,
                    axis: first_piece.axis as u8,
                    center,
                    measure: (end - at) as f32,
                    negative_cell: first_piece.negative,
                    positive_cell: first_piece.positive,
                    aperture: 1.0,
                    solid_velocity: 0.0,
                });
                at = end;
            }
        } else {
            for piece in pieces {
                subfaces.push(Subface {
                    id: subfaces.len() as u32,
                    row_id: id,
                    axis: piece.axis as u8,
                    center: atomic_center::<D>(&piece),
                    measure: 1.0,
                    negative_cell: piece.negative,
                    positive_cell: piece.positive,
                    aperture: 1.0,
                    solid_velocity: 0.0,
                });
            }
        }
    }

    let mut incidences = vec![Vec::new(); cells.len()];
    for row in &rows {
        for term in &row.terms {
            incidences[term.cell_id as usize].push(row.id);
        }
    }
    for cell_rows in &mut incidences {
        cell_rows.sort_by_key(|&row_id| {
            let row = &rows[row_id as usize];
            let owner = row
                .terms
                .iter()
                .filter_map(|t| cells[t.cell_id as usize].brick_key)
                .filter_map(|k| brick_order.get(&k).copied())
                .min()
                .unwrap_or(usize::MAX);
            let resolution = row
                .terms
                .iter()
                .find_map(|t| {
                    let key = cells[t.cell_id as usize].brick_key?;
                    (brick_order.get(&key).copied() == Some(owner))
                        .then(|| seed.bricks[owner].resolution)
                })
                .unwrap_or(8);
            (owner, rung_ordinal(resolution), row_id)
        });
    }
    let mut subface_incidences = vec![Vec::new(); cells.len()];
    let mut subfaces_by_row = vec![Vec::new(); rows.len()];
    for face in &subfaces {
        subfaces_by_row[face.row_id as usize].push(face);
    }
    // GV_CELL_FACE walks cell-major incidence order, then each row's contiguous
    // physical pieces. This order is observable in f32 limiter reductions.
    for (cell, cell_rows) in incidences.iter().enumerate() {
        for &row_id in cell_rows {
            for face in &subfaces_by_row[row_id as usize] {
                if face.negative_cell == cell as i32 {
                    subface_incidences[cell].push(SubfaceIncidence {
                        subface_id: face.id,
                        orientation: -1,
                    });
                } else if face.positive_cell == cell as i32 {
                    subface_incidences[cell].push(SubfaceIncidence {
                        subface_id: face.id,
                        orientation: 1,
                    });
                }
            }
        }
    }
    let graph = Graph {
        schema_version: SCHEMA_VERSION,
        dimension: D as u8,
        dimensions: seed.dimensions.map(|v| v as f32),
        topology_generation: seed.generation,
        cells,
        rows,
        subfaces,
        incidences,
        subface_incidences,
        solid_voxel_fraction: Vec::new(),
        spatial_owner_index: Default::default(),
    };
    graph.initialize_spatial_owner_cache();
    graph.validate().map_err(|_| TopologyError::IdOverflow)?;
    Ok(CompiledTopology {
        graph,
        boundaries: seed.boundaries,
        sparse_air_phi: seed.sparse_air_phi,
        bricks: records,
        cell_by_stable_id,
        transfer_groups: Vec::new(),
    })
}

pub fn build_transfer_groups<const D: usize>(
    source: &CompiledTopology<D>,
    target: &mut CompiledTopology<D>,
) -> Result<(), TopologyError> {
    build_transfer_groups_with_new_air(source, target, false)
}

pub fn build_transfer_groups_with_new_air<const D: usize>(
    source: &CompiledTopology<D>,
    target: &mut CompiledTopology<D>,
    allow_new_air: bool,
) -> Result<(), TopologyError> {
    if source.graph.dimensions != target.graph.dimensions {
        return Err(TopologyError::InvalidDimensions);
    }
    let mut groups = Vec::with_capacity(target.graph.cells.len());
    for next in &target.graph.cells {
        let mut ids = BTreeSet::new();
        for_each_integer_point::<D>(next.minimum, next.maximum, |point| {
            if let Some(id) =
                crate::numerics::owner_at(&source.graph, point.map(|v| v as f32 + 0.5))
                    .map(|v| v as u32)
            {
                ids.insert(id);
            }
        });
        let mut covered = 0.0_f32;
        let mut group = Vec::new();
        for id in ids {
            let old = &source.graph.cells[id as usize];
            let measure =
                box_overlap_measure::<D>(next.minimum, next.maximum, old.minimum, old.maximum);
            if measure > 0.0 {
                covered += measure;
                group.push(TransferSource {
                    source_cell: id,
                    overlap_measure: measure,
                });
            }
        }
        let tolerance = 1.0e-6_f32 * next.measure.max(1.0);
        if covered > next.measure + tolerance
            || (!allow_new_air && covered < next.measure - tolerance)
        {
            return Err(TopologyError::IncompleteTransfer(next.id));
        }
        groups.push(group);
    }
    target.transfer_groups = groups;
    Ok(())
}

fn validate_bricks<const D: usize>(
    bricks: &[BrickSeed],
    dimensions: [u32; 3],
) -> Result<(), TopologyError> {
    let mut ids = BTreeSet::new();
    let mut keys = BTreeSet::new();
    for brick in bricks {
        if !ids.insert(brick.id) {
            return Err(TopologyError::DuplicateBrickId(brick.id));
        }
        if !keys.insert(brick.key) {
            return Err(TopologyError::DuplicateBrickKey(brick.key));
        }
        if !matches!(brick.resolution, 1 | 2 | 4 | 8) {
            return Err(TopologyError::InvalidBrick(brick.key, "resolution"));
        }
        if !brick.span_bricks.is_power_of_two() {
            return Err(TopologyError::InvalidBrick(brick.key, "span"));
        }
        if (0..D).any(|a| brick.coordinate[a] % brick.span_bricks as i32 != 0) {
            return Err(TopologyError::InvalidBrick(brick.key, "alignment"));
        }
        let count = (brick.resolution as usize).pow(D as u32);
        if (!brick.density.is_empty() && brick.density.len() != count)
            || (!brick.gamma.is_empty() && brick.gamma.len() != count)
        {
            return Err(TopologyError::InvalidBrick(brick.key, "field length"));
        }
    }
    for i in 0..bricks.len() {
        for j in i + 1..bricks.len() {
            let (amin, amax) = brick_bounds::<D>(&bricks[i], dimensions);
            let (bmin, bmax) = brick_bounds::<D>(&bricks[j], dimensions);
            if (0..D).all(|a| amin[a] < bmax[a] && bmin[a] < amax[a]) {
                return Err(TopologyError::OverlappingBricks(
                    bricks[i].key,
                    bricks[j].key,
                ));
            }
            let share = (0..D).any(|normal| {
                (amax[normal] == bmin[normal] || bmax[normal] == amin[normal])
                    && (0..D)
                        .filter(|&a| a != normal)
                        .all(|a| amin[a] < bmax[a] && bmin[a] < amax[a])
            });
            if share {
                let aw = 8 * bricks[i].span_bricks / bricks[i].resolution as u32;
                let bw = 8 * bricks[j].span_bricks / bricks[j].resolution as u32;
                if aw.max(bw) / aw.min(bw) > 2 {
                    return Err(TopologyError::GradingViolation(
                        bricks[i].key,
                        bricks[j].key,
                    ));
                }
            }
        }
    }
    Ok(())
}

fn brick_bounds<const D: usize>(brick: &BrickSeed, dimensions: [u32; 3]) -> ([i32; 3], [i32; 3]) {
    let mut min = [0; 3];
    let mut max = [1; 3];
    for a in 0..D {
        min[a] = brick.coordinate[a] * 8;
        max[a] = (min[a] + 8 * brick.span_bricks as i32).min(dimensions[a] as i32);
    }
    (min, max)
}

fn decode_index<const D: usize>(mut index: usize, resolution: usize) -> [usize; 3] {
    let mut result = [0; 3];
    for item in result.iter_mut().take(D) {
        *item = index % resolution;
        index /= resolution;
    }
    result
}

fn for_each_integer_point<const D: usize>(
    min: [f32; 3],
    max: [f32; 3],
    mut visit: impl FnMut([i32; 3]),
) {
    let lo = min.map(|v| v.floor() as i32);
    let hi = max.map(|v| v.ceil() as i32);
    for z in lo[2]..if D == 3 { hi[2] } else { lo[2] + 1 } {
        for y in lo[1]..hi[1] {
            for x in lo[0]..hi[0] {
                visit([x, y, z]);
            }
        }
    }
}

fn atomic_faces<const D: usize>(cells: &[Cell]) -> Result<Vec<AtomicFace>, TopologyError> {
    let mut planes: BTreeMap<(usize, i32, [i32; 2]), (i32, i32)> = BTreeMap::new();
    for cell in cells {
        for axis in 0..D {
            for side in [-1_i32, 1] {
                let coordinate = if side < 0 {
                    cell.minimum[axis]
                } else {
                    cell.maximum[axis]
                } as i32;
                let tangents: Vec<_> = (0..D).filter(|&a| a != axis).collect();
                let a0 = tangents.first().map_or(0, |&a| cell.minimum[a] as i32);
                let a1 = tangents.first().map_or(1, |&a| cell.maximum[a] as i32);
                let b0 = tangents.get(1).map_or(0, |&a| cell.minimum[a] as i32);
                let b1 = tangents.get(1).map_or(1, |&a| cell.maximum[a] as i32);
                for b in b0..b1 {
                    for a in a0..a1 {
                        let pair = planes
                            .entry((axis, coordinate, [a, b]))
                            .or_insert((INVALID_CELL, INVALID_CELL));
                        let slot = if side > 0 { &mut pair.0 } else { &mut pair.1 };
                        if *slot != INVALID_CELL {
                            return Err(TopologyError::OverlappingBricks(
                                cell.brick_key.unwrap_or(0),
                                cell.brick_key.unwrap_or(0),
                            ));
                        }
                        *slot = cell.id as i32;
                    }
                }
            }
        }
    }
    Ok(planes
        .into_iter()
        .map(
            |((axis, coordinate, tangent), (negative, positive))| AtomicFace {
                axis,
                coordinate,
                tangent,
                negative,
                positive,
            },
        )
        .collect())
}

fn canonical_group_order<const D: usize>(
    pieces: &[AtomicFace],
    cells: &[Cell],
    brick_order: &HashMap<u32, usize>,
) -> Vec<i64> {
    let f = &pieces[0];
    let n = (f.negative >= 0).then(|| &cells[f.negative as usize]);
    let p = (f.positive >= 0).then(|| &cells[f.positive as usize]);
    if let (Some(n), Some(p)) = (n, p) {
        let no = brick_order[&n.brick_key.unwrap()];
        let po = brick_order[&p.brick_key.unwrap()];
        if no == po {
            if D == 2 {
                let local = p.stable_id.unwrap_or(p.id) % 64;
                return vec![
                    0,
                    no as i64,
                    f.axis as i64,
                    local as i64,
                    f.tangent[0] as i64,
                ];
            }
            vec![
                0,
                no as i64,
                f.axis as i64,
                f.tangent[1] as i64,
                f.tangent[0] as i64,
            ]
        } else {
            vec![
                1,
                no as i64,
                f.axis as i64,
                0,
                po as i64,
                f.tangent[1] as i64,
                f.tangent[0] as i64,
            ]
        }
    } else {
        let own = n.or(p).unwrap();
        let order = brick_order[&own.brick_key.unwrap()];
        if D == 2 {
            let local = own.stable_id.unwrap_or(own.id) % 64;
            return vec![
                1,
                order as i64,
                f.axis as i64,
                if n.is_some() { 1 } else { 2 },
                local as i64,
                f.tangent[0] as i64,
            ];
        }
        vec![
            1,
            order as i64,
            f.axis as i64,
            if n.is_some() { 1 } else { 2 },
            f.tangent[1] as i64,
            f.tangent[0] as i64,
        ]
    }
}

fn weighted_axis_center(
    pieces: &[AtomicFace],
    cells: &[Cell],
    negative: bool,
    axis: usize,
) -> Option<f32> {
    let mut total = 0.0;
    let mut count = 0;
    for p in pieces {
        let id = if negative { p.negative } else { p.positive };
        if id >= 0 {
            total += cells[id as usize].center[axis];
            count += 1;
        }
    }
    (count > 0).then(|| total / count as f32)
}

fn append_terms(
    out: &mut Vec<RowTerm>,
    pieces: &[AtomicFace],
    ids: &BTreeSet<u32>,
    negative: bool,
    measure: f32,
    distance: f32,
) {
    let sign = if negative { -1.0 } else { 1.0 };
    for &id in ids {
        let covered = pieces
            .iter()
            .filter(|p| {
                if negative {
                    p.negative == id as i32
                } else {
                    p.positive == id as i32
                }
            })
            .count() as f32;
        out.push(RowTerm {
            cell_id: id,
            coefficient: sign * covered / (measure * distance),
        });
    }
}

fn atomic_center<const D: usize>(face: &AtomicFace) -> [f32; 3] {
    let mut c = [0.0; 3];
    c[face.axis] = face.coordinate as f32;
    let tangents: Vec<_> = (0..D).filter(|&a| a != face.axis).collect();
    if let Some(&a) = tangents.first() {
        c[a] = face.tangent[0] as f32 + 0.5;
    }
    if let Some(&a) = tangents.get(1) {
        c[a] = face.tangent[1] as f32 + 0.5;
    }
    c
}

fn face_center<const D: usize>(pieces: &[AtomicFace]) -> [f32; 3] {
    let mut c = [0.0; 3];
    for p in pieces {
        let q = atomic_center::<D>(p);
        for a in 0..3 {
            c[a] += q[a];
        }
    }
    for item in &mut c {
        *item /= pieces.len() as f32;
    }
    c
}

fn boundary_for(
    face: &AtomicFace,
    dimensions: [u32; 3],
    boundaries: [BoundaryMode; 6],
) -> Option<BoundaryMode> {
    if face.coordinate == 0 {
        Some(boundaries[2 * face.axis])
    } else if face.coordinate == dimensions[face.axis] as i32 {
        Some(boundaries[2 * face.axis + 1])
    } else {
        None
    }
}

fn rung_ordinal(r: u8) -> u8 {
    match r {
        1 => 0,
        2 => 1,
        4 => 2,
        _ => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn brick(key: u32, coordinate: [i32; 3], resolution: u8) -> BrickSeed {
        BrickSeed {
            id: key,
            key,
            coordinate,
            span_bricks: 1,
            resolution,
            active: true,
            density: vec![],
            gamma: vec![],
            refinement_region_scale: None,
        }
    }
    #[test]
    fn compiles_2d_pair_and_boundary_rows() {
        let top = compile_topology::<2>(TopologySeed {
            dimensions: [16, 8, 1],
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Open; 6],
            bricks: vec![brick(1, [0, 0, 0], 1), brick(2, [1, 0, 0], 1)],
        })
        .unwrap();
        assert_eq!(top.graph.cells.len(), 2);
        assert!(top.graph.rows.iter().any(|r| r.kind == RowKind::BrickFace));
        assert_eq!(top.graph.subface_incidences.len(), 2);
    }
    #[test]
    fn compiles_true_3d_z_row() {
        let top = compile_topology::<3>(TopologySeed {
            dimensions: [8, 8, 16],
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Open; 6],
            bricks: vec![brick(1, [0, 0, 0], 1), brick(2, [0, 0, 1], 1)],
        })
        .unwrap();
        assert!(top
            .graph
            .rows
            .iter()
            .any(|r| r.axis == 2 && r.kind == RowKind::BrickFace));
    }
    #[test]
    fn mixed_fixture_matches_slice_compiler_receipt() {
        // Receipt captured from compileSliceTopology for the same deliberately
        // unsorted two-brick source input.
        let top = compile_topology::<2>(TopologySeed {
            dimensions: [16, 8, 1],
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Open; 6],
            bricks: vec![brick(2, [1, 0, 0], 8), brick(1, [0, 0, 0], 4)],
        })
        .unwrap();
        assert_eq!(
            (
                top.graph.cells.len(),
                top.graph.rows.len(),
                top.graph.subfaces.len()
            ),
            (80, 176, 180)
        );
        let mixed: Vec<_> = top
            .graph
            .rows
            .iter()
            .filter(|r| r.kind == RowKind::MixedSeam)
            .collect();
        assert_eq!(mixed.len(), 4);
        assert_eq!(mixed[0].center, [8.0, 1.0, 0.0]);
        assert_eq!(mixed[0].measure, 2.0);
        assert_eq!(mixed[0].distance, 1.5);
        assert_eq!(
            mixed[0].terms,
            vec![
                RowTerm {
                    cell_id: 3,
                    coefficient: -0.6666667
                },
                RowTerm {
                    cell_id: 16,
                    coefficient: 0.33333334
                },
                RowTerm {
                    cell_id: 24,
                    coefficient: 0.33333334
                }
            ]
        );
    }
    #[test]
    fn indexed_transfer_is_complete_and_source_ordered() {
        let seed = |resolution| TopologySeed::<2> {
            dimensions: [8, 8, 1],
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Open; 6],
            bricks: vec![brick(1, [0, 0, 0], resolution)],
        };
        let source = compile_topology(seed(1)).unwrap();
        let mut target = compile_topology(seed(2)).unwrap();
        build_transfer_groups(&source, &mut target).unwrap();
        assert!(target
            .transfer_groups
            .iter()
            .all(|g| g.len() == 1 && g[0].overlap_measure == 16.0));
    }
}
