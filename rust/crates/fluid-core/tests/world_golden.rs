use fluid_core::{
    initial_scene::SceneDocument,
    production_scene::ProductionSceneOptions,
    types::{Fields, Graph, RowKind},
    world::{World, WorldOptions},
};
use serde::Deserialize;
use std::collections::BTreeMap;
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Case {
    id: String,
    scene: SceneDocument,
    production_options: ProductionSceneOptions,
    world_options: WorldOptions,
    frames: Vec<Frame>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Frame {
    frame: u32,
    topology_generation: u32,
    cells: usize,
    rows: usize,
    subfaces: usize,
    float_hash: String,
    word_hash: String,
    field_hashes: BTreeMap<String, String>,
    microsteps: usize,
    source_pending: f32,
    pressure_iterations: u32,
    pressure_converged: bool,
    tracer: [u64; 5],
    pressure_authority: Option<[u32; 10]>,
    scalar_authority: Option<[u32; 9]>,
}
fn hash(b: &[u8]) -> String {
    let mut h = 14695981039346656037u64;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(1099511628211)
    }
    format!("{h:016x}")
}
fn fhash(v: &[f32]) -> String {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend(x.to_bits().to_le_bytes())
    }
    hash(&b)
}
fn whash(v: &[u32]) -> String {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend(x.to_le_bytes())
    }
    hash(&b)
}
fn graph_hashes(g: &Graph) -> (String, String) {
    let d = g.dimension as usize;
    let (mut f, mut w) = (Vec::new(), vec![g.dimension as u32]);
    w.extend(g.dimensions[..d].iter().map(|&x| x as u32));
    w.extend([
        g.cells.len() as u32,
        g.rows.len() as u32,
        g.subfaces.len() as u32,
    ]);
    for c in &g.cells {
        f.extend_from_slice(&c.minimum[..d]);
        f.extend_from_slice(&c.maximum[..d]);
        f.extend_from_slice(&c.center[..d]);
        f.extend_from_slice(&c.widths[..d]);
        f.extend([c.measure, c.refinement_region_scale.unwrap_or(1.)]);
        w.extend([c.id, c.stable_id.unwrap(), c.brick_key.unwrap()])
    }
    for r in &g.rows {
        f.extend_from_slice(&r.center[..d]);
        f.extend([
            r.measure,
            r.static_measure.unwrap(),
            r.distance,
            r.static_dual_weight.unwrap(),
            r.dual_weight,
            r.open_fraction,
            r.solid_velocity,
        ]);
        let k = match r.kind {
            RowKind::IntraBrick => 0,
            RowKind::BrickFace => 1,
            RowKind::MixedSeam => 2,
            RowKind::SparseAir => 3,
            RowKind::ClosedWorld => 4,
        };
        w.extend([r.id, r.axis as u32, k, r.terms.len() as u32]);
        for t in &r.terms {
            w.extend([t.cell_id, t.coefficient.to_bits()])
        }
    }
    for s in &g.subfaces {
        f.extend_from_slice(&s.center[..d]);
        f.extend([s.measure, s.aperture, s.solid_velocity]);
        w.extend([
            s.id,
            s.row_id,
            s.axis as u32,
            s.negative_cell as u32,
            s.positive_cell as u32,
        ])
    }
    for v in &g.incidences {
        w.push(v.len() as u32);
        w.extend(v)
    }
    for v in &g.subface_incidences {
        w.push(v.len() as u32);
        for e in v {
            w.extend([e.subface_id, if e.orientation < 0 { u32::MAX } else { 1 }])
        }
    }
    (fhash(&f), whash(&w))
}
fn plane<'a>(f: &'a Fields, n: &str) -> Option<&'a [f32]> {
    Some(match n {
        "density" => &f.density,
        "gamma" => &f.gamma,
        "capacity" => &f.capacity,
        "capacityBefore" => &f.capacity_before,
        "capacityAfter" => &f.capacity_after,
        "capacityRate" => &f.capacity_rate,
        "sourceRate" => &f.source_rate,
        "inflowCoverage" => &f.inflow_coverage,
        "cellVelocity" => &f.cell_velocity,
        "faceVelocity" => &f.face_velocity,
        "pressure" => &f.pressure,
        "pressureRhs" => &f.pressure_rhs,
        "pressureDiagonal" => &f.pressure_diagonal,
        "interfaceNormal" => &f.interface_normal,
        "interfaceOffset" => &f.interface_offset,
        "lowFlux" => &f.low_flux,
        "highFlux" => &f.high_flux,
        "limitedFlux" => &f.limited_flux,
        "characteristicClearance" => &f.characteristic_clearance,
        _ => return None,
    })
}
#[test]
fn production_world_matches_frozen_typescript_frames() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../../../core/testdata/world-golden.json")).unwrap();
    for c in fixture.cases {
        let mut world =
            World::from_document(c.scene, c.production_options, c.world_options).unwrap();
        for e in c.frames {
            if e.frame > 0 {
                world.advance(e.frame, world.timestep_s).unwrap()
            }
            let g = &world.state.topology.graph;
            assert_eq!(
                (g.cells.len(), g.rows.len(), g.subfaces.len()),
                (e.cells, e.rows, e.subfaces),
                "{} f{} counts",
                c.id,
                e.frame
            );
            assert_eq!(
                g.topology_generation, e.topology_generation,
                "{} f{} gen",
                c.id, e.frame
            );
            let (fh, wh) = graph_hashes(g);
            assert_eq!(fh, e.float_hash, "{} f{} graph floats", c.id, e.frame);
            assert_eq!(wh, e.word_hash, "{} f{} graph words", c.id, e.frame);
            for (n, want) in &e.field_hashes {
                let got = match n.as_str() {
                    "pressureMember" => hash(&world.state.fields.pressure_member),
                    "pressureRowMember" => hash(&world.state.fields.pressure_row_member),
                    "extensionDepth" => hash(&world.state.fields.extension_depth),
                    _ => fhash(plane(&world.state.fields, n).unwrap()),
                };
                assert_eq!(&got, want, "{} f{} {}", c.id, e.frame, n)
            }
            assert_eq!(
                world.microsteps, e.microsteps,
                "{} f{} microsteps",
                c.id, e.frame
            );
            assert_eq!(
                world.source_ledger.pending.to_bits(),
                e.source_pending.to_bits(),
                "{} f{} pending",
                c.id,
                e.frame
            );
            assert_eq!(
                (world.pressure.iterations, world.pressure.converged),
                (e.pressure_iterations, e.pressure_converged),
                "{} f{} pressure",
                c.id,
                e.frame
            );
            let t = world.tracer_receipt;
            assert_eq!(
                [
                    t.generation as u64,
                    t.seeded as u64,
                    t.count as u64,
                    t.live_count as u64,
                    t.retired_count as u64
                ],
                e.tracer,
                "{} f{} tracer",
                c.id,
                e.frame
            );
            let p = world
                .embedding
                .as_ref()
                .map(|e| &e.pressure_authority.receipt)
                .unwrap_or(&world.pressure_authority.receipt);
            assert_eq!(
                Some([
                    p.topology_generation,
                    p.cell_generation,
                    p.row_generation,
                    p.coefficient_generation,
                    p.execution_generation,
                    p.changed_diagonal_count,
                    p.pressure_cell_count,
                    p.pressure_row_count,
                    p.fault,
                    p.first_fault_id
                ]),
                e.pressure_authority,
                "{} f{} pressure authority",
                c.id,
                e.frame
            );
            let s = &world.scalar_authority.receipt;
            assert_eq!(
                Some([
                    s.generation,
                    s.topology_generation,
                    s.topology_slot as u32,
                    s.changed_cell_count,
                    s.nonexact_cell_count,
                    s.bulk_cell_count,
                    s.flip_cell_count,
                    s.fault,
                    s.first_fault_packet
                ]),
                e.scalar_authority,
                "{} f{} scalar authority",
                c.id,
                e.frame
            )
        }
    }
}
