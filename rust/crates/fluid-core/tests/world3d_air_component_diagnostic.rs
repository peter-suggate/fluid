//! Manual census for the frame-13 closed-air compatibility failure.
//!
//! Run with `cargo test -p fluid-core --test world3d_air_component_diagnostic
//! -- --ignored --nocapture`. This deliberately inspects the latched, partially
//! advanced state after the compatibility gate rejects the frame.

use fluid_core::{
    initial_scene::SceneDocument, production_scene::ProductionSceneOptions,
    runtime_options3d::apply_initial_values, types::RowKind, world::WorldOptions, world3d::World3d,
};

fn root(parent: &mut [usize], mut cell: usize) -> usize {
    while parent[cell] != cell {
        cell = parent[cell];
    }
    let result = cell;
    // The diagnostic does not depend on component numbering, but stable roots
    // make its output directly comparable with the production failure.
    while parent[cell] != cell {
        let next = parent[cell];
        parent[cell] = result;
        cell = next;
    }
    result
}

#[test]
#[ignore = "manual frame-13 compatibility census"]
fn frame13_air_and_pressure_flux_partition() {
    let scene: SceneDocument = serde_json::from_str(include_str!(
        "../../../core/testdata/water-box-body-free-world3d-scene.json"
    ))
    .unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/water-box-ui-default-cpu-world3d-options.json"
    ))
    .unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/water-box-ui-default-cpu-world3d-options.json"
    ))
    .unwrap();
    let mut production = ProductionSceneOptions::default();
    let mut options: WorldOptions = serde_json::from_value(fixture["options"].clone()).unwrap();
    fluid_core::runtime_options3d::apply_initial_values(
        &mut production,
        &mut options,
        &fixture["options"]["methodValues"],
    )
    .unwrap();
    let mut world = World3d::from_document(scene, production, options).unwrap();
    for sequence in 1..=13 {
        let result = world.advance(sequence, 1.0 / 30.0);
        eprintln!("advance {sequence}: {result:?}");
        if result.is_err() {
            break;
        }
    }

    let graph = &world.state.topology.graph;
    let fields = &world.state.fields;
    let n = graph.cells.len();
    let air: Vec<bool> = (0..n)
        .map(|i| {
            graph.cells[i].measure * fields.capacity[i] > 0.0 && fields.pressure_member[i] == 0
        })
        .collect();
    let mut parent: Vec<usize> = (0..n).collect();
    for face in &graph.subfaces {
        let row = &graph.rows[face.row_id as usize];
        if row.open_fraction <= 0.0 || face.negative_cell < 0 || face.positive_cell < 0 {
            continue;
        }
        let a = face.negative_cell as usize;
        let b = face.positive_cell as usize;
        if !air[a] || !air[b] {
            continue;
        }
        let ra = root(&mut parent, a);
        let rb = root(&mut parent, b);
        let r = ra.min(rb);
        parent[ra] = r;
        parent[rb] = r;
    }
    for i in 0..n {
        if air[i] {
            parent[i] = root(&mut parent, i);
        }
    }

    #[derive(Clone, Default)]
    struct Census {
        members: usize,
        capacity: f64,
        residual: f64,
        interface: f64,
        sparse: f64,
        closed_world: f64,
        separating_wall: f64,
        internal: f64,
        source: f64,
        capacity_rate: f64,
        open_faces: usize,
        interface_faces: usize,
        separating_faces: usize,
        min: [f32; 3],
        max: [f32; 3],
    }
    let mut census = vec![
        Census {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
            ..Census::default()
        };
        n
    ];
    let incidences = if graph.subface_incidences.len() == n {
        graph.subface_incidences.clone()
    } else {
        let mut result = vec![vec![]; n];
        for face in &graph.subfaces {
            if face.negative_cell >= 0 {
                result[face.negative_cell as usize].push(fluid_core::SubfaceIncidence {
                    subface_id: face.id,
                    orientation: -1,
                });
            }
            if face.positive_cell >= 0 {
                result[face.positive_cell as usize].push(fluid_core::SubfaceIncidence {
                    subface_id: face.id,
                    orientation: 1,
                });
            }
        }
        result
    };
    for cell in 0..n {
        if !air[cell] {
            continue;
        }
        let c = &mut census[parent[cell]];
        c.members += 1;
        let volume = graph.cells[cell].measure as f64 * fields.capacity[cell] as f64;
        c.capacity += volume;
        c.source += fields.source_rate.get(cell).copied().unwrap_or(0.0) as f64;
        c.capacity_rate += graph.cells[cell].measure as f64
            * fields.capacity_rate.get(cell).copied().unwrap_or(0.0) as f64;
        for axis in 0..3 {
            c.min[axis] = c.min[axis].min(graph.cells[cell].minimum[axis]);
            c.max[axis] = c.max[axis].max(graph.cells[cell].maximum[axis]);
        }
        for entry in &incidences[cell] {
            let face = &graph.subfaces[entry.subface_id as usize];
            let row = &graph.rows[face.row_id as usize];
            let velocity = fields.face_velocity[row.id as usize]
                - (1.0 - row.open_fraction) * row.solid_velocity;
            let signed = entry.orientation as f64 * face.measure as f64 * velocity as f64;
            c.residual += signed;
            let other = if face.negative_cell == cell as i32 {
                face.positive_cell
            } else {
                face.negative_cell
            };
            if row.kind == RowKind::ClosedWorld {
                c.closed_world += signed;
                if row.separating {
                    c.separating_wall += signed;
                    c.separating_faces += 1;
                }
            } else if other < 0 {
                c.sparse += signed;
                if row.open_fraction > 0.0 {
                    c.open_faces += 1;
                }
            } else if fields.pressure_member[other as usize] != 0 {
                c.interface += signed;
                c.interface_faces += 1;
            } else {
                c.internal += signed;
            }
        }
    }

    for (component, c) in census.iter().enumerate() {
        if c.members == 0 || (c.residual.abs() < 1e-4 && c.interface.abs() < 1e-4) {
            continue;
        }
        eprintln!(
            "AIR root={component} members={} cap={:.9} residual={:.9} interface={:.9}/{} sparse={:.9}/{} closed={:.9} separating={:.9}/{} internal={:.9} source={:.9} capRate={:.9} box={:?}..{:?}",
            c.members, c.capacity, c.residual + c.source - c.capacity_rate, c.interface, c.interface_faces,
            c.sparse, c.open_faces, c.closed_world, c.separating_wall,
            c.separating_faces, c.internal, c.source, c.capacity_rate, c.min, c.max
        );
    }

    let mut pressure_physical = 0.0f64;
    let mut pressure_algebraic = 0.0f64;
    let mut separating = 0.0f64;
    for cell in 0..n {
        if fields.pressure_member[cell] == 0 {
            continue;
        }
        for entry in &incidences[cell] {
            let face = &graph.subfaces[entry.subface_id as usize];
            let row = &graph.rows[face.row_id as usize];
            let velocity = fields.face_velocity[row.id as usize]
                - (1.0 - row.open_fraction) * row.solid_velocity;
            pressure_physical += entry.orientation as f64 * face.measure as f64 * velocity as f64;
        }
        for &row_id in &graph.incidences[cell] {
            let row = &graph.rows[row_id as usize];
            let Some(term) = row.terms.iter().find(|term| term.cell_id as usize == cell) else {
                continue;
            };
            let velocity = fields.face_velocity[row.id as usize]
                - (1.0 - row.open_fraction) * row.solid_velocity;
            pressure_algebraic += term.coefficient as f64
                * row.static_dual_weight.unwrap_or(row.dual_weight) as f64
                * velocity as f64;
            if row.kind == RowKind::ClosedWorld && row.separating {
                separating += term.coefficient as f64
                    * row.static_dual_weight.unwrap_or(row.dual_weight) as f64
                    * velocity as f64;
            }
        }
    }
    eprintln!(
        "PCM physical={pressure_physical:.9} algebraic={pressure_algebraic:.9} delta={:.9} separating={separating:.9} refinement={:?}",
        pressure_physical - pressure_algebraic,
        world.primary_refinement
    );
}
