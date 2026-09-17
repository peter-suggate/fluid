use fluid_core::world::{TransportExperiment, World, WorldOptions};

#[test]
fn unconstrained_symmetric_dam_keeps_reflected_adaptive_bricks() {
    let seed: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/split-resolution-ladder-seed.json"
    )).unwrap();
    let mut world = World::from_document(
        serde_json::from_value(seed["scene"].clone()).unwrap(),
        serde_json::from_value(serde_json::json!({"dtS":1.0/30.0,"timeStep":"paper"})).unwrap(),
        WorldOptions { pressure_iterations:256, pressure_relative_tolerance:1e-6,
            transport_experiment:TransportExperiment::LevelSetVolume, ..Default::default() },
    ).unwrap();
    // No enforcement regions: both support planning and final coarsening
    // must choose their own rungs, including the shallow advancing fronts.
    for frame in 1..=60 {
        world.advance(frame, 1.0/30.0).unwrap();
        let bricks = &world.state.topology.bricks;
        for brick in bricks {
            let b = &brick.seed;
            let x = 4 - b.coordinate[0] - b.span_bricks as i32;
            let reflected = bricks.iter().find(|other| {
                let o = &other.seed;
                o.coordinate == [x, b.coordinate[1], b.coordinate[2]] && o.span_bricks == b.span_bricks
            }).expect("reflected physical brick footprint");
            assert_eq!((b.active, b.resolution), (reflected.seed.active, reflected.seed.resolution),
                "frame {frame}, brick {:?} and reflected {:?}", b.coordinate, reflected.seed.coordinate);
        }
        let graph = &world.state.topology.graph;
        let fields = &world.state.fields;
        let mut volume = 0.0;
        for cell in &graph.cells {
            let mirror = graph.cells.iter().find(|c|
                c.center[0] == 32.0-cell.center[0] && c.center[1] == cell.center[1])
                .expect("reflected adaptive cell");
            assert!((fields.density[cell.id as usize]-fields.density[mirror.id as usize]).abs()<1e-4,
                "frame {frame}, reflected density at {:?}",cell.center);
            volume += cell.measure as f64 * fields.density[cell.id as usize] as f64;
        }
        assert!((volume-128.0).abs()<1e-4,"frame {frame}, conservative volume {volume}");
        for row in &graph.rows {
            let mirror = graph.rows.iter().find(|r| r.axis == row.axis && r.measure == row.measure
                && r.center[0] == 32.0-row.center[0] && r.center[1] == row.center[1])
                .expect("reflected staggered face");
            let sign = if row.axis == 0 { -1.0 } else { 1.0 };
            assert!((fields.face_velocity[row.id as usize]-sign*fields.face_velocity[mirror.id as usize]).abs()<1e-3,
                "frame {frame}, reflected velocity at {:?}",row.center);
        }
    }
}

#[test]
fn refinement_distributes_phi_volume_mismatch_without_child_order_bias() {
    use fluid_core::scene::{compile_scene_2d, SceneDescription};
    use fluid_core::transfer::transfer_fields_allow_overcapacity;
    let grid = |resolution| {
        let desc: SceneDescription = serde_json::from_value(serde_json::json!({
            "schemaVersion":1,"dimension":2,"dimensions":[8,8,1],"cellSizeM":0.2,
            "dtS":1.0/30.0,"densityKgM3":1.0,
            "boundaries":["closed","closed","closed","closed","closed","closed"],
            "bricks":[{"id":0,"key":0,"coordinate":[0,0,0],"resolution":resolution,"active":true}]
        })).unwrap();
        compile_scene_2d(desc).unwrap()
    };
    // Each parent has a horizontal phi plane occupying half its area, while
    // accepted volume can be less, greater, or above capacity. Reflection in
    // x must commute with refinement, independently of child enumeration.
    for density in [0.2, 0.8, 1.5] {
        let mut coarse = grid(2);
        let fine = grid(4);
        coarse.fields.density.fill(density);
        for c in &coarse.topology.graph.cells {
            coarse.fields.interface_normal[2*c.id as usize+1] = 1.0;
            coarse.fields.interface_offset[c.id as usize] = 0.0;
        }
        let moved = transfer_fields_allow_overcapacity(&coarse.topology.graph,
            &fine.topology.graph, &coarse.fields, &fine.fields.capacity, &[]).unwrap();
        for c in &fine.topology.graph.cells {
            let mirror = fine.topology.graph.cells.iter().find(|d|
                d.center[0] == 8.0-c.center[0] && d.center[1] == c.center[1]).unwrap();
            assert!((moved.density[c.id as usize]-moved.density[mirror.id as usize]).abs()<1e-6,
                "density {density}, reflected children {:?}/{:?}: {}/{}",c.center,mirror.center,
                moved.density[c.id as usize],moved.density[mirror.id as usize]);
            assert!(moved.density[c.id as usize] >= 0.0);
        }
        for c in &coarse.topology.graph.cells {
            let sum:f64 = fine.topology.graph.cells.iter().filter(|d|
                (0..2).all(|a|d.minimum[a]>=c.minimum[a] && d.maximum[a]<=c.maximum[a]))
                .map(|d|moved.density[d.id as usize] as f64*d.measure as f64).sum();
            assert!((sum-density as f64*c.measure as f64).abs()<2e-6);
        }
    }
}
