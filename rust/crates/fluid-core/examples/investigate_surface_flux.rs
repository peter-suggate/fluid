//! Integrate the sampled velocity through the authoritative phi contour.
use fluid_core::{
    levelset_redistance::sample_scalar, levelset_surface, staggered_velocity::StaggeredVelocity2d,
    Fields, Graph,
};
use serde_json::{json, Value};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let rows: Vec<Value> = std::fs::read_to_string(&args[1])
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    for pair in rows.windows(2) {
        let prev = &pair[0];
        let row = &pair[1];
        let surface = levelset_surface::publish(
            [32, 16],
            serde_json::from_value(prev["phi"].clone()).unwrap(),
            128.0,
        )
        .unwrap();
        let stage = row["stages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["name"] == "level-set-volume-velocity-extension")
            .unwrap();
        let f = &stage["fields"];
        let graph = Graph {
            dimension: 2,
            dimensions: [32.0, 16.0, 1.0],
            cells: serde_json::from_value(f["cells"].clone()).unwrap(),
            rows: serde_json::from_value(f["rows"].clone()).unwrap(),
            ..Default::default()
        };
        let mut fields = Fields {
            face_velocity: serde_json::from_value(f["faceVelocity"].clone()).unwrap(),
            capacity: serde_json::from_value(f["capacity"].clone()).unwrap(),
            ..Default::default()
        };
        if args.iter().any(|a| a == "--project-air") {
            fluid_core::levelset_air_extension::project_air_extension(
                &graph,
                &mut fields,
                &levelset_surface::cell_phi(&graph, &surface).unwrap(),
            )
            .unwrap();
        }
        let velocity = StaggeredVelocity2d::new(&graph, &fields).unwrap();
        let mut sides = vec![[0.0_f64; 4]; graph.cells.len()];
        let mut measures = sides.clone();
        let mut owner = vec![usize::MAX; 512];
        for c in &graph.cells {
            for y in c.minimum[1] as usize..c.maximum[1] as usize {
                for x in c.minimum[0] as usize..c.maximum[0] as usize {
                    owner[x + 32 * y] = c.id as usize;
                }
            }
        }
        for r in &graph.rows {
            for t in &r.terms {
                let i = t.cell_id as usize;
                let a = r.axis as usize;
                let side = 2 * a + usize::from(r.center[a] > graph.cells[i].center[a]);
                let w =
                    (t.coefficient.abs() * r.static_dual_weight.unwrap_or(r.dual_weight)) as f64;
                sides[i][side] += w * fields.face_velocity[r.id as usize] as f64;
                measures[i][side] += w;
            }
        }
        for i in 0..sides.len() {
            for k in 0..4 {
                if measures[i][k] > 0.0 {
                    sides[i][k] /= measures[i][k];
                }
            }
        }
        let sample = |p: [f32; 2]| -> [f32; 2] {
            if args.get(2).is_none_or(|a| a != "rt0") {
                return velocity.sample(p);
            }
            let p = [p[0].clamp(0.0, 31.99999), p[1].clamp(0.0, 15.99999)];
            let i = owner[p[0] as usize + 32 * p[1] as usize];
            if i == usize::MAX {
                return [0.0; 2];
            }
            let c = &graph.cells[i];
            [0, 1].map(|a| {
                let t = ((p[a] - c.minimum[a]) / c.widths[a]) as f64;
                ((1.0 - t) * sides[i][2 * a] + t * sides[i][2 * a + 1]) as f32
            })
        };
        let mut flux = [0.0; 2];
        let mut max_div = [0.0_f32; 2];
        let mut midflux = 0.0;
        for s in surface.segments_fine.chunks_exact(4) {
            let d = [s[2] - s[0], s[3] - s[1]];
            let normal = [d[1], -d[0]];
            for k in 0..16 {
                let t = (k as f32 + 0.5) / 16.0;
                let p = [s[0] + t * d[0], s[1] + t * d[1]];
                let g = [
                    sample_scalar(&surface, [p[0] + 0.001, p[1]]).unwrap_or(0.0)
                        - sample_scalar(&surface, [p[0] - 0.001, p[1]]).unwrap_or(0.0),
                    sample_scalar(&surface, [p[0], p[1] + 0.001]).unwrap_or(0.0)
                        - sample_scalar(&surface, [p[0], p[1] - 0.001]).unwrap_or(0.0),
                ];
                let sign = if g[0] * normal[0] + g[1] * normal[1] > 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let u = sample(p);
                let side = usize::from(p[0] >= 16.0);
                flux[side] += (sign * (normal[0] * u[0] + normal[1] * u[1]) / 16.0) as f64;
                let div = (sample([p[0] + 0.001, p[1]])[0] - sample([p[0] - 0.001, p[1]])[0]
                    + sample([p[0], p[1] + 0.001])[1]
                    - sample([p[0], p[1] - 0.001])[1])
                    / 0.002;
                max_div[side] = max_div[side].max(div.abs());
            }
        }
        for k in 0..4096 {
            let y = (k as f32 + 0.5) / 256.0;
            if sample_scalar(&surface, [16.0, y]).unwrap() < 0.0 {
                midflux += sample([16.0, y])[0] as f64 / 256.0;
            }
        }
        println!(
            "{}",
            json!({"frame":row["frame"],"contourFlux":flux,"middleFlux":midflux,"closedHalfFlux":[flux[0]+midflux,flux[1]-midflux],"maximumSurfaceSampleDivergence":max_div})
        );
    }
}
