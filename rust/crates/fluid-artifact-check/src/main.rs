//! Validate actual instructions, not byte patterns that may occur in immediates.
use wasmparser::{Parser, Payload, Validator, WasmFeatures};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    while let Some(artifact) = args.next() {
        let path = args.next().ok_or("expected artifact and path pairs")?;
        let bytes = std::fs::read(path)?;
        let mut features = WasmFeatures::default();
        features.set(WasmFeatures::RELAXED_SIMD, false);
        features.set(WasmFeatures::SIMD, artifact != "scalar");
        features.set(WasmFeatures::THREADS, artifact == "threaded");
        Validator::new_with_features(features).validate_all(&bytes)
            .map_err(|e| format!("{artifact} feature validation: {e}"))?;
        let (mut simd, mut atomic) = (0, 0);
        for payload in Parser::new(0).parse_all(&bytes) {
            if let Payload::CodeSectionEntry(body) = payload? {
                let mut operators = body.get_operators_reader()?;
                while !operators.eof() {
                    let offset = operators.original_position();
                    operators.read()?;
                    match bytes[offset] {
                        0xfd => simd += 1,
                        0xfe => atomic += 1,
                        _ => {}
                    }
                }
            }
        }
        if artifact != "scalar" && simd == 0 { return Err("SIMD artifact has no SIMD instructions".into()); }
        if artifact == "threaded" && atomic == 0 { return Err("threaded artifact has no atomic instructions".into()); }
        println!("{artifact}: {simd} SIMD instructions, {atomic} atomic instructions; strict feature validation passed");
    }
    Ok(())
}
