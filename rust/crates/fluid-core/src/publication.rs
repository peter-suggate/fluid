//! Versioned binary publication: small JSON metadata plus aligned, typed planes.
//! No field arrays pass through JSON, and NaNs retain their exact f32 payloads.
pub const PUBLICATION_MAGIC: &[u8; 8] = b"FLUIDCPU";
pub const PUBLICATION_VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 32;
pub const DIRECTORY_ENTRY_BYTES: usize = 16;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum PlaneId {
    Density = 1,
    Gamma = 2,
    Capacity = 3,
    CellVelocity = 4,
    FaceVelocity = 5,
    Pressure = 6,
    PressureRhs = 7,
    PressureDiagonal = 8,
    PressureMember = 9,
    PressureRowMember = 10,
    ExtensionDepth = 11,
    InterfaceNormal = 12,
    InterfaceOffset = 13,
    LowFlux = 14,
    HighFlux = 15,
    LimitedFlux = 16,
    CapacityBefore = 17,
    CapacityAfter = 18,
    CapacityRate = 19,
    SourceRate = 20,
    InflowCoverage = 21,
    CharacteristicClearance = 22,
    Tracers = 30,
    RdfVertices = 31,
    RdfSegments = 32,
    DensityBefore = 40,
    VelocityXBeforePressure = 41,
    VelocityYBeforePressure = 42,
    BrickResolutionBefore = 43,
    BrickActivity = 44,
    MaterialId = 45,
    CapacityFine = 46,
    Density3D = 60,
    SurfacePhi3D = 61,
    Velocity3D = 62,
    GraphJson = 100,
    SceneJson = 101,
}
pub enum Plane<'a> {
    F32(PlaneId, &'a [f32]),
    U8(PlaneId, &'a [u8]),
    Json(PlaneId, &'a [u8]),
}
impl Plane<'_> {
    fn description(&self) -> (u32, u32, usize) {
        match self {
            Self::F32(id, data) => (*id as u32, 1, data.len()),
            Self::U8(id, data) => (*id as u32, 2, data.len()),
            Self::Json(id, data) => (*id as u32, 3, data.len()),
        }
    }
}
fn align64(value: usize) -> usize {
    (value + 63) & !63
}

/// Caller owns the destination and may reuse its allocation after transfer.
pub fn encode_publication(
    metadata: &[u8],
    planes: &[Plane<'_>],
    destination: &mut Vec<u8>,
) -> Result<(), &'static str> {
    let directory_bytes = planes
        .len()
        .checked_mul(DIRECTORY_ENTRY_BYTES)
        .ok_or("publication directory overflow")?;
    let metadata_offset = HEADER_BYTES
        .checked_add(directory_bytes)
        .ok_or("publication metadata overflow")?;
    let metadata_end = metadata_offset
        .checked_add(metadata.len())
        .ok_or("publication metadata overflow")?;
    if metadata_end > u32::MAX as usize - 63 {
        return Err("publication exceeds wasm32 address space");
    }
    let mut end = align64(metadata_end);
    let mut entries = Vec::with_capacity(planes.len());
    for plane in planes {
        let (id, kind, count) = plane.description();
        let bytes = count
            .checked_mul(if kind == 1 { 4 } else { 1 })
            .ok_or("publication plane overflow")?;
        entries.push([
            id,
            kind,
            end as u32,
            u32::try_from(count).map_err(|_| "publication plane count overflow")?,
        ]);
        end = end.checked_add(bytes).ok_or("publication plane overflow")?;
        if end > u32::MAX as usize - 63 {
            return Err("publication exceeds wasm32 address space");
        }
        end = align64(end);
    }
    destination.resize(end, 0);
    destination.fill(0);
    destination[..8].copy_from_slice(PUBLICATION_MAGIC);
    let header = [
        PUBLICATION_VERSION,
        end as u32,
        planes.len() as u32,
        metadata_offset as u32,
        metadata.len() as u32,
        0,
    ];
    for (i, value) in header.iter().enumerate() {
        destination[8 + 4 * i..12 + 4 * i].copy_from_slice(&value.to_le_bytes())
    }
    destination[metadata_offset..metadata_end].copy_from_slice(metadata);
    for (i, (plane, entry)) in planes.iter().zip(entries).enumerate() {
        for (j, value) in entry.iter().enumerate() {
            let at = HEADER_BYTES + i * DIRECTORY_ENTRY_BYTES + j * 4;
            destination[at..at + 4].copy_from_slice(&value.to_le_bytes())
        }
        let at = entry[2] as usize;
        match plane {
            Plane::F32(_, data) => {
                for (i, value) in data.iter().enumerate() {
                    destination[at + 4 * i..at + 4 * i + 4]
                        .copy_from_slice(&value.to_bits().to_le_bytes())
                }
            }
            Plane::U8(_, data) | Plane::Json(_, data) => {
                destination[at..at + data.len()].copy_from_slice(data)
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_preserves_nan_and_signed_zero_and_aligns_planes() {
        let values = [f32::from_bits(0x7fc00042), -0.0, 1.0];
        let mut out = Vec::new();
        encode_publication(
            b"{}",
            &[Plane::F32(PlaneId::RdfVertices, &values)],
            &mut out,
        )
        .unwrap();
        assert_eq!(&out[..8], PUBLICATION_MAGIC);
        let offset = u32::from_le_bytes(out[40..44].try_into().unwrap()) as usize;
        assert_eq!(offset % 64, 0);
        assert_eq!(
            u32::from_le_bytes(out[offset..offset + 4].try_into().unwrap()),
            0x7fc00042
        );
        assert_eq!(
            u32::from_le_bytes(out[offset + 4..offset + 8].try_into().unwrap()),
            0x80000000
        );
    }
}
