//! Narrow-band FLIP (Ferstl et al. 2016) on the advance-lab's 2D lattice: the
//! fixed particle band of lib/methods/uniform/uniform-narrow-band-flip.ts on
//! its mixed frame, a 4h base with h tiles at the surface and a two-stage
//! pressure solve. The h grid stages (extension, level-set advect, forces,
//! inflow, rigid coupling) and the 4h solve are uniform_geometric's.
pub mod band;
pub mod layout;
pub mod options;
pub mod particles;
pub mod session;
pub mod surface;
pub mod world;
pub use options::NarrowBandFlipOptions;
