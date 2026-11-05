# libWetCloth water source assets

These 36 WCL1 topology assets are deterministic SI imports of released inputs
for Fei, Batty, Grinspun and Zheng, *A Multi-Scale Model for Simulating
Liquid-Fabric Interactions* (2018). Source repository:
https://github.com/raymondyfei/libwetcloth/tree/b43993d775a0f9e3a9e32611f26936f2528fac37/assets

Copyright 2018 Yun (Raymond) Fei, Christopher Batty, Eitan Grinspun, and Changxi
Zheng. The upstream project is distributed under Mozilla Public License 2.0;
see LICENSE.txt. The importer is `tools/import-wetcloth-scenes.py` and source
metadata/checksums are in `lib/cloth/scenes/source-catalog.json`.

Run `python3 tools/import-wetcloth-scenes.py --fetch` to obtain pinned source
XML and regenerate, or append `--check` to verify deterministic output.
The catalog's asset SHA-256 is over the uncompressed WCL1 bytes. The compressed
file uses deterministic gzip. Positions, velocities and material parameters
are converted from CGS to SI; position/velocity/scalar arrays are float32.
Vertex addresses and triangle/rod connectivity are unchanged. All rods and
shells from a source share one address space, preserving pile root attachment.

WCL1 starts with four ASCII magic bytes, a little-endian uint32 JSON-header
length (padded with spaces to a multiple of four), that header, then tightly
packed little-endian arrays described by byte offset/count/type. Header source
SHA-256 identifies the original XML. Metadata retains every authored script,
source primitive, CSG relationship, source duration and original setting.

Honey, oil and acetal comparison inputs are excluded at the user's direction.
The imported source material values are retained; no claim is made that a
simulation reproduces the paper merely because its geometry was imported.
Paper/source discrepancies remain explicit catalog audit notes. The release omits
`assets/meshes/spot_triangulated.obj`. The canonical same-name triangulated Spot
model was recovered from the author’s CC0 distribution; its provenance and hash
are recorded in the source inventory. Byte identity to the unpublished upstream
copy remains unverified and is an explicit audit note.
