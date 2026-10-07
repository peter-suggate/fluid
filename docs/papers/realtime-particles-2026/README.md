# Realtime incompressible particle method references

Nine papers supporting the [research and implementation plan](../../plans/realtime-particle-method-2026-10-07.md), retrieved on 7 October 2026. The collection includes recent methods and the APIC/DFSPH foundations needed to implement and compare them.

Each paper directory contains `paper.pdf`, layout-preserving UTF-8 `paper.txt`, complete 144-DPI page PNGs, native raster images and masks under `embedded/`, an image-to-page listing, and a manifest with its source URL and PDF SHA-256. Complete page images preserve vector figures, equations, table labels and captions that native image extraction cannot capture.

| Paper | Year | Text | Page and image index | Pages | Embedded files |
| --- | --- | --- | --- | --- | --- |
| Spatiotemporal FLIP | 2026 | [Text](braun-2026-st-flip/paper.txt) | [Images and source](braun-2026-st-flip/README.md) | 20 | 20 |
| Implicit Position-Based Fluids | 2025 | [Text](diaz-2025-ipbf/paper.txt) | [Images and source](diaz-2025-ipbf/README.md) | 9 | 30 |
| Adaptive Phase-Field-FLIP | 2025 | [Text](braun-2025-adaptive-phase-field-flip/paper.txt) | [Images and source](braun-2025-adaptive-phase-field-flip/README.md) | 23 | 37 |
| Tube Maps | 2026 | [Text](nogina-2026-tube-maps/paper.txt) | [Images and source](nogina-2026-tube-maps/README.md) | 15 | 232 |
| Impulse Particle-In-Cell | 2024 | [Text](sancho-2024-ipic/paper.txt) | [Images and source](sancho-2024-ipic/README.md) | 13 | 139 |
| Particle Flow Maps | 2024 | [Text](zhou-2024-particle-flow-maps/paper.txt) | [Images and source](zhou-2024-particle-flow-maps/README.md) | 20 | 138 |
| Vortex Particle Flow Maps | 2025 | [Text](wang-2025-vortex-particle-flow-maps/paper.txt) | [Images and source](wang-2025-vortex-particle-flow-maps/README.md) | 24 | 215 |
| Affine Particle-In-Cell | 2015 | [Text](jiang-2015-apic/paper.txt) | [Images and source](jiang-2015-apic/README.md) | 10 | 25 |
| Divergence-Free SPH | 2015 | [Text](bender-2015-dfsph/paper.txt) | [Images and source](bender-2015-dfsph/README.md) | 9 | 10 |

The archive totals **143 rendered pages and 846 embedded image/mask files**, approximately **496 MB** including PDFs. Embedded-file count is not figure count: a figure may contain multiple raster objects, separate transparency masks and vector annotations. The source PDFs are unchanged; text extraction retains PDF column layout and may contain ligatures or imperfect equation ordering.

## Pages to read first

- ST-FLIP: [Algorithm 1 and runtime methodology, page 10](braun-2026-st-flip/page-10.png); [stage timings and physical validation, page 12](braun-2026-st-flip/page-12.png); [limitations, page 16](braun-2026-st-flip/page-16.png).
- IPBF: [matched-error performance table and damping comparison, page 8](diaz-2025-ipbf/page-8.png); [limitations, page 9](diaz-2025-ipbf/page-9.png).
- Adaptive Phase-Field-FLIP: [large-scene timings and limitations, page 20](braun-2025-adaptive-phase-field-flip/page-20.png).
- Tube Maps: [timing table, page 8](nogina-2026-tube-maps/page-08.png); [sharp-feature limitations, page 10](nogina-2026-tube-maps/page-10.png). This PDF includes supplementary material.
- APIC: [opening comparison and abstract](jiang-2015-apic/page-01.png).
- IPIC: [opening APIC versus IPIC comparison](sancho-2024-ipic/page-01.png).

The earlier [GVDB FLIP archive](../wu-2018-gvdb-flip-assets/README.md) supplies an additional GPU implementation reference without duplicating those files here.

## Reproduction and provenance

[sources.json](sources.json) records the selected author/publisher download URLs. Run `python3 docs/papers/realtime-particles-2026/extract.py` from the repository root, or pass one or more paper directory names to regenerate a subset. The script reuses existing PDFs and requires Poppler's `pdfinfo`, `pdftotext`, `pdfimages` and `pdftoppm`.

For each paper it runs the equivalent of:

```sh
pdftotext -layout -enc UTF-8 paper.pdf paper.txt
pdfimages -png -j paper.pdf embedded/image
pdfimages -list paper.pdf > embedded-images.txt
pdftoppm -png -r 144 paper.pdf page
```

Every PDF checksum, rendered page count and extracted image was checked after extraction. All raster files decoded successfully. The nine opening pages and the IPBF timing table were visually inspected; the complete text and page renders preserve the authors' attribution and rights notices.
