# Narrow-band FLIP and Uniform 4h pressure references

Retrieved 7 October 2026 for the [repository-specific analysis](../../plans/uniform-narrow-band-flip-analysis-2026-10-07.md).

Seven downloaded references: **65 complete rendered pages**, **140 native raster image/mask files**, approximately **382.2 MiB** including original PDFs. This is a focused set covering NB-FLIP, EXNB-FLIP, coarse/local projection, adaptive pressure layers and large-step transport; it is not an exhaustive survey.

| Paper | PDF | UTF-8 text | Image index | Pages |
| --- | --- | --- | --- | ---: |
| Narrow Band FLIP for Liquid Simulations (2016) | [PDF](ferstl-2016-nbflip/paper.pdf) | [Text](ferstl-2016-nbflip/paper.txt) | [Pages and images](ferstl-2016-nbflip/README.md) | 8 |
| Extended Narrow Band FLIP for Liquid Simulations (2018) | [PDF](sato-2018-exnbflip/paper.pdf) | [Text](sato-2018-exnbflip/paper.txt) | [Pages and images](sato-2018-exnbflip/README.md) | 9 |
| A Practical Octree Liquid Simulator with Adaptive Surface Resolution (2020) | [PDF](ando-batty-2020-octree/paper.pdf) | [Text](ando-batty-2020-octree/paper.txt) | [Pages and images](ando-batty-2020-octree/README.md) | 17 |
| Spatially Adaptive FLIP Fluid Simulations in Bifrost (2016) | [PDF](nielsen-2016-bifrost/paper.pdf) | [Text](nielsen-2016-bifrost/paper.txt) | [Pages and images](nielsen-2016-bifrost/README.md) | 2 |
| Adaptive Optical Layers: Efficient Tall Cell Grids for Liquid Simulation (2026) | [PDF](narita-2026-optical-layers/paper.pdf) | [Text](narita-2026-optical-layers/paper.txt) | [Pages and images](narita-2026-optical-layers/README.md) | 10 |
| Simulating Free Surface Flow with Very Large Time Steps (2012) | [PDF](lentine-2012-large-timesteps/paper.pdf) | [Text](lentine-2012-large-timesteps/paper.txt) | [Pages and images](lentine-2012-large-timesteps/README.md) | 10 |
| A Novel Algorithm for Incompressible Flow Using Only a Coarse Grid Projection (2010) | [PDF](lentine-2010-coarse-projection/paper.pdf) | [Text](lentine-2010-coarse-projection/paper.txt) | [Pages and images](lentine-2010-coarse-projection/README.md) | 9 |

## Existing companion archives

These already contain PDFs, extracted text, all page PNGs and embedded images; they are reused without duplication.

- [ST-FLIP (2026)](../realtime-particles-2026/braun-2026-st-flip/README.md): high-CFL transfer and projection; [text](../realtime-particles-2026/braun-2026-st-flip/paper.txt).
- [APIC (2015)](../realtime-particles-2026/jiang-2015-apic/README.md): affine transfer; [text](../realtime-particles-2026/jiang-2015-apic/paper.txt).
- [Adaptive Phase-Field-FLIP (2025)](../realtime-particles-2026/braun-2025-adaptive-phase-field-flip/README.md): alternative surface/projection formulation; [text](../realtime-particles-2026/braun-2025-adaptive-phase-field-flip/paper.txt).
- [Liu et al. Schur-complement analysis](../liu-2016-schur-complement-fluids-notes.md): global interface coupling and the limits of treating it as a physical adaptive discretization.

## Read these pages first

- NB-FLIP [page 4](ferstl-2016-nbflip/page-4.png): the energy failure of naive velocity blending; [page 5](ferstl-2016-nbflip/page-5.png): nested bands, interior surface and resampling.
- EXNB-FLIP [page 5](sato-2018-exnbflip/page-5.png): APIC transfer and position correction; [page 6](sato-2018-exnbflip/page-6.png): activity field and complete algorithm.
- Coarse projection [page 7](lentine-2010-coarse-projection/page-7.png): why water needs an additional connected fine free-surface solve.
- Octree liquids [page 12](ando-batty-2020-octree/page-12.png): EXNB-FLIP integration; [page 16](ando-batty-2020-octree/page-16.png): accuracy and conservation limits.
- Adaptive Optical Layers [page 6](narita-2026-optical-layers/page-06.png): EXNB-FLIP, pressure accuracy and CFL settings.
- Large timesteps [page 4](lentine-2012-large-timesteps/page-04.png): conservative transport and velocity extrapolation.

## Provenance, conversion and limitations

[sources.json](sources.json) records author/publisher URLs and retrieval date. Each paper manifest records its SHA-256, PDF page count, render resolution and native-image count. The source PDFs are unchanged; original author attribution and rights notices remain. Copies are research references, not newly licensed assets.

Run `python3 docs/papers/narrow-band-flip-2026/extract.py` from any working directory using the appropriate absolute script path, or supply paper IDs for a subset. Existing PDFs are reused. Poppler commands:

```sh
pdftotext -layout -enc UTF-8 paper.pdf paper.txt
pdfimages -png -j paper.pdf embedded/image
pdfimages -list paper.pdf > embedded-images.txt
pdftoppm -png -r 144 paper.pdf page
```

Complete 144-DPI page images retain vector plots, equations and captions that embedded raster extraction alone misses. Embedded-image counts include masks and figure fragments. Layout-preserving text may have ligatures, control characters or imperfect equation ordering; consult page images for equations.

All downloaded PDF checksums, rendered page counts and raster decodability were verified. Opening pages and selected algorithm/limitation pages were visually inspected; no claim of exhaustive visual review of every page is made.

**Reference-only / unavailable full text:** [Asynchronous Eulerian Liquid Simulation, Koike et al. (2020)](https://diglib.eg.org/items/a25b4a24-7b37-4612-88c8-3ddea8a81200). The legacy download redirected to login, and the current publisher PDF API returned HTTP 401. Its abstract was consulted; no PDF, text conversion or page images are claimed for it. The failed source and status are retained in sources.json.
