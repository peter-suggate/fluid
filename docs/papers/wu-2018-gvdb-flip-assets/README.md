# Wu et al. (2018) GVDB FLIP reference assets

These files accompany `../wu-2018-gvdb-flip.txt`.

- Title: *Fast Fluid Simulations with Sparse Volumes on the GPU*
- Authors: Kui Wu, Nghia Truong, Cem Yuksel, and Rama Hoetzlein
- Venue: Computer Graphics Forum 37(2), 2018, pages 157-167
- DOI: <https://doi.org/10.1111/cgf.13350>
- Author project page: <https://people.csail.mit.edu/kuiwu/gvdb_sim.html>
- Source PDF: <https://people.csail.mit.edu/kuiwu/GVDB_FLIP/gvdb_flip.pdf>
- Source PDF SHA-256: `3af396b1bd3d7239f2f1a32d88dfea4d130c9f3baeb4cae02631289354254b26`

The 11 numbered PNG files are complete 144-DPI page renders. Complete pages
are retained so equations, algorithms, table labels, captions, and surrounding
arguments remain together. The `embedded/` directory contains the ten native
raster images from the PDF at their original encoded resolutions.

The publication carries an author and publisher copyright notice. Confirm the
repository's distribution policy before pushing the PDF-derived page renders
or embedded images to a public remote.

## Page map

- 01: title, abstract, introduction, and Figure 1
- 02: related work and GVDB background
- 03: GVDB hierarchy, brick atlas, apron update, and Algorithm 1
- 04: full topology rebuild and Figure 3
- 05: incremental topology rebuild and subcell construction
- 06: subcell rasterization, Figures 4-5, and matrix-free CG overview
- 07: matrix-free CG details, Algorithm 2, and Figure 6
- 08: topology results, Table 1, solver results, and Figure 7
- 09: solver and pipeline timings, Tables 2-3, and Figures 8-9
- 10: rendering, conclusions, memory Table 4, Figure 10, and references
- 11: references

## Embedded image map

- `figure-000.jpg`: Figure 1 (page 1)
- `figure-001.png`: Figure 2 (page 3)
- `figure-002.png`: Figure 3 (page 4)
- `figure-003.png`: Figure 4 (page 6)
- `figure-004.png`: Figure 5 (page 6)
- `figure-005.jpg`: Figure 6 (page 7)
- `figure-006.jpg`: Figure 7 (page 8)
- `figure-007.jpg`: Figure 8 (page 9)
- `figure-008.jpg`: Figure 9 (page 9)
- `figure-009.png`: Figure 10 (page 10)

## Reproduction

The text, complete page renders, and native images were generated with Poppler:

```sh
pdftotext -layout -enc UTF-8 gvdb_flip.pdf wu-2018-gvdb-flip.txt
pdftoppm -png -r 144 gvdb_flip.pdf wu-2018-gvdb-flip-assets/page
pdfimages -png -j gvdb_flip.pdf wu-2018-gvdb-flip-assets/embedded/figure
```
