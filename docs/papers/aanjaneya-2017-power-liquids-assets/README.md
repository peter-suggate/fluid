# Aanjaneya et al. (2017) reference assets

These files accompany `../aanjaneya-2017-power-liquids.pdf` and
`../aanjaneya-2017-power-liquids.txt`.

- Title: *Power Diagrams and Sparse Paged Grids for High Resolution Adaptive Liquids*
- Authors: Mridul Aanjaneya, Ming Gao, Haixiang Liu, Christopher Batty, and Eftychios Sifakis
- Venue: ACM Transactions on Graphics 36(4), 2017
- DOI: <https://doi.org/10.1145/3072959.3073625>
- Source PDF SHA-256: `00b8fde137d832162bbf828df39c34170ada3f22dd5655c4af517ba10468fead`

The author-version notice in the PDF permits personal use and says it is not
for redistribution. Confirm the repository's distribution policy before
committing the PDF or derived page renders to a public remote.

The numbered PNG files are complete 144-DPI page renders. Complete pages are
kept instead of extracting disconnected image masks so that captions, labels,
equations, and surrounding argument remain together.

## Page map

- 01: abstract and Figure 1
- 02: introduction and Figure 2
- 03: related work
- 04: method overview and Figure 3
- 05: pressure projection, power-diagram discretization, and Figure 4
- 06: sparse uniform grid pyramid and Figure 5
- 07: multigrid preconditioner and pipeline interventions
- 08: pipeline interventions and Figure 6
- 09: topology encoding and Figure 7
- 10: hierarchical operator evaluation, results, and Figure 8
- 11: timings, memory, limitations, and conclusions
- 12: references

## Reproduction

The text and page renders were generated with Poppler:

```sh
pdftotext -layout -enc UTF-8 high_resolution_liquids-a.pdf aanjaneya-2017-power-liquids.txt
pdftoppm -png -r 144 high_resolution_liquids-a.pdf page
```
