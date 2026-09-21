# Uniform Geometric page-size control

The simulation pipeline's Velocity extension section exposes Page size: 16³ or
32³ (default). It is a solver-rebuild parameter: changing it resets simulation
time and reconstructs page metadata and compiled pipelines. It does not change
cell resolution. The same selection controls domain ownership, transport page
records, and the page visualization. Rectangular native execution remains enabled.

Validation:
- Parameter/domain tests: 5 passed, including default and invalid-value handling.
- Existing exact volume-page and render-overlay check: passed.
- Garden 12-frame cross-size check: V and phi bit-identical, including live
  insertion and transport/sharpening mode changes; 45 pages at 32³, 324 at 16³.
- Mini32 single-page versus multi-page check: passed with maximum differences
  0.00001460314 cell volumes and 0.000002488494 m in phi after five steps.
  This new cross-kernel test permits less than 0.0001 cell in phi and one tenth
  of the default 0.001 cell-volume dust floor. Its first exploratory volume
  bound of 0.00001 was exceeded; existing bit-exact tests were not relaxed.
- Browser: selected 16³ after advancing Garden; time reset to zero, resident
  count became 324, and a subsequent step completed. Selected 32³ again and
  verified the reset and restored count of 45 pages.
- Type checking retains the 15 existing errors outside the changed files.

The page switch changes granularity, not the unfinished allocation policy:
the authored domain still remains resident.
