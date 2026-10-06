# Withdrawn and superseded Phase 3 prototypes

Each patch is independently based on commit `11ea9c04`. These are research
artifacts, not active production changes. Do not stack them on each other.
Measured variants were checked against the Uniform source hashes captured by
their benchmark reports. Test and diagnostic plumbing is recorded separately
in the parent analysis; these patches contain the production-source prototypes.

| Patch | Disposition |
| --- | --- |
| `local-a.patch` | Local search queue with tile leader election; slower |
| `block-a.patch` | Block list published during surface preparation; slower |
| `brick-a.patch` | Branched compact brick addressing; numerical parity failure |
| `brick-b.patch` | Branchless compact brick addressing; exact, slower |
| `slab-a.patch` | Contiguous x rows in y/z slabs; exact, slower |
| `initial-sparse.patch` | Sparse integer first-gradient arithmetic; parity failure, never enabled |
| `initial-weighted.patch` | Reused integer first-gradient samples; parity failure, never enabled |
| `split-a.patch` | Three extension sweep launches; whole-run overhead |
| `split-b.patch` | Two launches before specializing the fine staging footprint; superseded |

The retained implementation is the two-way extension split with one staged
neighborhood for fine seams. See Phase 3 in the parent analysis for measurements,
validation and numerical restrictions.
