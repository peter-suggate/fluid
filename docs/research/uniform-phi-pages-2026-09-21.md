# Phi page experiment: not promoted

The experimental page-driven advection/redistancing code was removed from the
working implementation after the longer moved-inlet test diverged. The dense
far-air phi history requires an explicit replacement contract before this can
become a sparse page-domain operator. Fixing the independent census-summary
index race (6671ce77) did not resolve that numerical difference.

Redistancing alone also failed to improve timings sufficiently; no opt-in phi
mode was shipped. The earlier timings in the raw experiment capture are research
observations, not production performance claims.

The current architecture decision is documented in
[Uniform page-domain design](uniform-page-domain-design.md): active pages replace
the liquid window as the sole computational-domain authority.
