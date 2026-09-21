# Initial production page-domain cutover

Historical checkpoint. The later [persistent field page cutover](uniform-persistent-field-pages.md)
replaces its dense storage and Mini64 traversal specialization. Timing receipts and
legacy-oracle parity below apply to this earlier checkpoint, not the current code.

This is an incremental migration, not completion of the sparse solver described in
[the architecture design](uniform-page-domain-design.md).

Uniform Geometric now constructs a page domain for every scene. Its normal factory
cannot select the liquid window, and stale saved window parameters are discarded.
Cell and vertex kernels use page-domain addressing. Shared vertices have one owner,
including partial pages and physical upper boundaries. The 64³-and-smaller contiguous
specialization derives its dispatch from the same page generation. The older address
path is retained as an internal numerical oracle and for the separate Uniform method.

The Domain pages layer displays the accepted domain rather than whichever scratch
pages the last sharpening pass used. SIM reports domain ownership and migration
status. Transport/sharpening tile lists remain derived work selections; they do not
replace the domain generation. No page-demand readback or allocation splits an advance.

## Deliberate transitional limits

Every authored page is currently retained. Sparse activation/retirement and spatial
growth are not implemented in this cutover. Persistent textures and the pressure
hierarchy remain dense migration adapters. The pressure hierarchy is no longer sized
or moved by the liquid window, but it is not yet the intended sparse hierarchy.

This preserves the existing far-air phi history while field addressing and the finite
phi-band contract are replaced. It does not establish empty-world scaling, budgeted
sparse field allocation, or a faster garden scene. Those remain acceptance requirements,
not claims for this checkpoint. The UI states these limitations explicitly.

## Verification

- CPU ownership enumeration checks every cell and vertex exactly once, including
  reversed page order and partial pages.
- GPU comparison against the full-domain address oracle matches complete V, advected
  phi and final phi bit-for-bit through 64 advances, including liquid insertion,
  stopped/moved/restarted inflow and redistancing changes.
- Existing dense/paged scratch comparison covers live tile-map changes and insertion.
- The page overlay is encoded under a GPU validation scope.
- The Mini64 benchmark uses ABBA ordering, matched fixed pressure schedules and
  queue-fenced advances; it excludes rendering and initialization.

Commands:

```
npm run test:uniform-page-domain
npm run test:dawn:uniform-page-domain
UNIFORM_BENCH_SCENE=minimal-power-dam-break-64 npm run benchmark:uniform-page-domain
```

Keep the browser GPU unloaded while running Dawn.

### Mini64 checkpoint (2026-09-21)

The matched liquid-window/page-domain ABBA capture measured 75.206 ms versus
74.612 ms per advance, respectively: 100.8% of baseline throughput, above the
95% requirement for this checkpoint. See the
[raw matched capture](uniform-page-domain-mini64-vs-window-2026-09-21.json).
An earlier comparison with full-domain addressing in both arms measured 98.4%
of control throughput; its [raw capture](uniform-page-domain-mini64-2026-09-21.json)
is retained separately. Absolute timings varied between captures; these are local
matched measurements, not a claim of a general speedup or final sparse acceptance.

The rebuilt production UI on port 3001 was exercised with the garden hose scene.
It reports 45 resident domain pages, displays the domain overlay, advances inflow,
and publishes per-stage timings. Dense level-set and pressure work still dominate.

The next implementation stage must introduce persistent page field pools, the explicit
finite phi band and GPU frontier allocation together. The following pressure cutover
must make its complete hierarchy consume page membership. Until those are complete,
the architecture acceptance gates in the design document remain open.
