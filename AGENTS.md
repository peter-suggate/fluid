# Repository guidance

Checks: `npm run check:types` and `npm run test:unit`.

Scene authoring contract:
- Every visible terrain, scenery and voxel surface must support nearest-visible-surface hit testing and editing through the shared geometry path. Do not add render-only geometry that editing cannot address.
- Edits must retain the source geometry, materials and detail outside the edited region. Never replace an entire authored model with coarse voxels after a local cut.
- Where a surface also has collision geometry, the same edit must open both surfaces without exposing a duplicate collision mesh.
- New geometry representations need regressions for hit testing, local edits, untouched visual fidelity, and save/load/undo before scenes may use them. Unknown kinds must fail explicitly.
