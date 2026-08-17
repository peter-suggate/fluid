# Sparse CM12 FPL1 FramePlan

FPL1 is the compact GPU-owned scheduling authority for the B16/P16 path. It is
double-buffered as `Current` and `Next`. The CPU uploads external controls such
as dt and forces; accepted frame generation and ping-pong parity are committed
by the GPU header.

The native execution shape is one 64-lane workgroup per physical B16 brick and
one lane per 4³ tile. B8 uses lanes 0–7 and B4 uses lane 0 without changing the
ABI or shader architecture.

## Layout

The 256-byte global header contains shape validation, Current/Previous slot
indices, accepted/candidate plan and topology generations, GPU frame
generation/parity, global transition faults, slot bases, packet census, and a
tightly packed fixed range of six indirect triplets. Finalize publishes the
accepted slot's triplets into that fixed range, so one device-side copy updates
the indirect-only buffer without the host choosing Current or reading parity.

Each slot contains:

- a 128-byte generation/fault header with up to six physical packet counters
  and indirect triplets;
- one 128-byte brick record per physical brick;
- one 16-byte record per 4³ tile;
- one deterministic brick-indexed list per physical packet.

The tile record is exactly four words:

```text
0  generation
1  direct[6] | closure[6] | executed[6] | skipped[6] | uncovered[6]
2  origin causes[16] | inherited causes[16]
3  six 4-bit closure depths | reserved[8]
```

The brick record owns six 64-bit logical-stage masks, physical packet
assignment for all six stages, aggregate cause/depth receipts, producer and
consumer generations, valid-tile mask, and the local fault identity. Multiple
logical stages may name the same physical packet, which is the authoritative
pass-packing overlay.

## Scheduling

`beginSparseCM12FramePlanNext` opens the alternate slot. One dispatch of
`initializeSparseCM12FramePlanNext` gives every tile a unique lane owner.
Producers call `cm12FramePlanMarkOwnedNextTile`; cross-brick closure writes only
bounded neighbor brick masks, consumed by
`resolveSparseCM12FramePlanNextClosure` in fixed rounds.

`sealSparseCM12FramePlanNextBricks` verifies tile generation and mask coverage.
It writes deterministic packet-list slot `packet * brickCapacity + brick`; it
does not append tiles or reorder bricks. `finalizeSparseCM12FramePlanNext`
authors fixed indirect packet headers and flips Current only when global shape,
generation, and coverage are complete. A local brick fault leaves that brick's
packet-list entries invalid and publishes its fault to the native overlay;
unrelated bricks proceed. A global ABI or transition fault retains Current.

Consumers dispatch physical packets, then use the same brick masks to execute
lane-owned logical stages. They mark executed or certified-skipped bits in the
same tile record. The stage-specialized verification pipeline latches an
uncovered-write fault when neither receipt exists.

## Observability

`SparseCM12FramePlanSource` exposes the arena and copy-isolated indirect packet
snapshot. `cm12FramePlanOverlayTileAt` returns the exact scheduling record:
logical stages, physical packet assignment, direct/closure class, causes,
closure depths, generation, executed/skipped coverage, and local fault. Missing
or mismatched generation is unknown/magenta. There is no second overlay-only
dirty reconstruction and no per-tile global atomic journal.
