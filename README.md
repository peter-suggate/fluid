# Fluid Lab

Fluid Lab combines a narrow-band FLIP simulation, an SVO scene renderer, a plugin architecture, and an interactive editing UI. The independent advance lab runs its two-dimensional solver through Rust/Wasm.

## Run locally

```sh
npm install
npm run build:physics-wasm
npm run dev
```

## Checks

```sh
npm run check:types
npm run test:unit
```

`npm run build` builds the physics modules and production application. See [AGENTS.md](AGENTS.md) for repository guidance and [the purge plan](docs/plans/narrow-band-flip-only-purge-2026-10-11.md) for the cleanup scope.
