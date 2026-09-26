# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this repo is

A **read-mostly mirror of the published `BREP_kernel` crate** (library name
`brep_kernel`): an exact NURBS / manifold B-rep CAD geometry kernel in Rust for
native and WASM. Each commit titled `Version X https://crates.io/crates/BREP_kernel/X`
is an unmodified crates.io release, imported by `tools/bin/sync-crate.sh`.

Consequences:

- **Do not hand-edit `src/`, `Cargo.toml`, `build.rs` or `README.md`.** The next
  import deletes and replaces every tracked file except the local-only ones
  below. Upstream development happens in a separate monorepo.
- **Local-only files** (preserved across imports): `tools/`, `AGENTS.md`,
  `CLAUDE.md`, `.devin/`. If you add another repo-local file, add it to the
  `keep` list in `tools/bin/sync-crate.sh`.
- **Not in this repo:** tests and test data (excluded from the published crate),
  the design documents comments cite (`per-entity-tolerances.md`,
  `step-assembly-import.md`, `fillet-stripe-network.md`, …), and the
  `BREP_app` / `BREP_render` / umbrella `BREP` crates the README mentions. Do
  not guess at their contents.

## Commands

```sh
cargo check                        # type-check (~25s cold; ~23 warnings are expected upstream noise)
cargo build --features parallel    # rayon parallelism; `wasm-threads` = parallel + wasm-bindgen-rayon
cargo doc --no-deps --open         # browse the public API

tools/bin/sync-crate.sh --dry-run  # list newer crates.io releases
tools/bin/sync-crate.sh            # import them, one commit per version (needs clean tree, on main)
tools/bin/sync-crate.sh 0.6.0      # import a specific version

git diff <prev-version-commit> <version-commit> --stat   # what changed between releases
```

There is no test suite here; `cargo test` compiles but runs nothing meaningful.

## Typical tasks

- **Summarize a release:** diff consecutive `Version …` commits. Use the
  `//!` module headers of added/changed files and the `pub use` diff in
  `src/lib.rs` (the public API surface) rather than reading every line.
- **Explain code:** start from `src/lib.rs` re-exports, then the module's `//!`
  header, which in this codebase usually states the contract precisely.
- **Maintain the wiki:** `.devin/wiki.json` steers DeepWiki, which is the crate's
  published manual (linked from README.md). Limits: 30 pages, 100 notes,
  10,000 chars per note; only listed pages are generated.

## Code map

`src/lib.rs` mounts files from topical directories as flat modules via
`#[path = "..."]`, so `mod fit;` may live at `src/geometry/fit.rs`. **The public
API is exactly what `lib.rs` re-exports**; `#[doc(hidden)]` items are
diagnostic hooks, not API.

| Directory | Contents |
|---|---|
| `geometry/` | NURBS curves/surfaces, fitting, pcurves, projection, analytic carriers, Coons/Gordon surfaces, tolerances |
| `intersect/` | curve/curve, curve/surface, marched surface/surface intersection; arrangements |
| `brep/` | topology records + arena, transforms, codec, point/face classification, soundness checks |
| `construction/` | extrude, revolve, loft, sweep (mitres, twist, envelopes, ribs) |
| `csg/` | booleans: imprint → fragment → oracle → select → assemble |
| `blending/` | fillets, chamfers, radius laws, blend marching, corners, blend networks |
| `edit/` | direct edit: delete-and-heal, face move/rotate/offset, split |
| `offset/` | offset surfaces, offset regularity / fold locus / carve, offset shell, thicken |
| `healing/` | sew, coalesce, face merge, faceted repair, `accept_sound` repair-or-refuse |
| `meshing/` | watertight tessellation, mesh weld, mesh segmentation → B-rep |
| `props/` | mass properties (adaptive Gauss–Legendre), diagnostics |
| `solvers/` | sketch constraint solver, assembly solver |
| `io/` | STEP import/export (+PMI, colours, assemblies), IGES, 3MF, GLB, mesh I/O, snapshots |
| `feature_pipeline/` | JSON feature-history engine, feature catalog, sketches, sheet metal, assemblies, PMI, ports/pins, parts library, wire harness |
| `abi/` | WASM-facing ABI: handles, typed-array transfers, requests |

`build.rs` sets `BREP_KERNEL_SOURCE_HASH` (hash of `src/**/*.rs`), which the
parts library uses to invalidate cached geometry.

## Vocabulary

These terms recur throughout and are used precisely:

- **carrier** — the untrimmed surface a face lies on; a face = carrier + trim loops.
- **analytic carrier** — a NURBS surface recognized as plane/cylinder/cone/sphere/torus/revolution, enabling closed-form projection and intersection.
- **pcurve** — an edge's curve in a face's (u, v) parameter space.
- **lane** — one of several alternative strategies an operation picks per input.
- **refusal** — a deliberate typed failure. Many have exported string-prefix constants (`*_REFUSAL`) so callers classify without matching free text. The kernel prefers refusing to returning an unsound solid.
- **report / ledger / `*_reported`** — variants returning diagnostics with the result.
- **census** — diagnostic enumeration of the cases a lane met.
- **`*_SWITCH`** — named toggles for alternative classification behaviour.
- **validate vs soundness** — `BrepSolid::validate()` checks incidence only. `brep/soundness.rs` (self-intersection, connectivity, Euler, closure) and `healing/accept.rs` (repair or refuse) are the stronger checks.
