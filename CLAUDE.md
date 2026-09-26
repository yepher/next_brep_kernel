@AGENTS.md

## Claude Code notes

- `AGENTS.md` (imported above) is the single source of truth for repo guidance; edit it there, not here, so other agents see the same instructions.
- Commits you create on `main` must not touch `src/`, `Cargo.toml`, `build.rs` or `README.md`; those only change through `tools/bin/sync-crate.sh` imports.
