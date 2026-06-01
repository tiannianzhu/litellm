---
name: sync-upstream
description: Synchronize the LiteLLM fork with upstream/main while preserving and auditing the local patch stack. Use for checking upstream updates, rebasing local commits, resolving merge conflicts, deciding whether upstream supersedes local fixes, or improving this synchronization workflow; do not use for ordinary feature work or unrelated repositories
---

# Sync Upstream

Rebase upstream beneath the local patch stack. Audit semantic overlap even when Git reports no conflicts

## Prepare

Follow repository instructions and pull the current branch. Fetch upstream once when needed, then record fixed IDs: old head, old merge base, upstream target, ordered local commits, and the remote branch head for an exact push lease

Preserve user changes with a named stash when necessary and track its restoration. Record existence and hashes, never contents, of ignored runtime inputs such as `.env`, `.env.production`, `config.yaml`, `model_prices.local.json`, and `prometheus.local.yml`; inspect Compose binds for additional inputs. Back up only files the operation could replace. Avoid broad clean, reset, or checkout commands

Decide commit ownership before editing, following the user's requested grouping. Keep changes small, gather corrections, and fold them into their original commits once. Do not leave `fixup!` commits or add automation for routine Git operations

## Rebase and audit

Inspect the current patch, conflicting stages, and upstream intent. During rebase, `ours` is upstream plus replayed patches; `theirs` is the current local commit. Combine compatible behavior, prefer upstream where it supersedes a local fix, and skip a commit only when its entire purpose is redundant

Compare local changes from `old_base..old_head` with upstream changes from `old_base..new_upstream`, starting with overlapping functions and callers. Code merely present in the old checkout is not necessarily local. Patch IDs are hints, not semantic proof. Preserve unrelated intent and avoid adjacent refactors

Review `git range-diff old_base..old_head new_upstream..HEAD`, local commit purposes and count, upstream ancestry, pending Git operations, worktree status, and runtime hashes

## Validate efficiently

Select regressions from changed upstream functions, conflict resolutions, and affected local contracts and callers. Prefer owning test cases or parameter groups over whole files selected merely because the cumulative patch stack touched them. Expand to full files or suites when shared fixtures or broad behavior changes require it, and record the coverage rationale

Inspect the selected tests' imports, fixtures, and dependency groups before starting. In a fresh worktree, explicitly choose a supported test interpreter and install required extras/groups from the frozen lock, including optional SDKs. Prepare the Rust extension and Prisma client before tests use them. Finish environment provisioning before launching concurrent checks

Use `LITELLM_LOCAL_MODEL_COST_MAP=True` except for remote price-map contracts. Once the source tree is finalized and environments are ready, launch independent regressions and the required complete gate together. Run independent test files or coherent groups in a bounded number of separate processes, honoring `no_parallel`, shared-service fixtures, and machine-wide gate slots. Wait only for actual dependencies between commands

For an upstream rebase, run the required complete gate with nothing staged; staged checks cover only staged changes. Workflow-only changes need their affected checks

Include `--durations=20 --durations-min=1` in pytest runs and save their output. Record each command's start/end timestamps and elapsed time so execution, setup, diagnosis, and reporting delays can be distinguished. `make check` buffers child output; inspect saved logs and process/slot status during quiet periods instead of treating silence as a hang

The type gate may need an extra full baseline scan when the base or environment fingerprint changes and no matching CI artifact exists. Preserve its shared base-count cache and let the gate compute a missing baseline once. Diagnose through its owned environment and budget policy, never lower budgets to pass

On failure, inspect the relevant log and rerun failing cases or the necessary predecessor group in a fresh process to distinguish missing dependencies, source regressions, and order-dependent state. Broaden only for changed code, new failures, or unresolved concerns. An isolated pass does not resolve an order-dependent failure; report the failing combined run and the passing scope accurately

For database regressions, the test PostgreSQL service is localhost:5432; confirm authentication and a test-only `DATABASE_URL` without printing credentials. Missing `pg_config` does not rule out external PostgreSQL

Use `tests/_support/postgresql.py` fixtures for isolated schemas and authenticated psycopg/Prisma URLs. Load the confirmed connection with `uv run --env-file .env --no-sync pytest <paths>`. Preserve fresh-database/server isolation contracts. Keep pure tests in `tests/unit` and real database tests in `tests/integration/database`

## Finish

Record the tested tree, comparison base, relevant environment inputs, and command timings and logs. Reuse results after a pure history rewrite only when these inputs and relevant external state match. Read saved logs instead of rerunning checks for different output. The type gate saves raw head diagnostics at Git's `basedpyright-diagnostics.json` path; preserve it before another run overwrites it

Respect gate queues and use the repository's sandbox retry policy. Restore user changes and remove only workflow-owned temporary backups, refs, and stashes after verification

Reuse existing authorization to commit and push. For an authorized rewrite, use `git push <remote> --force-with-lease=refs/heads/<branch>:<recorded_remote_head>`; inspect changed remote state on lease rejection. Otherwise finish local verification and provide the command. Report upstream changes, local patch changes and count, checks with their scope, and any remaining action
