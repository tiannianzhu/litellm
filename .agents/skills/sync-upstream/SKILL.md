---
name: sync-upstream
description: Synchronize the LiteLLM fork with upstream/main while preserving and auditing the local patch stack. Use for checking upstream updates, rebasing local commits, resolving merge conflicts, or deciding whether upstream supersedes local fixes; do not use for ordinary feature work or unrelated repositories.
---

# Sync Upstream

Keep upstream history below a small, intentional local patch stack. Treat a clean Git merge as insufficient evidence: upstream may have replaced a local fix without producing a textual conflict.

## Establish the operation

Read the repository instructions first. Confirm the requested upstream ref, current branch, and whether the user already fetched it. Do not fetch again when the user says the ref is current. If a network fetch fails, diagnose the environment without changing proxy configuration or credentials.

Before rewriting history, record:

- old `HEAD`, upstream ref, merge base, and ordered local commit list
- local commit count and `git status --short`
- any active merge, rebase, cherry-pick, or stash
- existence and hashes, never contents, of ignored runtime inputs such as `.env`, `.env.production`, `config.yaml`, `model_prices.local.json`, and `prometheus.local.yml`

Inspect Compose bind sources and include any additional ignored operational files that a release depends on. Rebase does not protect ignored files through a stash. Back up only the relevant ignored files to a temporary directory when an in-scope operation could replace them. Do not run `git clean`, `git reset --hard`, or broad checkout commands.

Do not hide an unexplained dirty worktree. Preserve user changes and use a named safety stash only when needed for the requested workflow. Track it explicitly, restore it after the rewrite, and remove it after verification.

## Rebase the local stack

Use rebase so `upstream/main` is an ancestor of the final `HEAD` and local commits remain above it. Keep each local commit's purpose and modification surface unless upstream makes part or all of it redundant.

For each conflict:

1. Inspect the current local patch with `git rebase --show-current-patch` or `REBASE_HEAD`
2. Inspect the conflicting stages and the upstream history for the affected functions or lines
3. Recover intent from local commit messages and diffs, then from the relevant upstream commit, PR, issue, or blame history when needed
4. Classify the overlap:
   - upstream fully solves the same problem: remove the local implementation; skip the commit only when its entire intent is redundant
   - both changes are compatible: combine them with the smallest patch that preserves both intents
   - implementations are incompatible: prefer upstream for the shared problem and retain only independent local behavior
5. Stage the resolved files and continue until the whole stack is rebased

During rebase, remember that `ours` is the upstream-plus-already-replayed side and `theirs` is the current local commit. Never resolve a whole file by flag without inspecting both intents. Do not invent behavior that existed on neither side.

Ordinary conflicts are not a reason to abort. Abort when the base, revision range, worktree, or operation itself is wrong, then correct the setup before retrying.

## Audit semantic overlap

Perform this audit even if Git reports no conflicts.

- Compare the old and new local stacks with `git range-diff`
- Inspect upstream changes since the old merge base for files and functions touched by local commits
- Use `git cherry` or patch-id evidence as a hint, not proof of semantic equivalence
- Remove local code already supplied by upstream instead of retaining two implementations
- Preserve unrelated portions of a partially superseded commit and fold cleanup into that original commit when the user wants a minimal stack

Do not broaden a local commit while resolving overlap. A new upstream capability is not permission to refactor adjacent local code.

## Verify the result

Require all of the following before reporting completion:

- no Git operation remains in progress and the worktree contains no unexplained changes
- `git merge-base --is-ancestor upstream/main HEAD` succeeds
- every remaining local commit has a clear purpose; any decrease in commit count is explained by upstream supersession
- `git range-diff` shows no silently lost or duplicated local intent
- ignored runtime inputs still exist with their original hashes, or are restored from the temporary backup
- targeted checks for every conflicted or semantically changed area pass

Run the smallest relevant checks first. Use the repository's heavier quality gate only when the affected surface or project instructions justify it. Remove temporary backup refs, stashes, and files created by this workflow after successful verification.

Summarize the upstream additions, which local commits changed or disappeared, conflicts and trade-offs, checks run, final local commit count, and any remaining operational action.

## Remote history boundary

Rewriting local history does not authorize rewriting a remote default branch. Use `git push --force-with-lease`, never `--force`, only after explicit authorization for that remote branch. Without it, stop after the verified local rewrite and provide the exact push command. A lease rejection is a stop condition: inspect the new remote state instead of bypassing it.
