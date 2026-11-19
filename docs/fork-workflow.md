# Fork development workflow

This checkout develops local editor improvements in the
[ArchAerialData fork](https://github.com/ArchAerialData/splat-repair).
The [upstream contribution guide](../CONTRIBUTING.md) remains authoritative.
No pull request is created as part of this setup.

## Remotes and branches

- `origin`: `https://github.com/ArchAerialData/splat-repair.git` (default push target).
- `upstream`: `https://github.com/NicolasDrum/splat-repair.git` (fetch upstream changes).
- `main`: upstream baseline; do not develop directly here.
- `codex/editor-refinements`: integrated, reviewed editor changes, split into topic commits.
- `codex/classification`: classification work built on the reviewed baseline.

Use the current checkout unless isolation is needed. Keep customer scans, local
editor settings, profiles, generated screenshots and test output out of commits.
The local `.vscode` folder is excluded through `.git/info/exclude`, not deleted.

## Before publishing changes

Review `git status` and `git diff`; stage explicit paths. Run the engine suite,
TypeScript, production build and relevant smoke tests (see the audit record).
Push explicitly to `origin`. A push to the fork is not a request to open a PR.

For a future upstream contribution, create a fresh topic branch from the then
current `upstream/main` and cherry-pick only that topic's commits plus its required
dependencies. Review the resulting diff; the integrated editor branch is not a
single-topic upstream PR. Re-run checks after resolving any conflicts. Explain
the behavior and verification in the PR; changes to fill algorithms require
before/after pictures as specified by upstream. Only share authorized fixtures.

Do not force-push shared branches or reset local work to update from upstream.
Fetch first and inspect divergence; merge/rebase deliberately on a topic branch.
