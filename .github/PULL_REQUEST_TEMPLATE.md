<!--
Thank you for your contribution!
Please review the contributing docs: https://modin.readthedocs.io/en/latest/development/contributing.html
if you have questions about contributing.
-->

## What do these changes do?

<!-- Please give a short brief about these changes. -->

- [x] first commit message and PR title follow format outlined [here](https://modin.readthedocs.io/en/latest/development/contributing.html#commit-message-formatting)
  > **_NOTE:_**  If you edit the PR title to match this format, you need to add another commit (even if it's empty) or amend your last commit for the CI job that checks the PR title to pick up the new PR title.
- [ ] passes `flake8 modin/ asv_bench/benchmarks scripts/doc_checker.py`
- [ ] passes `black --check modin/ asv_bench/benchmarks scripts/doc_checker.py`
- [ ] signed commit with `git commit -s` <!-- you can amend your commit with a signature via `git commit -amend -s` -->
- [ ] Resolves #? <!-- issue must be created for each patch -->
- [ ] tests added and passing
- [ ] module layout described at `docs/development/architecture.rst` is up-to-date <!-- if you have added, renamed or removed files or directories please update the documentation accordingly -->
