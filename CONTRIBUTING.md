# Contributing to mussel-nf

Thank you for your interest in contributing!

## Reporting bugs

Please open a [GitHub issue](https://github.com/pathology-data-mining/mussel-nf/issues/new/choose) using the **Bug Report** template. Include:
- Nextflow version (`nextflow -v`)
- Execution profile (Docker / Apptainer / Conda)
- Minimal `samples_csv` and command used to reproduce
- Relevant log output or error message

## Requesting features

Open an issue using the **Feature Request** template. Describe the use case and why the current pipeline doesn't address it.

## Submitting pull requests

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Keep commits focused and use [conventional commit](https://www.conventionalcommits.org/) messages (`feat:`, `fix:`, `docs:`, etc.).
3. Add or update tests where applicable:
   - Nextflow stub tests live in `tests/` and use [nf-test](https://www.nf-test.com/)
   - Python unit tests for the dispatcher live in `dispatcher/tests/`
4. Run tests locally before opening a PR:
   ```bash
   # nf-test (pipeline)
   nf-test test tests/

   # pytest (dispatcher)
   cd dispatcher && pytest
   ```
5. Open a pull request against `main`. The CI workflow will run automatically.

## Code style

- Nextflow: follow DSL2 conventions; use process labels from `nextflow.config` for resource allocation.
- Python: standard formatting (`black`-compatible); type hints encouraged.
- No secrets or institution-specific paths in committed code.

## Questions

For general questions about usage, open a [Discussion](https://github.com/pathology-data-mining/mussel-nf/discussions) rather than an issue.
