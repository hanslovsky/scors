.PHONY: develop
develop: ## Build and install the extension in-place (debug, fast iteration)
	@uv run maturin develop

.PHONY: develop-release
develop-release: ## Build and install the extension in-place (release, for benchmarks)
	@uv run maturin develop --release

.PHONY: install
install: ## Install all dependencies
	@uv sync

.PHONY: test
test: ## Run the test suite
	@uv run pytest

.PHONY: bench
bench: develop-release ## Build release extension and run all benchmarks
	@uv run python tests/benchmarks/bench_loo_cossim.py
	@uv run python tests/benchmarks/bench_ap_auroc.py

.PHONY: build
build: ## Build wheels for all interpreters found by maturin
	@uv run maturin build --release

.PHONY: lock
lock: ## Regenerate uv.lock from current pyproject.toml constraints
	@uv lock

.PHONY: release
release: ## Verify versions match, no .dev suffix, then print release steps
	@python verify-version.py
	@python -c "\
import tomllib; \
v = tomllib.load(open('pyproject.toml','rb'))['project']['version']; \
(print(f'ERROR: version {v!r} contains .dev — bump to a release version first') or exit(1)) \
if '.dev' in v else (\
print(f'Version {v!r} looks good.'), \
print(), \
print('Steps:'), \
print(f'  git add pyproject.toml rust/Cargo.toml'), \
print(f'  git commit -m \"Release version {v}\"'), \
print(f'  git tag v{v}'), \
print(f'  git push origin v{v}'), \
print(f'  git push origin main'), \
)"

.PHONY: docs
docs: ## Build HTML documentation with Sphinx
	@uv run sphinx-build -b html docs public

.PHONY: clean-docs
clean-docs: ## Remove built documentation
	@rm -rf public docs/generated

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
