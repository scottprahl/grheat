SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

lint:
	-pylint grheat/__init__.py
	-pylint grheat/point_source.py
	-pylint grheat/line_source.py
	-pylint grheat/plane_source.py
	-pylint grheat/absorber.py
	-pylint grheat/exp_source.py
	-pylint tests/test_point.py
	-pylint tests/test_line.py
	-pylint tests/test_plane.py
	-pylint tests/test_absorber.py

doccheck:
	-pydocstyle grheat/point_source.py
	-pydocstyle grheat/line_source.py
	-pydocstyle grheat/plane_source.py
	-pydocstyle grheat/absorber.py
	-pydocstyle grheat/__init__.py

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	open docs/_build/index.html

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf build
	rm -rf dist
	rm -rf grheat.egg-info
	rm -rf docs/.ipynb_checkpoints
	rm -rf docs/_build
	rm -rf docs/api
	rm -rf grheat/__pycache__
	rm -rf tests/__pycache__

notecheck:
	pytest --verbose tests/test_all_notebooks.py

rcheck:
	make clean
	make lint
	make doccheck
	make test
	make html
	check-manifest
	pyroma -d .
	make notecheck

test:
	pytest --verbose tests/test_point.py
	pytest --verbose tests/test_line.py
	pytest --verbose tests/test_plane.py
	pytest --verbose tests/test_absorber.py

.PHONY: clean check rcheck html