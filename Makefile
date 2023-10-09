SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

lint:
	-pylint grheat/point_source.py
	-pylint grheat/line_source.py
	-pylint grheat/plane_source.py
	-pylint grheat/absorber.py
	-pylint grheat/__init__.py
	-pylint tests/test_point.py

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
	make clean
	pytest --verbose test_all_notebooks.py
	rm -rf __pycache__

rcheck:
	make clean
	make pycheck
	make doccheck
	make notecheck
	touch docs/*ipynb
	touch docs/*rst
	make html
	check-manifest
	pyroma -d .

test:
	pytest tests/test_point.py
	pytest tests/test_line.py
	pytest tests/test_plane.py
	pytest tests/test_absorber.py

.PHONY: clean check rcheck html