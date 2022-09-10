SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

pycheck:
	-pylint grheat/point_source.py
	-pylint grheat/line_source.py
	-pylint grheat/plane_source.py
	-pylint grheat/__init__.py

doccheck:
	-pydocstyle grheat/point_source.py
	-pydocstyle grheat/line_source.py
	-pydocstyle grheat/plane_source.py
	-pydocstyle grheat/__init__.py

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	open docs/_build/index.html

clean:
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf grheat.egg-info
	rm -rf grheat/__pycache__
	rm -rf docs/_build
	rm -rf docs/api
	rm -rf tests/__pycache__
	rm -rf .tox
	rm -rf build
	rm -rf 

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
	tox

test:
	python3 -m pytest tests/test_point.py

.PHONY: clean check rcheck html