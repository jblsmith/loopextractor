build:
	python -m build

install:
	python setup.py install
	pip install git+https://github.com/CPJKU/madmom
	rm -r build

install-dev: install
	pip3 install pytest

test:
	pytest tests