build:
	python -m build

install:
	python setup.py install
	rm -r build

install-dev: install
	pip3 install pytest

test:
	pytest tests