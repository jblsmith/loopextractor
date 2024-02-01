install:
	python setup.py install

install-dev:
	python setup.py install
	pip3 install pytest

test:
	pytest tests