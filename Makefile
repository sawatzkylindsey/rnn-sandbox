.PHONY: test report clean build

default: build

test:
	coverage run --append --source=nnwd,ml -m unittest test.all

report:
	coverage report -m

clean:
	rm -rf build
	rm -rf dist
	rm -rf pytils.egg-info
	find . -name "*.pyc" -delete
	coverage erase

build:
	python dev-server.py

