.ONESHELL:
SHELL=bash
VERSION := $(shell cat version.txt)

default:
	make install

version:
	$(info ECOSTRESS Collection 2 pipeline version ${VERSION})

mamba:
ifeq ($(word 1,$(shell mamba --version)),mamba)
	@echo "mamba already installed"
else
	-conda deactivate; conda install -y -c conda-forge "mamba>=0.23"
endif

environment:
	make mamba
	-mamba env create -n sensitivity -f sensitivity.yml

remove:
	conda run -n base conda env remove -n sensitivity

refresh-env:
	make remove
	make environment

clean:
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
