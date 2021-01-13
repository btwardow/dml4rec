NAME := $(USER)_docker
TAG := $(shell git describe --long --dirty=_dev)
IMG := $(NAME):$(TAG)
LATEST := $(NAME):latest
GPU := $(if $(GPU),$(GPU),all)

USER_PATH=$(shell dirname $(shell pwd))
DOCKER_PATH=/dml4rec

.PHONY: build start run

build:
	docker build -t $(IMG) -f Dockerfile .
	docker tag $(IMG) $(LATEST)

run:
	docker run --gpus device=$(GPU) --rm -v $(USER_PATH):$(DOCKER_PATH) -it $(LATEST)

default: build
