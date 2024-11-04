IMAGE_NAME := tenyx/wip
IMAGE_TAG := 2.4.0

docker/build:
	@mkdir -p build
	@cp vendor/text-generation-inference*.zip build/tgi.zip
	cd build/; \
	unzip -u tgi.zip; \
	rm tgi.zip; \
	cd text-generation-inference-*/; \
	docker build --tag $(IMAGE_NAME):$(IMAGE_TAG) --file ../../Dockerfile .

docker/run:
	@model=teknium/OpenHermes-2.5-Mistral-7B; \
    volume=$(PWD)/data; \
    docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
        $(IMAGE_NAME):$(IMAGE_TAG) \
        --model-id $model

clean:
	@-rm -r build/

.PHONY: docker/build docker/run