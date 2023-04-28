docker build -t fuse3d -f ./dockerized/fuse3d.Dockerfile .

#-p 8080:8080 \

docker run --rm --gpus all --name fuse3d \
-v "$(pwd)/:/home/fuse3d/" \
-v "$(pwd)/huggingface_cache/huggingface/:/home/fuser/.cache/huggingface/" \
-it --entrypoint=/bin/bash \
fuse3d
