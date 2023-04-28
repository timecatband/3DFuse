docker build -t fuse3d -f ./dockerized/Dockerfile .

#-p 8080:8080 \

docker run --rm --gpus all --name fuse3d \
-v "$(pwd)/:/home/fuse3d/" \
-it --entrypoint=/bin/bash \
fuse3d
