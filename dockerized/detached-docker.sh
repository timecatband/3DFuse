#docker build -t fuse3d-cuda11-8 -f ./dockerized/fuse3d.Dockerfile .

#-p 8080:8080 \
#-it --entrypoint=/bin/bash \
docker run --rm --gpus all --name fuse3d \
-v "$(pwd)/:/home/fuse3d/" \
-v "$(pwd)/huggingface_cache/huggingface/:/home/fuser/.cache/huggingface/" \
-d --entrypoint="/bin/bash" \
fuse3d-cuda11-8 \
"/home/fuse3d/dockerized/fuse-prompt.sh"
