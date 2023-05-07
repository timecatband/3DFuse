FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# RUN apk update
# RUN apk add git

RUN addgroup -gid 1000 fuser && adduser -uid 1000 --gid 1000 fuser
RUN mkdir /home/fuse3d
RUN chown -R 1000:1000 /home/fuse3d 

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y git wget

WORKDIR /home/fuse3d
USER fuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /home/fuse3d/dockerized/
COPY ./dockerized/get_torch_version.py /home/fuse3d/dockerized/get_torch_version.py
RUN pip install fvcore iopath
RUN export torch_version=$(python3 dockerized/get_torch_version.py) && pip install --no-index --no-cache-dir pytorch3d -f "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${torch_version}/download.html"
RUN pip install xformers
RUN pip install pytorch-lightning==1.8.3.post0
RUN pip install git+https://github.com/arogozhnikov/einops.git

# should get weights locally instead probably, they will get copied into container below, or passed in with -v map at docker run
# RUN mkdir /home/fuse3d/weights
# WORKDIR /home/fuse3d/weights
# RUN wget https://huggingface.co/jyseo/3DFuse_weights/resolve/main/models/3DFuse_sparse_depth_injector.ckpt

# install missing deps, but dont want to have to redo the cached pip layer above :/
USER root
RUN apt-get install ffmpeg libsm6 libxext6  -y
USER fuser

#https://github.com/NVlabs/tiny-cuda-nn
ENV CUDA_HOME=/usr/local/cuda-11.8/
# see "compute capability" in this page and remove deicmal to get architecture var for cuda: https://developer.nvidia.com/cuda-gpus
ENV TCNN_CUDA_ARCHITECTURES=89
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

WORKDIR /home/fuse3d/
COPY . .

CMD ["python", "src/test-api.py"]
