To run GPU in container, you need:
1. nvidia and cuda drivers on host
2. nvidia-container-toolkit package 

On host, running:
`nvidia-smi` 
Should give output including line similar to this with cuda version:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+


Then to install the container toolkit, follow instructions here: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

NOTE: When following above I had to set distribution to supported ubuntu18.04:
  e.g: distribution="ubuntu18.04" && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list |             sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |             sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

Then you can test that the container env can see the GPU using this nvidia docker image and running the same nvidia-smi inside:
`docker run --rm --gpus all nvidia/cuda:12.0.0-devel-ubuntu22.04 nvidia-smi`