from nvcr.io/nvidia/cuda:11.2.2-base-ubuntu20.04

WORKDIR /project

RUN apt-get update && apt-get upgrade -y && apt-get install -y sudo

RUN sudo chown -Rv _apt:root /var/cache/apt/archives/partial/ && \
    sudo chmod -Rv 700 /var/cache/apt/archives/partial/

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6 git && \
                       nvidia-cuda-toolkit nvidia-container-toolkit

RUN python3 -m pip install numpy matplotlib scikit-learn scikit-image opencv-python opencv-contrib-python jupyterlab 

#PyTorch installation for CUDA 11.1
RUN python3 -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python3 -m pip install -e detectron2

EXPOSE 8822

CMD ["sh", "-c", "jupyter-lab --port=8822 --no-browser --allow-root --ip=0.0.0.0"]
