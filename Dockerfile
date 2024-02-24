# nvidia container 

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as base

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3.10 python3-pip git wget curl build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install ffmpeg
RUN wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz &&\
    wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz.md5 &&\
    md5sum -c ffmpeg-git-amd64-static.tar.xz.md5 &&\
    tar xvf ffmpeg-git-amd64-static.tar.xz &&\
    mv ffmpeg-git-*-static/ffprobe ffmpeg-git-*-static/ffmpeg /usr/local/bin/ &&\
    rm -rf ffmpeg-git-*

RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

COPY . .

RUN pip install -e .

ENTRYPOINT ["python3", "whisper_api.py"]

# sudo docker build -t whisper-api .
# sudo docker rm whisper-api
# sudo docker run -d -it --gpus '"device=0"' -p 8777:8777 --name whisper-api whisper-api
# sudo docker stop whisper-api
