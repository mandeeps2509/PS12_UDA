# ----------- Build stage -------------
ARG PY_VER="3.10"

# docker base image (minimal)
FROM python:${PY_VER}-slim AS base

# build args
ARG TORCHCODEC_URL="https://github.com/meta-pytorch/torchcodec/archive/refs/tags/v0.7.0.tar.gz"
ARG BACKEND="mps"
ARG USERNAME="docker_usr"
ARG USER_UID="10001"
ARG USER_GID="12345"
ARG GROUPNAME="docker_ml"

# labels for meta-data
LABEL description="AIGC25: Underwater Domain Awareness"
LABEL version="1.0"
LABEL author="Mandeep Singh"
LABEL mail="mandeeps2509@gmail.com"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- create non-root GPU-safe user ---
# RUN groupadd --gid ${USER_GID} ${GROUPNAME} && \
#     useradd -m --uid ${USER_UID} --gid ${USER_GID} ${USERNAME}
RUN groupadd --gid ${USER_GID} ${GROUPNAME} && \
    useradd -m --uid ${USER_UID} --gid ${USER_GID} ${USERNAME}
     
# set curr working dir
WORKDIR /ml_app

# --- install System deps ---
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    cmake \
    git \
    wget && \
    rm -rf /var/lib/apt/lists/*

# create libs dir for ffmpeg & torchcodec
RUN mkdir /ml_app/libs

# RUN add-apt-repository ppa:ubuntuhandbook1/ffmpeg6
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ffmpeg<8 \
    # libavdevice-dev \
    # libavutil-dev \
    # libavutil58 \
    python3-dev 
# RUN git clone https://git.ffmpeg.org/ffmpeg.git /ml_app/libs/ffmpeg
RUN wget https://www.ffmpeg.org/releases/ffmpeg-7.0.1.tar.gz -O /ml_app/libs/ffmpeg.tar.gz && \
    tar -xvzf /ml_app/libs/ffmpeg.tar.gz -C /ml_app/libs && \
    rm /ml_app/libs/ffmpeg.tar.gz && \
    /ml_app/libs/ffmpeg-7.0.1/configure --enable-cross-compile --arch=aarch64 \
    --enable-shared --enable-pthreads --disable-static --install-name-dir='/ml_app/libs/ffmpeg-7.0.1/lib' && \
    make install
ENV DYLD_FALLBACK_LIBRARY_PATH=/ml_app/libs/ffmpeg-7.0.1/lib
ENV PATH="$PATH:/ml_app/libs/ffmpeg-7.0.1/lib"
RUN bash -c "source ~/.bashrc" && \
    ldconfig && \
    ffmpeg -version

# change ownership to current user
# RUN chown -R ${USERNAME}:${GROUPNAME} /ml_app
RUN chown -R ${USERNAME}:${GROUPNAME} /ml_app
# RUN chown -R ${USERNAME}:${GROUPNAME} /opt/torchcodec
# USER ${USERNAME}:${GROUPNAME}
# switch to custom non-root user
USER ${USERNAME}

# print UID & GID
RUN echo ">> In container (after sys dependencies install):" && \
    echo "User: $(whoami) UID: $(id -u) GID: $(id -g)"

# copy dependencies into image
COPY requirements-core.txt /ml_app
COPY requirements-extend.txt /ml_app

# install py dependencies
# RUN if ["$PY_VER" = "3.9"]; then \
# else \
#     pip install --no-cache-dir torchcodec; \
# fi
# install core req
RUN pip install --no-cache-dir -r requirements-core.txt --no-deps && \
    python -c "import torch; print('torch.ver:', torch.__version__); import ffmpeg; print('imported ffmpeg:');"

## ------- Stage1.2: extended env stage --------
FROM base AS extended

# USER root

# ENV for torchcodec to be built for installed ffmpeg only
# ENV BUILD_AGAINST_ALL_FFMPEG_FROM_S3=0 \
#     BUILD_AGAINST_ALL_FFMPEG_FROM_S3_VAR=0
# RUN echo "BUILD_AGAINST_ALL_FFMPEG_FROM_S3: $BUILD_AGAINST_ALL_FFMPEG_FROM_S3" && \
#     echo "BUILD_AGAINST_ALL_FFMPEG_FROM_S3_VAR: $BUILD_AGAINST_ALL_FFMPEG_FROM_S3_VAR"
ENV NOT_A_LICENSE_VIOLATION_VAR=I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION
RUN echo "NOT_A_LICENSE_VIOLATION_VAR: $NOT_A_LICENSE_VIOLATION_VAR"
RUN pip install findpython \
    pybind11[global] \
    setuptools
ENV PATH="$PATH:/ml_app/libs"
ENV PATH="$PATH:/home/$USERNAME/.local"
ENV PATH="$PATH:/home/$USERNAME/.local/bin"
ENV PATH="$PATH:/home/$USERNAME/.local/lib"
RUN bash -c "source ~/.bashrc" && \
    echo "$PATH"
# get torchcodec source
RUN wget ${TORCHCODEC_URL} -O /ml_app/libs/torchcodec.tar.gz && \
    tar -xvzf /ml_app/libs/torchcodec.tar.gz -C /ml_app/libs && \
    rm /ml_app/libs/torchcodec.tar.gz && \
# RUN make clean /ml_app/libs/torchcodec-0.7.0
    # pip install /ml_app/libs/torchcodec-0.7.0
    pip install -e /ml_app/libs/torchcodec-0.7.0 --no-build-isolation
# RUN pip install torchcodec==0.5

# USER ${USERNAME}

# install extended req
RUN pip install --no-cache-dir -r requirements-extend.txt --no-deps

# copy application code & req torch models into image
RUN mkdir -p /ml_app/src
COPY --chmod=755 src/*.py /ml_app/src
COPY --chmod=755 src/trained_ast_clf /ml_app/src/trained_ast_clf
COPY --chmod=755 src/trained_ast_clf_lora /ml_app/src/trained_ast_clf_lora
COPY --chmod=755 docker/entrypoint.sh /ml_app

# set entry point
ENTRYPOINT ["bash", "/ml_app/entrypoint.sh"]
