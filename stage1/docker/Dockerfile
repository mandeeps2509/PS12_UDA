# ----------- Build stage -------------
# build args
ARG PY_VER="3.10"
ARG BACKEND="mps"
ARG USERNAME="docker_usr"
# ARG USER_UID="10001"
# ARG USER_GID="12345"
ARG GROUPNAME="docker_ml"

# docker base image (minimal)
FROM python:${PY_VER}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- create non-root GPU-safe user ---
# RUN groupadd --gid ${USER_GID} ${GROUPNAME} && \
#     useradd -m --uid ${USER_UID} --gid ${USER_GID} ${USERNAME}
RUN groupadd --gid 12345 docker_ml && \
    useradd -m --uid 10001 --gid 12345 docker_usr
     
# set curr working dir
WORKDIR /ml_app

# --- install System deps ---
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# change ownership to current user
# RUN chown -R ${USERNAME}:${GROUPNAME} /ml_app
RUN chown -R docker_usr:docker_ml /ml_app
# USER ${USERNAME}:${GROUPNAME}
# switch to custom non-root user
USER docker_usr

# print UID & GID
CMD sh -c "echo '>> In container (after sys dependencies install):' && echo 'User: $(whoami) UID: $(id -u) GID: $(id -g)'"

# copy dependencies into image
COPY requirements.txt /ml_app

# install py dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy application code into image
COPY . /ml_app
COPY --chmod=755 docker/entrypoint.sh /ml_app

# set permissions for script
# RUN chmod +x /ml_app/entrypoint.sh
# RUN chown -R ${USER_UID}:${USER_GID} /ml_app

# set entry point
ENTRYPOINT ["bash", "/ml_app/entrypoint.sh"]
