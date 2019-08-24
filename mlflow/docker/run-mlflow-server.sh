#!/usr/bin/env sh

PORT=${PORT:-5000}
CURRENT_DIRECTORY="$(readlink -f ${BASH_SOURCE[0]})"
echo $CURRENT_DIRECTORY

# docker run \
# -d \
# --rm \
# -v /home/rodion_martynov/projects/ml-eng/docker/mlflow-storage:/mnt/mlflow \
# --name martynov-mlflow \
# -p "${PORT}":"${PORT}" \
# --group-add 5000 \
# --user $UID:$(cut -d: -f3 < <(getent group rm_volume)) \
# rmartynov/mlflow-server \
# $1