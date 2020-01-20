#!/bin/bash
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

set -e

TF_RESEARCH=/home/jovyan/work/ds-course-adv/cv/a3/models/research
export PYTHONPATH=${PYTHONPATH}:${TF_RESEARCH}:${TF_RESEARCH}/slim

wrapper=""
if [[ "${RESTARTABLE}" == "yes" ]]; then
  wrapper="run-one-constantly"
fi

if [[ ! -z "${JUPYTERHUB_API_TOKEN}" ]]; then
  # launched by JupyterHub, use single-user entrypoint
  exec /usr/local/bin/start-singleuser.sh "$@"
elif [[ ! -z "${JUPYTER_ENABLE_LAB}" ]]; then
  . /usr/local/bin/start.sh $wrapper jupyter lab "$@"
else
  . /usr/local/bin/start.sh $wrapper jupyter notebook "$@"
fi