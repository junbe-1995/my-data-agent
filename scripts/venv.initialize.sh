#!/bin/bash

if ! [ -x "$(command -v pyenv)" ]; then
  echo 'Error: pyenv is not installed.' >&2
  exit 1
fi

PROJECT_DIR=$(pwd)
rm -rf $PROJECT_DIR/.venv

poetry install --no-root
poetry run pre-commit install
