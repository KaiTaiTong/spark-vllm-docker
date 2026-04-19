#!/bin/bash
if [[ "$1" == "--list" || "$1" == "-l" ]]; then
    echo "Available recipes:"
    for f in "$(dirname "$0")/my-recipes"/*.yaml; do
        echo "  $(basename "$f" .yaml)"
    done
    exit 0
fi
VLLM_SPARK_EXTRA_DOCKER_ARGS="-v /opt/llms/models:/opt/llms/models" \
  ./run-recipe.sh "my-recipes/$1.yaml" --solo "${@:2}"
