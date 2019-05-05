#!/bin/bash -e

if [[ $# -lt 1 ]]; then
    echo "Require the path to the corpus."
    exit
elif [[ $# -lt 2 ]]; then
    echo "Require the name to use for the generated artifacts."
    exit
fi

CORPUS_PATH=$1
NAME=$2

echo -e "Running.. logs generated to the files '.generate-*.log'\n"

echo "python generate-data.py -v lm raw ${CORPUS_PATH} ${NAME}-data"
# Assumes a language model from a raw corpus.
#                          v  v
python generate-data.py -v lm raw ${CORPUS_PATH} "${NAME}-data"
echo -e "Finished data: ${NAME}-data\n"

echo "python generate-sequential-model.py -v ${NAME}-data ${NAME}-sequential --consecutive-decays 1"
python generate-sequential-model.py -v "${NAME}-data" "${NAME}-sequential" --consecutive-decays 1
echo -e "Finished sequential model: ${NAME}-sequential\n"

echo "python generate-hidden-states.py -v ${NAME}-data ${NAME}-sequential ${NAME}-states --sample-rates .2 .5"
python generate-hidden-states.py -v "${NAME}-data" "${NAME}-sequential" "${NAME}-states" --sample-rates .2 .5
echo -e "Finished hidden states: ${NAME}-states\n"

echo "python generate-reduction-buckets.py -v ${NAME}-states ${NAME}-buckets 10"
# Assumes a target reduction size:                                         v
python generate-reduction-buckets.py -v "${NAME}-states" "${NAME}-buckets" 10
echo -e "Finished reduction buckets: ${NAME}-buckets\n"

echo "python generate-semantic-model.py -v ${NAME}-data ${NAME}-states ${NAME}-encoding"
python generate-semantic-model.py -v "${NAME}-data" "${NAME}-states" "${NAME}-encoding"
echo -e "Finished semantic model: ${NAME}-encoding\n"

# TODO:
# python dev-server.py ..

