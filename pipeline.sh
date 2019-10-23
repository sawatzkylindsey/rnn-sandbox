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

echo -e "python generate-data.py -v lm raw ${CORPUS_PATH} ${NAME}-data"
# Assumes a language model from a raw corpus.
#                          v  v
python generate-data.py -v lm raw ${CORPUS_PATH} "${NAME}-data"
echo -e "Finished data: ${NAME}-data\n"

echo -e "python generate-sequential-model.py -v ${NAME}-data ${NAME}-sequential"
python generate-sequential-model.py -v "${NAME}-data" "${NAME}-sequential"
echo -e "Finished sequential model: ${NAME}-sequential\n"

echo -e "python generate-hidden-states.py -v ${NAME}-data ${NAME}-sequential ${NAME}-states ..."
python generate-hidden-states.py -v "${NAME}-data" "${NAME}-sequential" "${NAME}-states" train -s .25
python generate-hidden-states.py -v "${NAME}-data" "${NAME}-sequential" "${NAME}-states" validation -s .5
python generate-hidden-states.py -v "${NAME}-data" "${NAME}-sequential" "${NAME}-states" test -s .5
echo -e "Finished hidden states: ${NAME}-states\n"

# TODO
# Required for dev-server - should be removed
python generate-reduction-buckets.py -v "${NAME}-data" "${NAME}-sequential" "${NAME}-states" "${NAME}-buckets" 1

echo -e "python generate-semantic-model.py -v ${NAME}-data ${NAME}-sequential ${NAME}-states ${NAME}-encoding"
python generate-semantic-model.py -v "${NAME}-data" "${NAME}-sequential" "${NAME}-states" "${NAME}-encoding"
echo -e "Finished semantic model: ${NAME}-encoding\n"

echo -e "python dev-server.py -v ${NAME}-data ${NAME}-sequential ${NAME}-buckets ${NAME}-encoding"
echo -e "Use ctrl + c to quit.."
python dev-server.py -v "${NAME}-data" "${NAME}-sequential" "${NAME}-buckets" "${NAME}-encoding"

