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

echo -e "python generate-reduction-buckets.py -v ${NAME}-data ${NAME}-sequential ${NAME}-states ${NAME}-buckets-individual-8 8"
python generate-reduction-buckets.py -v "${NAME}-data" "${NAME}-sequential" "${NAME}-states" "${NAME}-buckets" 8
echo -e "Finished reduction buckets: ${NAME}-buckets\n"

echo -e "python generate-semantic-model.py -v ${NAME}-data ${NAME}-sequential ${NAME}-states ${NAME}-encoding"
python generate-semantic-model.py -v "${NAME}-data" "${NAME}-sequential" "${NAME}-states" "${NAME}-encoding"
echo -e "Finished semantic model: ${NAME}-encoding\n"

echo -e "python generate-activation-states.py -v ${NAME}-data ${NAME}-sequential ${NAME}-activations"
python generate-activation-states.py -v "${NAME}-data" "${NAME}-sequential" "${NAME}-activations"
echo -e "Finished activation states: ${NAME}-activations\n"

echo -e "python generate-query-database.py -v ${NAME}-data ${NAME}-sequential ${NAME}-activations ${NAME}-query"
python generate-query-database.py -v "${NAME}-data" "${NAME}-sequential" "${NAME}-activations" "${NAME}-query"
echo -e "Finished query database: ${NAME}-query\n"

echo -e "python dev-server.py -v ${NAME}-data ${NAME}-sequential ${NAME}-buckets ${NAME}-encoding ${NAME}-query"
echo -e "Use ctrl + c to quit.."
python dev-server.py -v "${NAME}-data" "${NAME}-sequential" "${NAME}-buckets" "${NAME}-encoding" --query-dir "${NAME}-query"

