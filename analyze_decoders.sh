#!/bin/bash

# This script is used to probe model decoders and uncover how they rank/associate words (of a given data-set) to one another
# @author Alexander G. Ororbia II

THETA="" # /path/to/parameters.theta
DECODER_NAME="R" # name of decoder parameter (in OGraph)
DICT="" # /path/to/lexicon.dict
K=10 # top-k symbols to retrieve w/ respect to query
PERFORM_ROT=false # perform a rotation of decoder based on eigenvectors
QUERY_WORD="government"
METRIC="euclidean"  # euclidean  OR cosine
TRANSPOSE_DECODER=false # leave this @ false unless analyzing a non-decoder

# Run tool given configuration
java -Xmx1g -jar embeddingAnalyzer.jar "$THETA" "$DECODER_NAME" "$DICT" "$K" "$PERFORM_ROT" "$QUERY_WORD" "$METRIC" "$TRANSPOSE_DECODER"
