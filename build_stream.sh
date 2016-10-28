#!/bin/bash

# Splits a pre-processed corpus file (line by line) into a stream containing labels and a stream containing processed texts

SRC=$1
OUT_LAB=$2
OUT_TEXT=$3
id=$4

# Build label stream for doc-stream
cut -f1 "$SRC" > tmp
echo -n "" > "$OUT_LAB"
ptr=$id
while read line; do
        echo "[[$ptr]]" >> "$OUT_LAB" # id for current doc
        outLine="$line"
        echo "$outLine" >> "$OUT_LAB"
        ptr=$((ptr+1)) # increment doc/sentence unique identifier
done < tmp

# Build actual doc-stream
cut -f2 "$SRC" > tmp
echo -n "" > "$OUT_TEXT"
ptr=$id
while read line; do
        echo "[[$ptr]]" >> "$OUT_TEXT" # id for current doc
        outLine="$line"
        echo "$outLine" >> "$OUT_TEXT"
        ptr=$((ptr+1)) # increment doc/sentence unique identifier
done < tmp
rm tmp
