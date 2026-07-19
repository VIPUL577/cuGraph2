#!/bin/bash

for file in graphs/*_mod.txt; do
    echo "========================================"
    echo "Running on: $(basename "$file")"
    echo "========================================"

    ./a.out < "$file"

    echo
done