#!/bin/bash

for file in graphs/*_csr.txt; do
    echo "========================================"
    echo "Running on: $(basename "$file")"
    echo "========================================"

    ./a.out < "$file"

    echo
done