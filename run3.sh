#!/bin/bash

BIN="/home/vipulagarwal/Documents/gunrock/build/bin/bfs_csr"

for g in graphs/*_csr.txt; do
  echo "===== $g ====="
  "$BIN" < "$g"
  echo
done