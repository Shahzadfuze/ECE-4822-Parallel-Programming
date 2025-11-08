#!/bin/bash





# Arrays of blocks and threads
blocks=(1, 100, 300, 600)
threads=(1, 10, 15, 20, 32)

for b in "${blocks[@]}"; do
    for t in "${threads[@]}"; do
	echo "Running with blocks=$b, threads=$t"
	time ./src/add.exe $b $t
    done
    done
