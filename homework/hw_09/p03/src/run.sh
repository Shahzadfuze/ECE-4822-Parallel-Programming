#!/bin/bash





# Arrays of blocks and threads
blocks=(1, 100, 300, 600)
threads=(1, 10, 15, 20, 32)
gpus=(1, 2, 3, 4)

for b in "${blocks[@]}"; do
    for t in "${threads[@]}"; do
	for g in "${gpus[@]}"; do
		 echo "Running with blocks=$b, threads=$t, GPU=$g"
	time ./add.exe $b $t $g
    done
    done
    done
