#!/bin/bash



for (( i=1; i<33; i++ )); do
    # Commands to execute for each number

    echo "Number of threads: $i \n"
time ./x.exe 1000 1000 1000 $i
    done



