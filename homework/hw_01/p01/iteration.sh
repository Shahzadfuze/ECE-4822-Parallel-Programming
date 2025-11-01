#! /bin/bash


OUTFILE=timeing.csv
echo "N,Time" > $OUTFILE


for N in $(seq 10 10000); do
    echo "Running N=$N"
    T=$(/usr/bin/time -f "%U" ./mmulti $N $N 10 2>&1)
    echo "$N,$T" >> $OUTFILE
done






