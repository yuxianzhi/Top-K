#!/bin/bash

K_max=20

# compile
make -B

# init
rm result.txt
./random_init.sh /dev/shm/input.txt 7000 10000


# run
for((i=1;i<=${K_max};i++));  
do   
    ./main /dev/shm/input.txt 70000000 $i >> result.txt
done

# use Python matplotlit to Plot 
#grep "Result" result.txt | cut -d " " -f 3 | awk '++i%2' | awk BEGIN{RS=EOF}'{gsub(/\n/,",");print}'
#grep "Result" result.txt | cut -d " " -f 3 | awk 'i++%2' | awk BEGIN{RS=EOF}'{gsub(/\n/,",");print}'
#python plot.py
#eog result.jpg
