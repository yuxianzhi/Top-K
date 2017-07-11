#!/bin/bash

line=""
for((i=1;i<=$2;i++));  
do   
    #random=`expr $RANDOM % 100000`; 
    #line=${line}"$random "
    line=${line}"$i "
done

rm $1
for((i=1;i<=$3;i++));
do
    echo ${line} >> $1
done
