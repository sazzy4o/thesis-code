#!/bin/bash

while :
do
    count=`sq | wc -l`
    echo $count
    if [ "$count" == "1" ]; then
        ./B_predict_all.sh dev
    fi
    sleep 1m
done