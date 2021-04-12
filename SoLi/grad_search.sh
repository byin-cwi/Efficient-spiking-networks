#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for h in $(seq 1 1 11)
do
    for s in $(seq 1 1 10)
    do 
        python srnn_frame-grad.py --scale $s --hight $h
    done
done


# for s in $(seq 1 1 10)
# do 
#     python srnn_frame-grad.py --scale $s
# done