#!/bin/bash


scp ./build/tools/classification.bin lambda:~/package/bin/classification_hed.bin

scp ./build/lib/libcaffe_hed.so lambda:~/package/lib
