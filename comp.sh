#!/bin/sh
g++ -o main main.cpp -I /usr/local/cuda/include -I ../nvcc_test/inc/ -L /usr/local/cuda/lib -lcudart -lcublas
