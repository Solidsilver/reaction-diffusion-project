#!/bin/bash
gcc -c main.cpp
g++ main.o -o diff-app -lsfml-graphics -lsfml-window -lsfml-system 
