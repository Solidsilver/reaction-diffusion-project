#!/bin/bash
g++ -O0 -g3 -Wall -c -fmessage-length=0 -o GL01Hello.o main.cpp
# g++ -o GL01Hello GL01Hello.o -lGL -lGLU -lglut
g++ -o GL01Hello GL01Hello.o -lsfml-graphics -lsfml-window -lsfml-system -lcurand