CC = nvcc
CFLAGS = -std=c++11 --compiler-options -Wall --compiler-options -Wextra --compiler-options -ggdb
LDFLAGS = -lsfml-graphics -lsfml-window -lsfml-system -lcurand
# LDFLAGS = -lGL -lGLU -lglut

all: diff3 diff4 diff5

%: %.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) $< -c