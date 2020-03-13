#include <SFML/Graphics.hpp>
#include <cstdlib>
#include <iostream>

#define WIDTH 1000
#define HEIGHT 1000
#define DEPTH 2
#define A 0
#define B 1
#define dT 1

using namespace std;
using namespace sf;


__constant__ float weights[3][3] = {
    {0.05, 0.2, 0.05}, {0.2, -1, 0.2}, {0.05, 0.2, 0.05}};

__device__ float diffA(int x, int y) { return 1; }

__device__ float diffB(int x, int y) { return 0.5; }

__device__ float feed(int x, int y) { return 0.0367; }

__device__ float kill(int x, int y) { return 0.0649; }

// Translate indicies of 3d array index to flattened 1d array
__device__ int trIdx(int i, int j, int k) {
  return i * WIDTH * DEPTH + j * DEPTH + k;
}

__device__ int trIdx2(int i, int j, int k, int width, int depth) {
  return i * width * depth + j * depth + k;
}

__device__ float conv(float *arr, int x, int y, int z) {
  float sum = 0;
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      sum += weights[i + 1][j + 1] * arr[trIdx(y + i, x + j, z)];
    }
  }
  return sum;
}

__device__ float reactA(float valA, float valB, int x, int y) {
  return feed(x, y) * (1 - valA) - valA * valB * valB;
}

__device__ float reactB(float valA, float valB, int x, int y) {
  return valA * valB * valB - (kill(x, y) + feed(x, y)) * valB;
}

__device__ float initA(int x, int y) { return 1; }

__device__ float initB(int x, int y) {
  if (y > HEIGHT / 2 - 200 && y < HEIGHT / 2 + 200 && x > WIDTH / 2 - 200 &&
      x < WIDTH / 2 + 200) {
    return 1;
  }
  return 0;
}

__global__ void init_state(float *arr) {
  // Setup GPU thread indexing
  int tidxI = blockIdx.x;
  int strideI = gridDim.x;
  int tidxJ = threadIdx.x;
  int strideJ = blockDim.x;

  for (int i = tidxI; i < HEIGHT; i += strideI) {
    for (int j = tidxJ; j < WIDTH; j += strideJ) {
      arr[trIdx(i, j, A)] = initA(j, i);
      arr[trIdx(i, j, B)] = initB(j, i);
    }
  }
}

__global__ void update(float *cur, float *prev) {
  // Setup GPU thread indexing
  int tidxI = blockIdx.x;
  int strideI = gridDim.x;
  int tidxJ = threadIdx.x;
  int strideJ = blockDim.x;

  for (int i = tidxI + 1; i < HEIGHT - 1; i += strideI) {
    for (int j = tidxJ + 1; j < WIDTH - 1; j += strideJ) {
      float prevA = prev[trIdx(i, j, A)];
      float prevB = prev[trIdx(i, j, B)];

      float curA =
          prevA +
          (diffA(j, i) * conv(prev, j, i, A) + reactA(prevA, prevB, j, i)) * dT;
      float curB =
          prevB +
          (diffB(j, i) * conv(prev, j, i, B) + reactB(prevA, prevB, j, i)) * dT;

      cur[trIdx(i, j, A)] = curA;
      cur[trIdx(i, j, B)] = curB;
    }
  }
}


__global__ void fill_pixels(Uint8 *pixels, float *arr) {
  int indexY = blockIdx.x;
  int strideY = gridDim.x;
  int indexX = threadIdx.x;
  int strideX = blockDim.x;
  for (int i = indexY; i < HEIGHT; i += strideY) {
    for (int j = indexX; j < WIDTH; j += strideX) {
      int indexA = trIdx(i, j, A);
      int indexB = trIdx(i, j, B);
      // int comb = arr[indexA] + arr[indexB];
      // if (comb < 0) {
      //   comb *= -1;
      // }
      pixels[trIdx2(i, j, 0, WIDTH, 4)] = 0 * 255;
      pixels[trIdx2(i, j, 1, WIDTH, 4)] = (arr[indexB]) * 255;
      pixels[trIdx2(i, j, 2, WIDTH, 4)] = (arr[indexA]) * 255;
      pixels[trIdx2(i, j, 3, WIDTH, 4)] = 255;
    }
  }
}

int main() {

  // cuda settings
  int blockSize = 1024;
  int numBlocks = HEIGHT;
  bool paused = true;

  // Create the window
  sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Reaction-Diffusion");
  // window.setFramerateLimit(10);

  sf::Event event;

  sf::Texture texture;
  texture.create(WIDTH, HEIGHT);
  sf::IntRect r(0, 0, WIDTH, HEIGHT);
  sf::Sprite sprite(texture, r);

  // Uint8 *pixels = new Uint8[WIDTH * HEIGHT * 4];
  Uint8 *pixels;
  Uint8 *pixLocal = new Uint8[WIDTH * HEIGHT * 4];
  cudaMalloc((void **)&pixels, WIDTH * HEIGHT * 4);

  // Setup cuda
  float *cur;
  float *prev;
  cudaMalloc((void **)&cur, HEIGHT * WIDTH * 2 * sizeof(float));
  cudaMalloc((void **)&prev, HEIGHT * WIDTH * 2 * sizeof(float));
  init_state<<<numBlocks, blockSize>>>(cur);
  init_state<<<numBlocks, blockSize>>>(prev);

  // Show initial state
  fill_pixels<<<numBlocks, blockSize>>>(pixels, prev);
  cudaDeviceSynchronize();
  cudaMemcpy(pixLocal, pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
  texture.update(pixLocal);
  window.draw(sprite);

  // Start the anim. loop
  while (window.isOpen()) {
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        window.close();
      }
      if (event.type == Event::KeyPressed &&
          event.key.code == Keyboard::Space) {
        paused = !paused;
      }
    }

    if (!paused) {
      update<<<numBlocks, blockSize>>>(cur, prev);
      fill_pixels<<<numBlocks, blockSize>>>(pixels, cur);
      cudaMemcpy(pixLocal, pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();

      texture.update(pixLocal);
      window.draw(sprite);

      float *tmp = cur;
      cur = prev;
      prev = tmp;
    }

    // Update the window
    window.display();
  }

  // Free mem
  cudaFree(cur);
  cudaFree(prev);
  cudaFree(pixels);
  free(pixLocal);

  return 0;
}
