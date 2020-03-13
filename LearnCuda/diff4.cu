#include <SFML/Graphics.hpp>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <time.h>
#define WIDTH 1000
#define HEIGHT 1000
#define A 0
#define B 1
#define RANDOM 5 // Set to -1 to disable
#define HSV 0    // 1 to use hsv 0 to use rgv
#define DT 1
#define USEDIST = true;
using namespace std;
using namespace sf;

typedef struct {
  double r; // a fraction between 0 and 1
  double g; // a fraction between 0 and 1
  double b; // a fraction between 0 and 1
} rgb;

__constant__ float weights[3][3] = {
    {0.05, 0.2, 0.05}, {0.2, -1, 0.2}, {0.05, 0.2, 0.05}};

__device__ float DB(int i, int j) {
  return 0.5;
  float d = ((i - HEIGHT / 2) * (i - HEIGHT / 2)) +
            ((j - WIDTH / 2) * (j - WIDTH / 2));

  return 1.4 - (0.05 + d / (((HEIGHT / 2) * (HEIGHT / 2)) +
                            ((WIDTH / 2) * (WIDTH / 2))));
}

__device__ float DA(int i, int j) {
  return 1;
  float d = ((i - HEIGHT / 2) * (i - HEIGHT / 2)) +
            ((j - WIDTH / 2) * (j - WIDTH / 2));
  return 1.4 - (0.05 + d / (((HEIGHT / 2) * (HEIGHT / 2)) +
                            ((WIDTH / 2) * (WIDTH / 2))));
}

// Interesting value pairs:
// (Feed, Kill):
// (0.035, 0.065) - spots @ edges of init. cond.
// (0.055, 0.065) - long, pushing tendrils
// (0.055, 0.063) - wobbly/stripy with spot-tipped tendrils
// (0.045, 0.063) - spots and wobbly stripes
// (0.050, 0.060) - fast-spreading inverted lines + inverted spots
// (0.045, 0.060) - fast-spreading lines + holes

__device__ float feed(int i, int j) { return 0.035; }

__device__ float kill(int i, int j) { return 0.065; } // 0.062 | 0.065

__device__ int transIdx(int i, int j, int chem) {
  return i * WIDTH * 2 + j * 2 + chem;
}

__device__ bool randD(int tdX, int bdX) {
  bool useDist = false;
  if (useDist) {
    return (tdX * bdX % 256 < RANDOM || tdX * tdX % 256 > -RANDOM);
  } else {
    return false;
  }
}

__global__ void init_map(float *arr) {
  int indexY = blockIdx.x;
  int strideY = gridDim.x;
  int indexX = threadIdx.x;
  int strideX = blockDim.x;
  for (int i = indexY; i < HEIGHT; i += strideY) {
    for (int j = indexX; j < WIDTH; j += strideX) {
      if (i < HEIGHT && j < WIDTH) {
        arr[transIdx(i, j, A)] = 1;
        if (randD((int)threadIdx.x, (int)blockIdx.x)) {
          arr[transIdx(i, j, B)] = 1; // 1
        } else {
          arr[transIdx(i, j, B)] = 0;
        }
      }
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
      int indexA = transIdx(i, j, A);
      int indexB = transIdx(i, j, B);
      Uint8 c = max((arr[indexA] - arr[indexB]), (float)0) * 255;
      pixels[(((i * WIDTH) + j) * 4) + 0] = c;
      pixels[(((i * WIDTH) + j) * 4) + 1] = c;
      pixels[(((i * WIDTH) + j) * 4) + 2] = c;
      pixels[(((i * WIDTH) + j) * 4) + 3] = 255;
    }
  }
}

__device__ float conv(float *arr, int index, int y, int x) {
  float sum = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      sum += weights[i][j] * arr[transIdx((y - 1 + i), (x - 1 + j), index)];
    }
  }
  return sum;
}

__global__ void update(float *cur, float *prev) {
  int indexY = blockIdx.x;
  int strideY = gridDim.x;
  int indexX = threadIdx.x;
  int strideX = blockDim.x;
  for (int i = indexY + 1; i < HEIGHT - 1; i += strideY) {
    if (i <= HEIGHT - 1) {
      for (int j = indexX + 1; j < WIDTH - 1; j += strideX) {
        if (j <= WIDTH - 1) {
          // Computed 3d-to-1d index
          int indexA = transIdx(i, j, A);
          int indexB = transIdx(i, j, B);
          float pa = prev[indexA];
          float pb = prev[indexB];
          cur[indexA] = pa + ((DA(i, j) * conv(prev, A, i, j)) -
                              (pa * pb * pb) + (feed(i, j) * (1 - pa))) *
                                 DT;
          cur[indexB] =
              pb + ((DB(i, j) * conv(prev, B, i, j)) + (pa * pb * pb) -
                    ((kill(i, j) + feed(i, j)) * pb)) *
                       DT;

          cur[indexA] = max((float)0.0, min((float)1.0, cur[indexA]));
          cur[indexB] = max((float)0.0, min((float)1.0, cur[indexB]));
        }
      }
    }
  }
}

__device__ double f(int x, int y) {
  return 1;
  // return x/y;
  // return __fdiv_rd(y,x);
  float nX = __fdiv_rd(x-(HEIGHT/2),200);
  float nY = __fdiv_rd(y-(WIDTH/2),200);
  float len = nX*nX + nY*nY;
  return 1-len;
}

__global__ void initState(float *prev) {
  for (int i = HEIGHT / 2 - 200; i < HEIGHT / 2 + 200; i++) {
    for (int j = WIDTH / 2 - 200; j < WIDTH / 2 + 200; j++) {
      prev[transIdx(i, j, B)] = f(i, j);
    }
  }
}

int main() {

  // cuda settings
  int blockSize = 1024;
  int numBlocks = HEIGHT;
  bool paused = true;

  srand(time(0));
  Clock clock;
  // Create the main window
  sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Reaction-Diffusion");
  // window.setFramerateLimit(60);

  sf::Event event;

  sf::Image img;
  img.create(WIDTH, HEIGHT);

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
  init_map<<<numBlocks, blockSize>>>(cur);
  init_map<<<numBlocks, blockSize>>>(prev);
  initState<<<1, 1>>>(prev);

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
      // cudaDeviceSynchronize();

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
