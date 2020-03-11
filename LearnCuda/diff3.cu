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
#define HSV 1    // 1 to use hsv 0 to use rgv
#define DT 1
using namespace std;
using namespace sf;

typedef struct {
  double r; // a fraction between 0 and 1
  double g; // a fraction between 0 and 1
  double b; // a fraction between 0 and 1
} rgb;

typedef struct {
  double h; // angle in degrees
  double s; // a fraction between 0 and 1
  double v; // a fraction between 0 and 1
} hsv;

__device__ rgb hsv2rgb(hsv in) {
  double hh, p, q, t, ff;
  long i;
  rgb out;

  if (in.s <= 0.0) { // < is bogus, just shuts up warnings
    out.r = in.v;
    out.g = in.v;
    out.b = in.v;
    return out;
  }
  hh = in.h;
  if (hh >= 360.0)
    hh = 0.0;
  hh /= 60.0;
  i = (long)hh;
  ff = hh - i;
  p = in.v * (1.0 - in.s);
  q = in.v * (1.0 - (in.s * ff));
  t = in.v * (1.0 - (in.s * (1.0 - ff)));

  switch (i) {
  case 0:
    out.r = in.v;
    out.g = t;
    out.b = p;
    break;
  case 1:
    out.r = q;
    out.g = in.v;
    out.b = p;
    break;
  case 2:
    out.r = p;
    out.g = in.v;
    out.b = t;
    break;

  case 3:
    out.r = p;
    out.g = q;
    out.b = in.v;
    break;
  case 4:
    out.r = t;
    out.g = p;
    out.b = in.v;
    break;
  case 5:
  default:
    out.r = in.v;
    out.g = p;
    out.b = q;
    break;
  }
  return out;
}

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

__device__ float feed(int i, int j) {
  return 0.055+0.1*(float)(i*i)/(HEIGHT*HEIGHT);
  // return 0.055;
  // return 0.01 + 0.1 * ((float)i / HEIGHT) - 0.01 * ((float)j / HEIGHT);
}

__device__ float kill(int i, int j) {
  return 0.062+0.01*(float)(j*j)/(WIDTH*WIDTH);
  // return 0.055;
  // return 0.062;
  // return 0.055 + 0.01 * ((float)j / WIDTH) - 0.001 * ((float)i / WIDTH);
}

__global__ void init_map(float *arr) {
  int indexY = blockIdx.x;
  int strideY = gridDim.x;
  int indexX = threadIdx.x;
  int strideX = blockDim.x;
  for (int i = indexY; i < HEIGHT; i += strideY) {
    for (int j = indexX; j < WIDTH; j += strideX) {
      if (i < HEIGHT && j < WIDTH) {
        arr[i * WIDTH * 2 + j * 2 + A] = 1;
        if (threadIdx.x * blockIdx.x % 256 < RANDOM ||
            threadIdx.x * blockIdx.x % 256 > -RANDOM) {
          arr[i * WIDTH * 2 + j * 2 + B] = 1;
        } else {
          arr[i * WIDTH * 2 + j * 2 + B] = 0;
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
      if (HSV) {
        hsv h;
        h.s = 1;
        h.v = 1 - max((arr[i * WIDTH * 2 + j * 2 + A] -
                       arr[i * WIDTH * 2 + j * 2 + B]),
                      (float)0);
        h.h = max((arr[i * WIDTH * 2 + j * 2 + A] -
                   arr[i * WIDTH * 2 + j * 2 + B]),
                  (float)0) *
              360;
        rgb a = hsv2rgb(h);
        pixels[(((i * WIDTH) + j) * 4) + 0] = a.r * 255;
        pixels[(((i * WIDTH) + j) * 4) + 1] = a.g * 255;
        pixels[(((i * WIDTH) + j) * 4) + 2] = a.b * 255;
        pixels[(((i * WIDTH) + j) * 4) + 3] = 255;
      } else {
        Uint8 c = max((arr[i * WIDTH * 2 + j * 2 + A] -
                       arr[i * WIDTH * 2 + j * 2 + B]),
                      (float)0) *
                  255;
        pixels[(((i * WIDTH) + j) * 4) + 0] = c;
        pixels[(((i * WIDTH) + j) * 4) + 1] = c;
        pixels[(((i * WIDTH) + j) * 4) + 2] = c;
        pixels[(((i * WIDTH) + j) * 4) + 3] = 255;
      }
    }
  }
}

__device__ float conv(float *arr, int index, int y, int x) {
  float sum = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      sum += weights[i][j] *
             arr[(y - 1 + i) * WIDTH * 2 + (x - 1 + j) * 2 + index];
    }
  }
  return sum;
}

__global__ void update(float *cur, float *prev) {
  // cout << "UPDATE"<<endl;
  int indexY = blockIdx.x;
  int strideY = gridDim.x;
  int indexX = threadIdx.x;
  int strideX = blockDim.x;
  for (int i = indexY + 1; i < HEIGHT - 1; i += strideY) {
    if (i <= HEIGHT - 1) {
      for (int j = indexX + 1; j < WIDTH - 1; j += strideX) {
        if (j <= WIDTH - 1) {
          // cout << i << " " << j << endl;
          float pa = prev[i * WIDTH * 2 + j * 2 + A];
          float pb = prev[i * WIDTH * 2 + j * 2 + B];
          cur[i * WIDTH * 2 + j * 2 + A] =
              pa + ((DA(i, j) * conv(prev, A, i, j)) - (pa * pb * pb) +
                    (feed(i, j) * (1 - pa))) *
                       DT;
          cur[i * WIDTH * 2 + j * 2 + B] =
              pb + ((DB(i, j) * conv(prev, B, i, j)) + (pa * pb * pb) -
                    ((kill(i, j) + feed(i, j)) * pb)) *
                       DT;

          cur[i * WIDTH * 2 + j * 2 + A] =
              max((float)0.0, min((float)1.0, cur[i * WIDTH * 2 + j * 2 + A]));
          cur[i * WIDTH * 2 + j * 2 + B] =
              max((float)0.0, min((float)1.0, cur[i * WIDTH * 2 + j * 2 + B]));
        }
      }
    }
  }
}

__global__ void initState(float *prev) {
  for (int i = HEIGHT / 2 - 200; i < HEIGHT / 2 + 200; i++) {
    for (int j = WIDTH / 2 - 200; j < WIDTH / 2 + 200; j++) {
      if (i == j || i == -j) {
        prev[i * WIDTH * 2 + j * 2 + B] = 1;
      }
    }
  }
}

// __global__ void init_rand(curandState *state) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   curand_init(314159, idx, 0, &state[idx]);
// }

int main() {
  srand(time(0));
  Clock clock;
  // Create the main window
  sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Reaction-Diffusion");
  sf::Image img;
  // window.setFramerateLimit(60);
  img.create(WIDTH, HEIGHT);
  // Uint8 *pixels = new Uint8[WIDTH * HEIGHT * 4];
  Uint8 *pixels;
  cudaMallocManaged(&pixels, WIDTH * HEIGHT * 4);
  sf::Texture texture;
  texture.create(WIDTH, HEIGHT);

  sf::IntRect r(0, 0, WIDTH, HEIGHT);
  sf::Sprite sprite(texture, r);

  int blockSize = 512;
  int numBlocks = HEIGHT;
  bool paused = true;
  float *cur;
  float *prev;
  // cudaMalloc(&cur, HEIGHT * sizeof(float **));
  // cudaMalloc(&prev, HEIGHT * sizeof(float **));
  cudaMalloc((void **)&cur, HEIGHT * WIDTH * 2 * sizeof(float));
  cudaMalloc((void **)&prev, HEIGHT * WIDTH * 2 * sizeof(float));
  init_map<<<numBlocks, blockSize>>>(cur);
  init_map<<<numBlocks, blockSize>>>(prev);
  initState<<<1, 1>>>(prev);
  // Show initial state
  fill_pixels<<<numBlocks, blockSize>>>(pixels, cur);
  cudaDeviceSynchronize();
  // Start the anim. loop
  while (window.isOpen()) {
    float t = clock.restart().asSeconds();
    window.setTitle(to_string(1 / t));
    sf::Event event;
    while (window.pollEvent(event)) {
      // Close window : exit
      if (event.type == sf::Event::Closed)
        window.close();
      if (event.type == Event::KeyPressed && event.key.code == Keyboard::Space)
        paused = !paused;
    }
    if (!paused) {
      update<<<numBlocks, blockSize>>>(cur, prev);
      fill_pixels<<<numBlocks, blockSize>>>(pixels, cur);
      cudaDeviceSynchronize();
    }
    texture.update(pixels);
    window.draw(sprite);

    // Update the window
    window.display();
    if (!paused) {
      float *tmp = cur;
      cur = prev;
      prev = tmp;
    }
  }
  // cudaFreeArr(cur, HEIGHT, WIDTH);
  cudaFree(cur);
  // cudaFreeArr(prev, HEIGHT, WIDTH);
  cudaFree(prev);
  cudaFree(pixels);
  return 0;
}
