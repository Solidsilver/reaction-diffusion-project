#include <metal_stdlib>
using namespace metal;

#define WIDTH 1920
#define HEIGHT 1080
#define DEPTH 2
#define A 0
#define B 1
#define dT 0.5

constant float weights[3][3] = {
    {0.05, 0.2, 0.05},
    {0.2, -1.0, 0.2},
    {0.05, 0.2, 0.05}
};

// Option 1: The "Parameter Space Map" (Active)
// Varies feed across X and kill across Y to show spots, stripes, chaos, and extinction in one frame.
//float diffA(int x, int y) { return 0.9; }
//float diffB(int x, int y) { return 0.5; }
//float feed(int x, int y) { return 0.01 + 0.09 * ((float)x / WIDTH); }
//float kill(int x, int y) { return 0.045 + 0.025 * ((float)y / HEIGHT); }

// Option 2: The "Coral Reef" (Traveling Waves)
 //Produces organic, brain-like traveling waves.
 //float diffA(int x, int y) { return 1.0; }
 //float diffB(int x, int y) { return 0.5; }
 //float feed(int x, int y) { return 0.0545; }
 //float kill(int x, int y) { return 0.062; }

 float diffA(int x, int y) { return 1.0; }
 float diffB(int x, int y) { return 0.2; }
 float feed(int x, int y) { return 0.034; }
 float kill(int x, int y) { return 0.065; }


// Option 3: The "Labyrinth / Maze"
// Medium feed and kill produce persistent evolving maze patterns.
 //float diffA(int x, int y) { return 1.0; }
 //float diffB(int x, int y) { return 0.5; }
 //float feed(int x, int y) { return 0.029; }
 //float kill(int x, int y) { return 0.057; }

// Option 4: The "Rapidly Evolving Chaos"
// Higher feed with lower kill creates fast, noisy, spiky growth.
 //float diffA(int x, int y) { return 0.8; }
 //float diffB(int x, int y) { return 0.4; }
 //float feed(int x, int y) { return 0.04; }
 //float kill(int x, int y) { return 0.06; }

// Option 5: The "Living Gradient"
// Varying diffusion rates creates a boundary where patterns dissolve and reform.
 //float diffA(int x, int y) { return 0.5 + 0.5 * ((float)x / WIDTH); }
 //float diffB(int x, int y) { return 0.2 + 0.3 * ((float)y / HEIGHT); }
 //float feed(int x, int y) { return 0.045; }
 //float kill(int x, int y) { return 0.065; }

// Option 6: The "Ring / Bullseye"
// Radial variation from the center.
// float diffA(int x, int y) { return 0.9; }
// float diffB(int x, int y) { return 0.5; }
// float feed(int x, int y) {
//     float dx = x - WIDTH/2.0;
//     float dy = y - HEIGHT/2.0;
//     float dist = sqrt(dx*dx + dy*dy) / (WIDTH/2.0);
//     return 0.02 + 0.06 * dist;
// }
// float kill(int x, int y) {
//     float dx = x - WIDTH/2.0;
//     float dy = y - HEIGHT/2.0;
//     float dist = sqrt(dx*dx + dy*dy) / (WIDTH/2.0);
//     return 0.05 + 0.02 * dist;
// }

int trIdx(int i, int j, int k) {
    return i * WIDTH * DEPTH + j * DEPTH + k;
}

float conv(device float *arr, int x, int y, int z) {
    float sum = 0.0;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            sum += weights[i + 1][j + 1] * arr[trIdx(y + i, x + j, z)];
        }
    }
    return sum;
}

float reactA(float valA, float valB, int x, int y) {
    return feed(x, y) * (1.0 - valA) - valA * valB * valB;
}

float reactB(float valA, float valB, int x, int y) {
    return valA * valB * valB - (kill(x, y) + feed(x, y)) * valB;
}

bool bdry(int x, int y) { return true; }

kernel void update(device float *cur [[buffer(0)]],
                   device float *prev [[buffer(1)]],
                   uint2 gid [[thread_position_in_grid]]) {
    int i = int(gid.y);
    int j = int(gid.x);
    if (i <= 0 || i >= HEIGHT - 1 || j <= 0 || j >= WIDTH - 1) return;
    if (!bdry(j, i)) return;

    float prevA = prev[trIdx(i, j, A)];
    float prevB = prev[trIdx(i, j, B)];

    float curA = prevA + (diffA(j, i) * conv(prev, j, i, A) + reactA(prevA, prevB, j, i)) * dT;
    float curB = prevB + (diffB(j, i) * conv(prev, j, i, B) + reactB(prevA, prevB, j, i)) * dT;

    cur[trIdx(i, j, A)] = curA;
    cur[trIdx(i, j, B)] = curB;
}

kernel void fill_pixels(texture2d<float, access::write> pixels [[texture(0)]],
                        device float *arr [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    int i = int(gid.y);
    int j = int(gid.x);
    if (i >= HEIGHT || j >= WIDTH) return;

    int indexA = trIdx(i, j, A);
    int indexB = trIdx(i, j, B);
    float amt = fmax(0.0, arr[indexA] - arr[indexB]);

    pixels.write(float4(amt, amt, amt, 1.0), gid);
}
