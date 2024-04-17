#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "photon_map.h"
#include <glm/glm.hpp>
#include <chrono>

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 100000;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;
    double *ref_c;
 
    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
    ref_c = (double*)malloc(bytes);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }
    
    // get reference sol
    for (i = 0; i < n; i++) {
        ref_c[i] = h_a[i] + h_b[i];
    }

    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);
 
    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
 
    // Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
 
    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    double ref_sum = 0;
    for(i=0; i<n; i++) {
        sum += h_c[i];
        ref_sum += ref_c[i];
    }
    printf("final result, expected result: %f, %f\n", sum/(double)n, ref_sum/(double)n);
 
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // write a png of all black
    int width = 1920;
    int height = 1080;
    int channels = 3;
    char *pixels = (char*)malloc(width*height*channels*sizeof(char));
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < channels; k++) {
                pixels[i*height*channels + j*channels + k] = (char)(rand() % 255);
            }
        }
    }
    stbi_write_png("test.png", width, height, channels, pixels, width*channels);

    auto map = new PhotonMap();
    auto phots = new std::vector<PhotonMap::Photon>();
    for (int i = 0; i < 1000000; i++) {
        phots->emplace_back(glm::vec4(i, i, i, i), glm::vec4(0, 0, 0, 0), glm::vec4(0, 0, 0, 0));
    }
    
    auto naive_start = std::chrono::high_resolution_clock::now();
    glm::vec4 loc(1.4, 1.4, 1.4, 1.4);
    int min_dist = 2000000;
    PhotonMap::Photon naive_closest;
    for (int i = 0; i < 1000000; i++) {
        double dist = distance(loc, phots->at(i).position);
        if (dist < min_dist) {
            min_dist = dist;
            naive_closest = phots->at(i);
        }
    }
    auto naive_time = std::chrono::high_resolution_clock::now() - naive_start;

    std::cout << "Naive nearest neighbor took " << naive_time.count() << ", and found the nearest neighbor (" << naive_closest.position.x << ", "
              << naive_closest.position.y << ", " << naive_closest.position.z << ", " << naive_closest.position.w << ") with a distance of "
              << min_dist << std::endl;

    auto kd_start = std::chrono::high_resolution_clock::now();
    map->buildMap(phots);
    auto kd_closest = map->kNearestNeighbors(1, loc)[0];
    auto kd_dist = distance(loc, kd_closest.position);
    auto kd_time = std::chrono::high_resolution_clock::now() - kd_start;
    std::cout << "KD nearest neighbor took " << kd_time.count() << ", and found the nearest neighbor (" << kd_closest.position.x << ", "
              << kd_closest.position.y << ", " << kd_closest.position.z << ", " << kd_closest.position.w << ") with a distance of "
              << kd_dist << std::endl;
    return 0;
}