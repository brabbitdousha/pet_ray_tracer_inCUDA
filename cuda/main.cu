#include <iostream>
#include <fstream>
#include <time.h>
#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define checkCudaErrors(val) check_cuda((val),#val,_FILE_,_LINE_)

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error =" << static_cast<unsigned int>(result) << "at" <<
            file << ":" << line << "'" << func << "'\n'";

        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(float* fb, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x * 3 + i * 3;
    fb[pixel_index + 0] = float(i) / max_x;
    fb[pixel_index + 1] = float(j) / max_y;
    fb[pixel_index + 2] = 0.2;
}

int main() {

    // Image
    int image_width = 1200;
    int image_height = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = image_width * image_height;
    size_t fb_size = 3 * num_pixels * sizeof(float);

    float* fb;
    cudaMallocManaged((void**)&fb, fb_size);

    //timer
    clock_t start, stop;
    start = clock();

    //block
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render <<<blocks, threads >>> (fb, image_width, image_height);
   cudaGetLastError();
    cudaDeviceSynchronize();

    // Render
    FILE* fp = fopen("output.ppm", "wb");
    fprintf(fp, "P3\n%d %d\n255\n", image_width, image_height);
  //  std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j) {
      //  std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            size_t pixel_index = j * 3 * image_width + i * 3;

            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            fprintf(fp, "%d %d %d\n", ir, ig, ib);
        }
    }
    fclose(fp);
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";
    cudaFree(fb);
}