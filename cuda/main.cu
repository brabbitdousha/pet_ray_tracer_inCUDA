#include <iostream>
#include <fstream>
#include <time.h>
#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "ray.h"

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

__device__ vec3 color(const ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3* fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    fb[pixel_index] = color(r);
  
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
    size_t fb_size = num_pixels * sizeof(vec3);

    vec3* fb;
    cudaMallocManaged((void**)&fb, fb_size);

    //timer
    clock_t start, stop;
    start = clock();

    //block
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render <<<blocks, threads >>> (fb, image_width, image_height,vec3(-2.0, -1.0, -1.0),
        vec3(4.0, 0.0, 0.0),
        vec3(0.0, 2.0, 0.0),
        vec3(0.0, 0.0, 0.0));
   cudaGetLastError();
    cudaDeviceSynchronize();

    // Render
    FILE* fp = fopen("output.ppm", "wb");
    fprintf(fp, "P3\n%d %d\n255\n", image_width, image_height);
  //  std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j) {
      //  std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            size_t pixel_index = j  * image_width + i ;

           int ir = int(255.99*fb[pixel_index].x());
            int ig = int(255.99*fb[pixel_index].y());
            int ib = int(255.99*fb[pixel_index].z());

            fprintf(fp, "%d %d %d\n", ir, ig, ib);
        }
    }
    fclose(fp);
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";
    cudaFree(fb);
}