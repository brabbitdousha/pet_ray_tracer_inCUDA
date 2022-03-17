#include <iostream>
#include <fstream>
#include <time.h>
#include <cstdio>
#include <float.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"

__device__ vec3 color(const ray& r, hittable** world, curandState* local_rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0);
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hittable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //printf("GPU: Hello world! now : %d\n", pixel_index);
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}
#define RND (curand_uniform(&local_rand_state))
__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
            new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                        new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                        new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}
int main() {

    // Image
    int image_width = 1200;
    int image_height = 800;
    int ns = 64;
    int tx = 16;
    int ty = 16;

    std::cerr << "width: " << image_width << "  height: " << image_height << "  samples: " << ns << "\nin" << tx << " X " << ty << " blocks.\n";

    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);
    
    vec3* fb;
    cudaMallocManaged((void**)&fb, fb_size);

    curandState* d_rand_state;
    cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState));
    curandState* d_rand_state2;
    cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState));
    
    //2nd random state is for the world
    rand_init << <1, 1 >> > (d_rand_state2);
    cudaGetLastError();
    cudaDeviceSynchronize();

    hittable** d_list;
    int num_hitables = 22 * 22 + 1 + 3;
    cudaMalloc((void**)&d_list, num_hitables * sizeof(hittable*));
    hittable** d_world;
    cudaMalloc((void**)&d_world, sizeof(hittable*));
    camera** d_camera;
    cudaMalloc((void**)&d_camera, sizeof(camera*));
    create_world << <1, 1 >> > (d_list, d_world, d_camera, image_width, image_height, d_rand_state2);
    cudaGetLastError();
    cudaDeviceSynchronize();

    clock_t start, stop;
    start = clock();
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render <<<blocks, threads >>> (fb, image_width, image_height, ns, d_camera, d_world, d_rand_state);
    cudaGetLastError();
    cudaDeviceSynchronize();

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Render
    FILE* fp = fopen("Final scene.ppm", "wb");
    fprintf(fp, "P3\n%d %d\n255\n", image_width, image_height);
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            size_t pixel_index = j * image_width + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            fprintf(fp, "%d %d %d\n", ir, ig, ib);
        }
    }
    fclose(fp);

    cudaDeviceSynchronize();
    free_world << <1, 1 >> > (d_list, d_world, d_camera);
    cudaGetLastError();
    cudaFree(d_camera);
    cudaFree(d_world);
    cudaFree(d_list);
   cudaFree(d_rand_state);
   cudaFree(d_rand_state2);
    cudaFree(fb);

    cudaDeviceReset();
    std::cerr << "\nDone!!!\n";
}
