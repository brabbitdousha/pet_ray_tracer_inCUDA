#include <iostream>
#include <fstream>
#include <time.h>
#include <cstdio>
#include "vec3.h"

int main() {
    
    // Image
    int image_width = 1200;
    int image_height = 600;

    // Render
    FILE* fp = fopen("output.ppm", "wb");
    fprintf(fp, "P3\n%d %d\n255\n",image_width,image_height);
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    //timer
   clock_t start, stop;

    start = clock();

    for (int j = image_height - 1; j >= 0; --j) {
    	std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            auto r=double(i)/(image_width-1);
            auto g=double(j)/(image_height-1);
            auto b=0.25;
            
            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);
    
            fprintf(fp,"%d %d %d\n",ir,ig,ib);
        }
    }
    //timer
     stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    fclose(fp);
}