#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>
struct int3
{
	int x,y,z;
};
int3 write_color(color pixel_color,int samples_per_pixel) {
    auto r=pixel_color.x();
    auto g=pixel_color.y();
    auto b=pixel_color.z();

    // Divide the color by the number of samples.
    auto scale=1.0/samples_per_pixel;
    r=sqrt(scale*r);
    g=sqrt(scale*g);
    b=sqrt(scale*b);
    
    // Write the translated [0,255] value of each color component.
    int3 temp;
    temp.x=static_cast<int>(256 * clamp(r,0.0,0.999));
    temp.y=static_cast<int>(256 * clamp(g,0.0,0.999));
    temp.z=static_cast<int>(256 * clamp(b,0.0,0.999));
    return temp;
}

#endif
