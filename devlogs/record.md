# 2022/10/28

## Output
width:1200 height:800</br>
>GPU: nvdia GeForce MX250</br>
>BLOCK: 16X16</br>
>SAMPLES:64 </br>
>TOOK：23min</br>
                     

>GPU: nvdia RTX3060</br>
>BLOCK: 16X16</br>
>SAMPLES:64 </br>
>TOOK：167.286seconds</br>

</br>

![img](https://developer-blogs.nvidia.com/wp-content/uploads/2018/10/chapter12-768x384.jpg)

Decide to turn this baby tracer into a bidirectional path tracing..
now start with a few simple things

- [x] Cmakelist
- [ ] obj reading and drawing
- [ ] the basic BVH
- [ ] optimize the BVH using morton code