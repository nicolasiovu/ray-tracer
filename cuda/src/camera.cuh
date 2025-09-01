#pragma once
#include "vec3.cuh"
#include "cuda_utils.cuh"
#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

#ifndef PI
#define PI 3.14159265358979323846f
#endif

struct Camera {
    Point3 lookfrom;
    Point3 lookat;
    Vec3 vup;

    float vfov;
    float aspect_ratio;
    float aperture;
    float focus_dist;

    Point3 origin;
    Point3 lower_left_corner;

    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;

    float lens_radius;
};

#ifdef __CUDACC__

__device__ Ray get_ray(const Camera& cam, float s, float t, curandState* state) {
    Vec3 rd = cam.lens_radius * random_in_unit_disk(state);
    Vec3 offset = cam.u * rd.x + cam.v * rd.y;

    return Ray(cam.origin + offset, cam.lower_left_corner + s * cam.horizontal + t * cam.vertical - cam.origin - offset);
}

#endif
