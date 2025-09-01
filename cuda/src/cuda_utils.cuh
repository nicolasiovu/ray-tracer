#pragma once
#ifdef __CUDACC__
#include <curand_kernel.h>
#include <cmath>
#include "vec3.cuh"
#include "ray.cuh"
#include "material.cuh"

__device__ float random_float(curandState* state) {
    return curand_uniform(state);
}

__device__ float random_float(curandState* state, float min, float max) {
    return min + (max - min) * random_float(state);
}

__device__ Vec3 random_vec3(curandState* state) {
    return Vec3(random_float(state), random_float(state), random_float(state));
}

__device__ Vec3 random_vec3(curandState* state, float min, float max) {
    return Vec3(random_float(state, min, max), random_float(state, min, max), random_float(state, min, max));
}

__device__ Vec3 random_in_unit_sphere(curandState* state) {
    Vec3 p;
    do {
        p = random_vec3(state, -1.0f, 1.0f);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ Vec3 random_unit_vector(curandState* state) {
    return random_in_unit_sphere(state).unit_vector();
}

__device__ Vec3 random_in_unit_disk(curandState* state) {
    Vec3 p;
    do {
        p = Vec3(random_float(state, -1.0f, 1.0f), random_float(state, -1.0f, 1.0f), 0);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2 * v.dot(n) * n;
}

__device__ Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = fminf((-uv).dot(n), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ float reflectance(float cosine, float ref_idx) {
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__global__ void init_random_states(curandState* state, int width, int height, unsigned long seed) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;
    int pixel_index = y * width + x;
    curand_init(seed + pixel_index, 0, 0, &state[pixel_index]);
}

#endif
