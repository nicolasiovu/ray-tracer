#pragma once
#include "vec3.cuh"

struct Ray {
    Point3 orig;
    Vec3 dir;
    float tm;

    __host__ __device__ Ray() {}

    __host__ __device__ Ray(const Point3& origin, const Vec3& direction, float time = 0.0f)
        : orig(origin), dir(direction), tm(time) {}
    
    __host__ __device__ Point3 origin() const { 
        return orig; 
    }

    __host__ __device__ Vec3 direction() const { 
        return dir;
    }

    __host__ __device__ float time() const { 
        return tm;
    }

    __host__ __device__ Point3 at(float t) const { 
        return orig + t * dir;
    }
};
