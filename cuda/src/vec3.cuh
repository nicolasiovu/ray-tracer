#pragma once
#include <cmath>

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}

    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const { 
        return Vec3(x + v.x, y + v.y, z + v.z); 
    }

    __host__ __device__ Vec3 operator-(const Vec3& v) const { 
        return Vec3(x - v.x, y - v.y, z - v.z); 
    }

    __host__ __device__ Vec3 operator*(float t) const { 
        return Vec3(x * t, y * t, z * t); 
    }

    __host__ __device__ Vec3 operator*(const Vec3& v) const { 
        return Vec3(x * v.x, y * v.y, z * v.z); 
    }

    __host__ __device__ Vec3 operator/(float t) const { 
        return Vec3(x / t, y / t, z / t); 
    }

    __host__ __device__ Vec3 operator-() const { 
        return Vec3(-x, -y, -z); 
    }

    __host__ __device__ float operator[](int i) const { 
        if (i == 0) return x; 
        if (i == 1) return y; 
        return z; 
    }

    __host__ __device__ float& operator[](int i) { 
        if (i == 0) return x; 
        if (i == 1) return y; 
        return z; 
    }

    __host__ __device__ float length() const { 
        return sqrtf(x*x + y*y + z*z); 
    }

    __host__ __device__ float length_squared() const { 
        return x*x + y*y + z*z; 
    }

    __host__ __device__ Vec3 unit_vector() const { 
        float len = length(); 
        return Vec3(x/len, y/len, z/len); 
    }

    __host__ __device__ float dot(const Vec3& v) const { 
        return x * v.x + y * v.y + z * v.z; 
    }

    __host__ __device__ Vec3 cross(const Vec3& v) const { 
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); 
    }
};

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) { 
    return v * t; 
}

using Point3 = Vec3;
using Color = Vec3;
