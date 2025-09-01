#pragma once
#include "vec3.cuh"
#include "ray.cuh"

struct HitRecord {
    Point3 p;
    Vec3 normal;
    float t;
    bool front_face;
    int material_id;

#ifdef __CUDACC__

    __device__ void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = r.direction().dot(outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
    
#endif
};
