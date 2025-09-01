#pragma once
#include "vec3.cuh"
#include "ray.cuh"
#include "hitrecord.cuh"

struct Sphere {
    Point3 center;
    float radius;
    int material_id;

#ifdef __CUDACC__
    __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        Vec3 oc = r.origin() - center;
        float a = r.direction().length_squared();
        float half_b = oc.dot(r.direction());
        float c = oc.length_squared() - radius * radius;
        float discriminant = half_b * half_b - a * c;

        if (discriminant < 0) return false;

        float sqrtd = sqrtf(discriminant);
        float root = (-half_b - sqrtd) / a;

        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root) return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);

        Vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        
        return true;
    }
#endif
};
