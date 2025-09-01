#pragma once
#include "vec3.cuh"
#include "ray.cuh"
#include "hitrecord.cuh"
#include "cuda_utils.cuh"
#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

enum MaterialType {
    LAMBERTIAN = 0,
    METAL = 1,
    DIELECTRIC = 2
};

struct Material {
    MaterialType type;
    Color albedo;
    float fuzz;
    float ref_idx;
};

#ifdef __CUDACC__
__device__ bool scatter(const Material& mat, const Ray& r_in, const HitRecord& rec, 
                       Color& attenuation, Ray& scattered, curandState* state) {
    switch (mat.type) {
        case LAMBERTIAN: {
            Vec3 scatter_direction = rec.normal + random_unit_vector(state);
            if (scatter_direction.length_squared() < 1e-8f) {
                scatter_direction = rec.normal;
            }
            scattered = Ray(rec.p, scatter_direction, r_in.time());
            attenuation = mat.albedo;
            return true;
        }
        case METAL: {
            Vec3 reflected = reflect(r_in.direction().unit_vector(), rec.normal);
            scattered = Ray(rec.p, reflected + mat.fuzz * random_in_unit_sphere(state), r_in.time());
            attenuation = mat.albedo;
            return scattered.direction().dot(rec.normal) > 0;
        }
        case DIELECTRIC: {
            attenuation = Color(1.0f, 1.0f, 1.0f);
            float refraction_ratio = rec.front_face ? (1.0f / mat.ref_idx) : mat.ref_idx;
            Vec3 unit_direction = r_in.direction().unit_vector();
            float cos_theta = fminf((-unit_direction).dot(rec.normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
            Vec3 direction;
            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(state)) {
                direction = reflect(unit_direction, rec.normal);
            } else {
                direction = refract(unit_direction, rec.normal, refraction_ratio);
            }
            scattered = Ray(rec.p, direction, r_in.time());
            return true;
        }
    }
    return false;
}
#endif
