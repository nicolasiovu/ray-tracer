#pragma once
#include <vector>
#include <algorithm>
#include "vec3.cuh"
#include "ray.cuh"
#include "hitrecord.cuh"
#include "material.cuh"
#include "sphere.cuh"

struct Interval {
    float min, max;
    
    __host__ __device__ Interval() : min(FLT_MAX), max(-FLT_MAX) {}

    __host__ __device__ Interval(float min, float max) : min(min), max(max) {}

    __host__ __device__ Interval(const Interval& a, const Interval& b) {
        min = fminf(a.min, b.min);
        max = fmaxf(a.max, b.max);
    }
    
    __host__ __device__ float size() const { 
        return max - min; 
    }

    __host__ __device__ bool contains(float x) const { 
        return min <= x && x <= max; 
    }

    __host__ __device__ bool surrounds(float x) const { 
        return min < x && x < max; 
    }
    
    __host__ __device__ float clamp(float x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }
    
    __host__ __device__ static Interval empty() {
        return Interval(FLT_MAX, -FLT_MAX);
    }
    
    __host__ __device__ static Interval universe() {
        return Interval(-FLT_MAX, FLT_MAX);
    }
};

struct BoundingBox {
    Interval x, y, z;
    
    __host__ __device__ BoundingBox() {}

    __host__ __device__ BoundingBox(const Interval& x, const Interval& y, const Interval& z) 
        : x(x), y(y), z(z) {}
    
    __host__ __device__ BoundingBox(const Point3& a, const Point3& b) {
        x = (a.x <= b.x) ? Interval(a.x, b.x) : Interval(b.x, a.x);
        y = (a.y <= b.y) ? Interval(a.y, b.y) : Interval(b.y, a.y);
        z = (a.z <= b.z) ? Interval(a.z, b.z) : Interval(b.z, a.z);
    }
    
    __host__ __device__ BoundingBox(const BoundingBox& box1, const BoundingBox& box2) {
        x = Interval(box1.x, box2.x);
        y = Interval(box1.y, box2.y);
        z = Interval(box1.z, box2.z);
    }
    
    __host__ __device__ const Interval& axis_interval(int n) const {
        if (n == 0) return x;
        if (n == 1) return y;
        return z;
    }
    
    __device__ bool hit(const Ray& r, Interval ray_t) const {
        const Point3& ray_orig = r.origin();
        const Vec3& ray_dir = r.direction();
        
        for (int axis = 0; axis < 3; axis++) {
            const Interval& ax = axis_interval(axis);
            const float adinv = 1.0f / ray_dir[axis];
            
            auto t0 = (ax.min - ray_orig[axis]) * adinv;
            auto t1 = (ax.max - ray_orig[axis]) * adinv;
            
            if (t0 < t1) {
                if (t0 > ray_t.min) ray_t.min = t0;
                if (t1 < ray_t.max) ray_t.max = t1;
            } else {
                if (t1 > ray_t.min) ray_t.min = t1;
                if (t0 < ray_t.max) ray_t.max = t0;
            }
            
            if (ray_t.max <= ray_t.min) return false;
        }
        return true;
    }
    
    __host__ __device__ int longest_axis() const {
        if (x.size() > y.size()) {
            return (x.size() > z.size()) ? 0 : 2;
        } else {
            return (y.size() > z.size()) ? 1 : 2;
        }
    }
};

struct BVHNode {
    BoundingBox bbox;
    int left_child;    // Index to left child, -1 if leaf
    int right_child;   // Index to right child, -1 if leaf
    int sphere_start;  // Start index in sphere array (for leaves)
    int sphere_count;  // Number of spheres (for leaves)
    
    __host__ __device__ BVHNode() : left_child(-1), right_child(-1), sphere_start(0), sphere_count(0) {}
    
    __host__ __device__ bool is_leaf() const { 
        return left_child == -1 && right_child == -1; 
    }
};

struct SphereWithBounds : public Sphere {
    BoundingBox bbox;
    
    __host__ __device__ SphereWithBounds() {}

    __host__ __device__ SphereWithBounds(const Sphere& sphere) : Sphere(sphere) {
        Vec3 r_vec(radius, radius, radius);
        bbox = BoundingBox(center - r_vec, center + r_vec);
    }
};

// Needed to make this non-recursive for CUDA, so just used a stack
__device__ bool bvh_hit(const BVHNode* nodes, const Sphere* spheres, int root_idx, 
                       const Ray& r, float t_min, float t_max, HitRecord& rec) {
    int stack[64]; 
    int stack_ptr = 0;
    stack[stack_ptr++] = root_idx;
    
    bool hit_anything = false;
    float closest_so_far = t_max;
    HitRecord temp_rec;
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode& node = nodes[node_idx];
        
        if (!node.bbox.hit(r, Interval(t_min, closest_so_far))) {
            continue;
        }
        
        if (node.is_leaf()) {
            // Test against all spheres in this leaf
            for (int i = 0; i < node.sphere_count; i++) {
                int sphere_idx = node.sphere_start + i;
                if (spheres[sphere_idx].hit(r, t_min, closest_so_far, temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                    rec.material_id = spheres[sphere_idx].material_id;
                }
            }
        } else {
            // Add children to stack (add right first so left is processed first)
            if (node.right_child != -1) {
                stack[stack_ptr++] = node.right_child;
            }
            if (node.left_child != -1) {
                stack[stack_ptr++] = node.left_child;
            }
        }
    }
    
    return hit_anything;
}


struct BVHBuilder {
    std::vector<SphereWithBounds> spheres;
    std::vector<BVHNode> nodes;
    int next_node_idx;
    
    BVHBuilder(const std::vector<Sphere>& input_spheres) {
        spheres.reserve(input_spheres.size());
        for (const auto& sphere : input_spheres) {
            spheres.push_back(SphereWithBounds(sphere));
        }
        
        nodes.reserve(input_spheres.size() * 2); 
        next_node_idx = 0;
        
        if (!spheres.empty()) {
            build_recursive(0, spheres.size());
        }
    }
    
    int build_recursive(int start, int end) {
        int node_idx = next_node_idx++;
        nodes.resize(next_node_idx);
        BVHNode& node = nodes[node_idx];
        
        // Calculate bounding box for all spheres in range
        if (start < end) {
            node.bbox = spheres[start].bbox;
            for (int i = start + 1; i < end; i++) {
                node.bbox = BoundingBox(node.bbox, spheres[i].bbox);
            }
        }
        
        int object_span = end - start;
        
        if (object_span == 1) {
            // Leaf node
            node.sphere_start = start;
            node.sphere_count = 1;
            return node_idx;
        }
        
        if (object_span <= 4) {
            // Small leaf node (multiple spheres)
            node.sphere_start = start;
            node.sphere_count = object_span;
            return node_idx;
        }
        
        // Split along longest axis
        int axis = node.bbox.longest_axis();

        auto comparator = axis == 0 ? box_x_compare : axis == 1 ? box_y_compare : box_z_compare;
        
        // Sort spheres along the chosen axis
        std::sort(spheres.begin() + start, spheres.begin() + end, comparator);
        
        int mid = start + object_span / 2;
        
        // Build children
        node.left_child = build_recursive(start, mid);
        node.right_child = build_recursive(mid, end);
        
        return node_idx;
    }
    
    std::vector<Sphere> get_reordered_spheres() const {
        std::vector<Sphere> result;
        result.reserve(spheres.size());
        for (const auto& sphere_with_bounds : spheres) {
            result.push_back(static_cast<const Sphere&>(sphere_with_bounds));
        }
        return result;
    }
    
private:

    static bool compare_by_axis(const SphereWithBounds& a, const SphereWithBounds& b, int axis) {
        return a.center[axis] < b.center[axis];
    }

    static bool box_x_compare(const SphereWithBounds& a, const SphereWithBounds& b) {
        return compare_by_axis(a, b, 0);
    }

    static bool box_y_compare(const SphereWithBounds& a, const SphereWithBounds& b) {
        return compare_by_axis(a, b, 1);
    }

    static bool box_z_compare(const SphereWithBounds& a, const SphereWithBounds& b) {
        return compare_by_axis(a, b, 2);
    }
};

__device__ Color ray_color_bvh(const Ray& r, const BVHNode* bvh_nodes, const Sphere* spheres, 
                              const Material* materials, int max_depth, curandState* state) {
    Ray current_ray = r;
    Color current_attenuation = Color(1.0f, 1.0f, 1.0f);
    
    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;
        
        if (bvh_hit(bvh_nodes, spheres, 0, current_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Color attenuation;
            
            if (scatter(materials[rec.material_id], current_ray, rec, attenuation, scattered, state)) {
                current_attenuation = current_attenuation * attenuation;
                current_ray = scattered;
            } else {
                return Color(0, 0, 0);
            }
        } else {
            // Sky gradient
            Vec3 unit_direction = current_ray.direction().unit_vector();
            float t = 0.5f * (unit_direction.y + 1.0f);
            Color sky = (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
            return current_attenuation * sky;
        }
    }
    
    return Color(0, 0, 0);
}

__global__ void render_kernel_bvh(uint32_t* pixels, int width, int height, int samples_per_pixel,
                                 int max_depth, Camera cam, BVHNode* bvh_nodes, Sphere* spheres, 
                                 Material* materials, curandState* rand_state) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;
    
    int pixel_index = y * width + x;
    curandState* local_rand_state = &rand_state[pixel_index];
    
    Color pixel_color(0, 0, 0);
    
    for (int s = 0; s < samples_per_pixel; s++) {
        float u = (float(x) + random_float(local_rand_state)) / float(width);
        float v = (float(height - y - 1) + random_float(local_rand_state)) / float(height);
        
        Ray r = get_ray(cam, u, v, local_rand_state);
        pixel_color = pixel_color + ray_color_bvh(r, bvh_nodes, spheres, materials, max_depth, local_rand_state);
    }
    
    // Average the samples and apply gamma correction
    pixel_color = pixel_color * (1.0f / float(samples_per_pixel));
    pixel_color = Color(sqrtf(pixel_color.x), sqrtf(pixel_color.y), sqrtf(pixel_color.z));
    
    // Clamp values
    pixel_color.x = fminf(pixel_color.x, 1.0f);
    pixel_color.y = fminf(pixel_color.y, 1.0f);
    pixel_color.z = fminf(pixel_color.z, 1.0f);
    
    // Convert to RGBA
    uint8_t r = (uint8_t)(255.0f * pixel_color.x);
    uint8_t g = (uint8_t)(255.0f * pixel_color.y);
    uint8_t b = (uint8_t)(255.0f * pixel_color.z);
    
    pixels[pixel_index] = (255 << 24) | (r << 16) | (g << 8) | b;
}