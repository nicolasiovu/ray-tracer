#include "renderer.cuh"
#include "vec3.cuh"
#include "ray.cuh"
#include "hitrecord.cuh"
#include "material.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include "cuda_utils.cuh"
#include "cuda_bvh.cuh" 
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include "interactive_camera.cuh"

// Global device memory for scene data
static Sphere* d_spheres = nullptr;
static Material* d_materials = nullptr;
static BVHNode* d_bvh_nodes = nullptr;
static curandState* d_rand_state = nullptr;
static bool scene_initialized = false;
static int scene_num_spheres = 0;
static int scene_num_materials = 0;
static int scene_num_bvh_nodes = 0;

void create_scene(Sphere** h_spheres, Material** h_materials, int& num_spheres, int& num_materials) {
    // Count objects first
    // (Just an example scene, will make various scene options later)
    num_spheres = 4 + 22 * 22; 
    num_materials = 4 + 22 * 22; 
    
    *h_spheres = new Sphere[num_spheres];
    *h_materials = new Material[num_materials];
    
    int sphere_idx = 0;
    int mat_idx = 0;
    
    // Ground sphere
    (*h_materials)[mat_idx] = {LAMBERTIAN, Color(0.5f, 0.5f, 0.5f), 0.0f, 0.0f};
    (*h_spheres)[sphere_idx] = {Point3(0, -1000, 0), 1000.0f, mat_idx};
    sphere_idx++;
    mat_idx++;
    
    // Random small spheres
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = (float)rand() / RAND_MAX;
            Point3 center(a + 0.9f * (float)rand() / RAND_MAX, 0.2f, b + 0.9f * (float)rand() / RAND_MAX);
            
            if ((center - Point3(4, 0.2f, 0)).length() > 0.9f) {
                if (choose_mat < 0.8f) {
                    // Diffuse
                    Color albedo = Color((float)rand() / RAND_MAX * (float)rand() / RAND_MAX,
                                       (float)rand() / RAND_MAX * (float)rand() / RAND_MAX,
                                       (float)rand() / RAND_MAX * (float)rand() / RAND_MAX);
                    (*h_materials)[mat_idx] = {LAMBERTIAN, albedo, 0.0f, 0.0f};
                } else if (choose_mat < 0.95f) {
                    // Metal
                    Color albedo = Color(0.5f + 0.5f * (float)rand() / RAND_MAX,
                                       0.5f + 0.5f * (float)rand() / RAND_MAX,
                                       0.5f + 0.5f * (float)rand() / RAND_MAX);
                    float fuzz = 0.5f * (float)rand() / RAND_MAX;
                    (*h_materials)[mat_idx] = {METAL, albedo, fuzz, 0.0f};
                } else {
                    // Glass
                    (*h_materials)[mat_idx] = {DIELECTRIC, Color(1.0f, 1.0f, 1.0f), 0.0f, 1.5f};
                }
                
                (*h_spheres)[sphere_idx] = {center, 0.2f, mat_idx};
                sphere_idx++;
                mat_idx++;
            }
        }
    }
    
    // Three large spheres
    (*h_materials)[mat_idx] = {DIELECTRIC, Color(1.0f, 1.0f, 1.0f), 0.0f, 1.5f};
    (*h_spheres)[sphere_idx] = {Point3(0, 1, 0), 1.0f, mat_idx};
    sphere_idx++;
    mat_idx++;
    
    (*h_materials)[mat_idx] = {LAMBERTIAN, Color(0.4f, 0.2f, 0.1f), 0.0f, 0.0f};
    (*h_spheres)[sphere_idx] = {Point3(-4, 1, 0), 1.0f, mat_idx};
    sphere_idx++;
    mat_idx++;
    
    (*h_materials)[mat_idx] = {METAL, Color(0.7f, 0.6f, 0.5f), 0.0f, 0.0f};
    (*h_spheres)[sphere_idx] = {Point3(4, 1, 0), 1.0f, mat_idx};
    sphere_idx++;
    mat_idx++;
    
    num_spheres = sphere_idx;
    num_materials = mat_idx;
}

void initialize_scene(int width, int height) {
    if (scene_initialized) return;
    
    Sphere* h_spheres;
    Material* h_materials;
    create_scene(&h_spheres, &h_materials, scene_num_spheres, scene_num_materials);
    
    std::vector<Sphere> sphere_vector(h_spheres, h_spheres + scene_num_spheres);
    
    std::cout << "Building BVH..." << std::endl;
    BVHBuilder bvh_builder(sphere_vector);
    
    std::vector<Sphere> reordered_spheres = bvh_builder.get_reordered_spheres();
    scene_num_bvh_nodes = bvh_builder.nodes.size();
    
    std::cout << "BVH built with " << scene_num_bvh_nodes << " nodes" << std::endl;
    
    // Allocate device memory
    cudaMalloc(&d_spheres, scene_num_spheres * sizeof(Sphere));
    cudaMalloc(&d_materials, scene_num_materials * sizeof(Material));
    cudaMalloc(&d_bvh_nodes, scene_num_bvh_nodes * sizeof(BVHNode));
    cudaMalloc(&d_rand_state, width * height * sizeof(curandState));
    
    // Copy to device
    cudaMemcpy(d_spheres, reordered_spheres.data(), scene_num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice);
    cudaMemcpy(d_materials, h_materials, scene_num_materials * sizeof(Material), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bvh_nodes, bvh_builder.nodes.data(), scene_num_bvh_nodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
    
    // Initialize random states
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    init_random_states<<<grid, block>>>(d_rand_state, width, height, time(nullptr));
    cudaDeviceSynchronize();
    
    // Clean up host memory
    delete[] h_spheres;
    delete[] h_materials;
    
    scene_initialized = true;
    std::cout << "Scene initialized with " << scene_num_spheres << " spheres, " 
              << scene_num_materials << " materials, and " << scene_num_bvh_nodes << " BVH nodes" << std::endl;
}

void render_with_camera(uint32_t *pixels, int width, int height, const InteractiveCamera& interactive_cam) {
    initialize_scene(width, height);
    
    // Convert interactive camera to CUDA camera
    Camera cam = interactive_cam.to_cuda_camera();
    
    // Render parameters
    // Higher SPP for high quality, but very slow
    const int samples_per_pixel = 12; 
    const int max_depth = 10;
    
    // Allocate device memory for pixels
    uint32_t *d_pixels;
    size_t size = width * height * sizeof(uint32_t);
    cudaMalloc(&d_pixels, size);
    
    // Launch BVH-accelerated kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    render_kernel_bvh<<<grid, block>>>(d_pixels, width, height, samples_per_pixel, max_depth,
                                      cam, d_bvh_nodes, d_spheres, d_materials, d_rand_state);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(pixels, d_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}

void cleanup_scene() {
    if (scene_initialized) {
        cudaFree(d_spheres);
        cudaFree(d_materials);
        cudaFree(d_bvh_nodes);
        cudaFree(d_rand_state);
        scene_initialized = false;
    }
}
