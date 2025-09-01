#include <SDL2/SDL.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "renderer.cuh"
#include "interactive_camera.cuh"

int main(int argc, char **argv) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    std::cout << "Running on GPU: " << props.name << "\n";
    
    const int WIDTH = 800;
    const int HEIGHT = 600;
    
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL init failed: " << SDL_GetError() << std::endl;
        return -1;
    }
    
    SDL_Window *window = SDL_CreateWindow("CUDA Raytracer - WASD to move, mouse to look, +/- to zoom",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          WIDTH, HEIGHT, 0);
    if (!window) {
        std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return -1;
    }
    
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);
    if (!renderer) {
        std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }
    
    SDL_Texture *texture = SDL_CreateTexture(renderer,
                                             SDL_PIXELFORMAT_ARGB8888,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             WIDTH, HEIGHT);
    if (!texture) {
        std::cerr << "Texture creation failed: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }
    
    uint32_t *pixels = new uint32_t[WIDTH * HEIGHT];
    
    InteractiveCamera camera(Vec3(13.0f, 2.0f, 3.0f));
    camera.aspect_ratio = (float)WIDTH / HEIGHT;
    
    SDL_SetRelativeMouseMode(SDL_FALSE);
    
    bool running = true;
    SDL_Event event;
    
    // Timing for smooth movement
    auto last_time = std::chrono::high_resolution_clock::now();

    // For FPS tracking
    int frames = 0;
    auto fps_last_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Controls:" << std::endl;
    std::cout << "WASD - Move around" << std::endl;
    std::cout << "QE - Move up/down" << std::endl;
    std::cout << "Hold Left Mouse - Look around" << std::endl;
    std::cout << "Shift - Move faster" << std::endl;
    std::cout << "+/- - Zoom in/out" << std::endl;
    std::cout << "ESC - Exit" << std::endl;
    std::cout << "Starting render loop..." << std::endl;
    
    while (running) {
        // Calculate delta time for smooth movement
        auto current_time = std::chrono::high_resolution_clock::now();
        float delta_time = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;
        
        // Handle events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
            if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) {
                running = false;
            }
            
            // Let camera handle the event
            camera.handle_event(event);
        }
        
        // Update camera based on input
        camera.process_keyboard(delta_time);
        
        // Render the scene with updated camera
        render_with_camera(pixels, WIDTH, HEIGHT, camera);
        
        // Update display
        SDL_UpdateTexture(texture, nullptr, pixels, WIDTH * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);

        // FPS counter
        frames++;
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(now - fps_last_time).count();
        if (elapsed >= 1.0f) { // Print once per second
            std::cout << "FPS: " << frames / elapsed << std::endl;
            frames = 0;
            fps_last_time = now;
        }
        
        static int frame_count = 0;
        if (++frame_count % 60 == 0) {  // Update title every 60 frames
            std::string title = "CUDA Raytracer - " + camera.get_info_string();
            SDL_SetWindowTitle(window, title.c_str());
        }
        
        SDL_RenderPresent(renderer);
    }
    
    // Cleanup
    cleanup_scene();
    delete[] pixels;
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    std::cout << "Cleanup complete." << std::endl;
    return 0;
}