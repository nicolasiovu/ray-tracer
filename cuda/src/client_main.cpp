#include <SDL2/SDL.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include "network_protocol.hpp"

class RayTracerClient {
private:
    SOCKET client_socket;
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;
    uint32_t* pixel_buffer;
    
    const int WIDTH = 800;
    const int HEIGHT = 600;
    bool running = true;
    
    // Input state tracking
    InputState input_state;
    bool keys[SDL_NUM_SCANCODES];
    bool mouse_captured = false;
    
public:
    RayTracerClient() : client_socket(INVALID_SOCKET), window(nullptr), 
                       renderer(nullptr), texture(nullptr) {
        pixel_buffer = new uint32_t[WIDTH * HEIGHT];
        
        // Initialize input state
        for (int i = 0; i < SDL_NUM_SCANCODES; i++) {
            keys[i] = false;
        }
    }
    
    ~RayTracerClient() {
        cleanup();
        delete[] pixel_buffer;
    }
    
    bool initialize() {
        // Initialize networking
        if (!NetworkUtils::initialize_networking()) {
            std::cerr << "Failed to initialize networking" << std::endl;
            return false;
        }
        
        // Initialize SDL
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL init failed: " << SDL_GetError() << std::endl;
            return false;
        }
        
        // Create window
        window = SDL_CreateWindow("CUDA Raytracer Client - WASD to move, mouse to look, +/- to zoom",
                                  SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED,
                                  WIDTH, HEIGHT, 0);
        if (!window) {
            std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
            return false;
        }
        
        // Create renderer
        renderer = SDL_CreateRenderer(window, -1, 0);
        if (!renderer) {
            std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
            return false;
        }
        
        // Create texture
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                                   SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
        if (!texture) {
            std::cerr << "Texture creation failed: " << SDL_GetError() << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool connect_to_server(const std::string& server_ip = "127.0.0.1") {
        // Create client socket
        client_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (client_socket == INVALID_SOCKET) {
            std::cerr << "Failed to create client socket" << std::endl;
            return false;
        }
        
        // Connect to server
        sockaddr_in server_addr = {};
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(DEFAULT_PORT);
        
        if (inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr) <= 0) {
            std::cerr << "Invalid server IP address: " << server_ip << std::endl;
            return false;
        }
        
        std::cout << "Connecting to server " << server_ip << ":" << DEFAULT_PORT << "..." << std::endl;
        
        if (connect(client_socket, reinterpret_cast<sockaddr*>(&server_addr), 
                   sizeof(server_addr)) == SOCKET_ERROR) {
            std::cerr << "Failed to connect to server" << std::endl;
            return false;
        }
        
        std::cout << "Connected to server!" << std::endl;
        return true;
    }
    
    void handle_events(float delta_time) {
        SDL_Event event;
        
        // Reset relative mouse movement
        input_state.mouse_x_rel = 0.0f;
        input_state.mouse_y_rel = 0.0f;
        input_state.left_mouse_pressed = false;
        input_state.left_mouse_released = false;
        
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                    
                case SDL_KEYDOWN:
                    if (event.key.keysym.sym == SDLK_ESCAPE) {
                        running = false;
                    }
                    if (event.key.keysym.scancode < SDL_NUM_SCANCODES) {
                        keys[event.key.keysym.scancode] = true;
                    }
                    break;
                    
                case SDL_KEYUP:
                    if (event.key.keysym.scancode < SDL_NUM_SCANCODES) {
                        keys[event.key.keysym.scancode] = false;
                    }
                    break;
                    
                case SDL_MOUSEBUTTONDOWN:
                    if (event.button.button == SDL_BUTTON_LEFT && !mouse_captured) {
                        mouse_captured = true;
                        SDL_SetRelativeMouseMode(SDL_TRUE);
                        input_state.left_mouse_pressed = true;
                    }
                    break;
                    
                case SDL_MOUSEBUTTONUP:
                    if (event.button.button == SDL_BUTTON_LEFT && mouse_captured) {
                        mouse_captured = false;
                        SDL_SetRelativeMouseMode(SDL_FALSE);
                        input_state.left_mouse_released = true;
                    }
                    break;
                    
                case SDL_MOUSEMOTION:
                    if (mouse_captured) {
                        input_state.mouse_x_rel = event.motion.xrel;
                        input_state.mouse_y_rel = event.motion.yrel;
                    }
                    break;
            }
        }
        
        // Update input state
        for (int i = 0; i < SDL_NUM_SCANCODES && i < 512; i++) {
            input_state.keys[i] = keys[i];
        }
        input_state.mouse_captured = mouse_captured;
        input_state.delta_time = delta_time;
    }
    
    bool send_input_to_server() {
        return NetworkUtils::send_message(client_socket, MSG_INPUT_DATA, 
                                        &input_state, sizeof(input_state));
    }
    
    bool receive_frame_from_server() {
        MessageHeader header;
        
        if (!NetworkUtils::recv_message(client_socket, header, pixel_buffer, PIXEL_BUFFER_SIZE)) {
            return false;
        }
        
        if (header.type != MSG_FRAME_DATA) {
            std::cerr << "Unexpected message type: " << static_cast<int>(header.type) << std::endl;
            return false;
        }
        
        if (header.size != PIXEL_BUFFER_SIZE) {
            std::cerr << "Invalid frame size: " << header.size << std::endl;
            return false;
        }
        
        return true;
    }
    
    void render_frame() {
        // Update texture with received pixel data
        SDL_UpdateTexture(texture, nullptr, pixel_buffer, WIDTH * sizeof(uint32_t));
        
        // Render to screen
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }
    
    void run() {
        std::cout << "Starting client render loop..." << std::endl;
        
        auto last_time = std::chrono::high_resolution_clock::now();
        int frame_count = 0;
        auto fps_last_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Controls:" << std::endl;
        std::cout << "WASD - Move around" << std::endl;
        std::cout << "QE - Move up/down" << std::endl;
        std::cout << "Hold Left Mouse - Look around" << std::endl;
        std::cout << "Shift - Move faster" << std::endl;
        std::cout << "+/- - Zoom in/out" << std::endl;
        std::cout << "ESC - Exit" << std::endl;
        
        while (running) {
            // Calculate delta time
            auto current_time = std::chrono::high_resolution_clock::now();
            float delta_time = std::chrono::duration<float>(current_time - last_time).count();
            last_time = current_time;
            
            // Handle SDL events and update input state
            handle_events(delta_time);
            
            if (!running) break;
            
            // Send input to server
            if (!send_input_to_server()) {
                std::cerr << "Failed to send input to server" << std::endl;
                running = false;
                break;
            }
            
            // Receive frame from server
            if (!receive_frame_from_server()) {
                std::cerr << "Failed to receive frame from server" << std::endl;
                running = false;
                break;
            }
            
            // Render frame
            render_frame();
            
            // FPS counter
            frame_count++;
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(now - fps_last_time).count();
            if (elapsed >= 1.0f) {
                std::cout << "Client FPS: " << frame_count / elapsed << std::endl;
                frame_count = 0;
                fps_last_time = now;
            }
            
            // Small delay to prevent excessive CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        // Send disconnect message to server
        NetworkUtils::send_message(client_socket, MSG_DISCONNECT, nullptr, 0);
        
        std::cout << "Client shutting down..." << std::endl;
    }
    
    void cleanup() {
        if (client_socket != INVALID_SOCKET) {
            close(client_socket);
            client_socket = INVALID_SOCKET;
        }
        
        if (texture) {
            SDL_DestroyTexture(texture);
            texture = nullptr;
        }
        
        if (renderer) {
            SDL_DestroyRenderer(renderer);
            renderer = nullptr;
        }
        
        if (window) {
            SDL_DestroyWindow(window);
            window = nullptr;
        }
        
        SDL_Quit();
        NetworkUtils::cleanup_networking();
    }
};

int main(int argc, char** argv) {
    std::cout << "CUDA Raytracer Client" << std::endl;
    
    std::string server_ip = "127.0.0.1"; // Default to localhost
    
    // Allow server IP to be specified as command line argument
    if (argc > 1) {
        server_ip = argv[1];
    }
    
    RayTracerClient client;
    
    if (!client.initialize()) {
        std::cerr << "Client initialization failed" << std::endl;
        return -1;
    }
    
    if (!client.connect_to_server(server_ip)) {
        std::cerr << "Failed to connect to server" << std::endl;
        return -1;
    }
    
    client.run();
    
    return 0;
}