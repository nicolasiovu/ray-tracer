#include <iostream>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include "renderer.cuh"
#include "interactive_camera.cuh"
#include "network_protocol.hpp"

#pragma comment(lib, "ws2_32.lib")

class RayTracerServer {
public:    
    SOCKET server_socket;
    SOCKET client_socket;
    InteractiveCamera camera;
    uint32_t* pixel_buffer;
    const int WIDTH = 800;
    const int HEIGHT = 600;
    bool running = true;
    
    RayTracerServer() : server_socket(INVALID_SOCKET), client_socket(INVALID_SOCKET) {
        camera = InteractiveCamera(Vec3(13.0f, 2.0f, 3.0f));
        camera.aspect_ratio = (float)WIDTH / HEIGHT;
        camera.mouse_sensitivity = 0.5f;
        pixel_buffer = new uint32_t[WIDTH * HEIGHT];
    }
    
    ~RayTracerServer() {
        cleanup();
        cleanup_server();
        delete[] pixel_buffer;
    }
    
    std::string get_socket_error_string(int error_code) {
        char* error_string = nullptr;
        FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                      nullptr, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPSTR)&error_string, 0, nullptr);
        std::string result = error_string ? error_string : "Unknown error";
        if (error_string) LocalFree(error_string);
        return result;
    }
    
    bool initialize() {
        // Initialize CUDA
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        std::cout << "Running on GPU: " << props.name << std::endl;
        
        // Initialize networking
        if (!NetworkUtils::initialize_networking()) {
            std::cerr << "Failed to initialize networking" << std::endl;
            return false;
        }
        
        return create_server_socket();
    }
    
    bool create_server_socket() {
        // Create server socket
        server_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (server_socket == INVALID_SOCKET) {
            int error_code = WSAGetLastError();
            std::cerr << "Failed to create server socket: " << get_socket_error_string(error_code) << std::endl;
            return false;
        }
        
        // Set socket options
        int opt = 1;
        if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, 
                       reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
            int error_code = WSAGetLastError();
            std::cerr << "Failed to set SO_REUSEADDR: " << get_socket_error_string(error_code) << std::endl;
            return false;
        }
        
        // Set SO_LINGER to ensure socket closes cleanly
        struct linger lng;
        lng.l_onoff = 1;
        lng.l_linger = 0;
        if (setsockopt(server_socket, SOL_SOCKET, SO_LINGER, 
                       reinterpret_cast<const char*>(&lng), sizeof(lng)) < 0) {
            std::cerr << "Warning: Failed to set SO_LINGER option" << std::endl;
        }
        
        // Bind socket
        sockaddr_in server_addr = {};
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(DEFAULT_PORT);
        
        if (bind(server_socket, reinterpret_cast<sockaddr*>(&server_addr), 
                 sizeof(server_addr)) == SOCKET_ERROR) {
            int error_code = WSAGetLastError();
            std::cerr << "Failed to bind socket to port " << DEFAULT_PORT 
                     << ": " << get_socket_error_string(error_code) << std::endl;
            return false;
        }
        
        // Listen for connections
        if (listen(server_socket, 1) == SOCKET_ERROR) {
            int error_code = WSAGetLastError();
            std::cerr << "Failed to listen on socket: " << get_socket_error_string(error_code) << std::endl;
            return false;
        }
        
        std::cout << "Server listening on port " << DEFAULT_PORT << std::endl;
        return true;
    }
    
    bool recreate_server_socket() {
        std::cout << "Recreating server socket..." << std::endl;
        
        // Close existing socket if valid
        if (server_socket != INVALID_SOCKET) {
            closesocket(server_socket);
            server_socket = INVALID_SOCKET;
        }
        
        // Wait a moment for the socket to fully close
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        return create_server_socket();
    }
    
    bool wait_for_client() {
        std::cout << "Waiting for client connection..." << std::endl;
        
        // Use select to check if server socket is ready and handle timeouts
        fd_set readfds;
        struct timeval timeout;
        
        while (true) {
            FD_ZERO(&readfds);
            FD_SET(server_socket, &readfds);
            
            // Set timeout for select (1 second)
            timeout.tv_sec = 1;
            timeout.tv_usec = 0;
            
            int select_result = select(0, &readfds, NULL, NULL, &timeout);
            
            if (select_result == SOCKET_ERROR) {
                int error_code = WSAGetLastError();
                std::cerr << "Server socket select error: " << get_socket_error_string(error_code) << std::endl;
                
                // Try to recreate socket on critical errors
                if (error_code == WSAEBADF || error_code == WSAEINVAL || error_code == WSAENOTSOCK) {
                    if (!recreate_server_socket()) {
                        return false;
                    }
                    continue; // Try again with new socket
                }
                return false;
            }
            
            if (select_result == 0) {
                // Timeout - continue waiting but allow checking for shutdown signals
                continue;
            }
            
            // Socket is ready for accept
            if (FD_ISSET(server_socket, &readfds)) {
                break;
            }
        }
        
        sockaddr_in client_addr;
        int client_addr_len = sizeof(client_addr);
        
        // Try to accept with error handling
        client_socket = accept(server_socket, reinterpret_cast<sockaddr*>(&client_addr), 
                              &client_addr_len);
        
        if (client_socket == INVALID_SOCKET) {
            int error_code = WSAGetLastError();
            std::cerr << "Failed to accept client connection: " << get_socket_error_string(error_code) << std::endl;
            
            // If we get a critical error, try to recreate the server socket
            if (error_code == WSAEBADF || error_code == WSAEINVAL || error_code == WSAENOTSOCK) {
                std::cout << "Server socket corrupted, attempting to recreate..." << std::endl;
                if (!recreate_server_socket()) {
                    return false;
                }
                // Don't try again immediately - let the caller retry
                return false;
            }
            return false;
        }
        
        // Get client IP for logging
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(client_addr.sin_addr), client_ip, INET_ADDRSTRLEN);
        std::cout << "Client connected from: " << client_ip << std::endl;
        
        return true;
    }
    
    void apply_input_to_camera(const InputState& input) {
        // Apply keyboard input
        for (int i = 0; i < 512 && i < SDL_NUM_SCANCODES; i++) {
            camera.keys[i] = input.keys[i];
        }
        
        // Apply mouse input
        if (input.mouse_captured) {
            float x_offset = input.mouse_x_rel * camera.mouse_sensitivity;
            float y_offset = -input.mouse_y_rel * camera.mouse_sensitivity;
            
            camera.yaw += x_offset;
            camera.pitch += y_offset;
            
            if (camera.pitch > 89.0f) camera.pitch = 89.0f;
            if (camera.pitch < -89.0f) camera.pitch = -89.0f;
        }
        
        camera.mouse_captured = input.mouse_captured;
        
        // Update camera based on input
        camera.process_keyboard(input.delta_time);
        camera.update_camera_vectors();
    }
    
    bool is_client_connected() {
        if (client_socket == INVALID_SOCKET) {
            return false;
        }
        
        // Test connection by sending a zero-byte message
        int result = send(client_socket, "", 0, 0);
        if (result == SOCKET_ERROR) {
            int error = WSAGetLastError();
            return (error != WSAECONNRESET && error != WSAECONNABORTED && 
                    error != WSAENOTCONN && error != WSAESHUTDOWN);
        }
        return true;
    }
    
    void run() {
        std::cout << "Starting render loop..." << std::endl;
        
        auto last_time = std::chrono::high_resolution_clock::now();
        int frame_count = 0;
        running = true; // Reset running flag for new client
        
        while (running) {
            auto current_time = std::chrono::high_resolution_clock::now();
            float delta_time = std::chrono::duration<float>(current_time - last_time).count();
            last_time = current_time;
            
            // Check for input from client
            MessageHeader header;
            InputState input_state;
            
            if (NetworkUtils::recv_message(client_socket, header, &input_state, sizeof(input_state))) {
                switch (header.type) {
                    case MSG_INPUT_DATA:
                        apply_input_to_camera(input_state);
                        break;
                    case MSG_DISCONNECT:
                        std::cout << "Client requested disconnect" << std::endl;
                        running = false;
                        continue;
                    default:
                        break;
                }
            } else {
                // Check if client disconnected unexpectedly
                if (!is_client_connected()) {
                    std::cout << "Client disconnected unexpectedly" << std::endl;
                    running = false;
                    continue;
                }
            }
            
            // Render frame
            render_with_camera(pixel_buffer, WIDTH, HEIGHT, camera);
            
            // Compress and send frame to client
            static std::vector<char> compressed_buffer;
            if (!NetworkUtils::compress_data(pixel_buffer, PIXEL_BUFFER_SIZE, compressed_buffer)) {
                std::cerr << "Failed to compress frame data" << std::endl;
                continue;
            }
            
            // Send compressed frame to client
            if (!NetworkUtils::send_message(client_socket, MSG_FRAME_DATA, 
                                          compressed_buffer.data(), compressed_buffer.size())) {
                std::cerr << "Failed to send frame data, client likely disconnected" << std::endl;
                running = false;
                continue;
            }
            
            // Print status every 60 frames
            frame_count++;
            if (frame_count % 60 == 0) {
                std::cout << "Rendered " << frame_count << " frames. "
                         << "Camera: (" << camera.position.x << ", " 
                         << camera.position.y << ", " << camera.position.z << ")" << std::endl;
            }
        }
        
        std::cout << "Render loop ended for current client" << std::endl;
    }
    
    void cleanup() {
        if (client_socket != INVALID_SOCKET) {
            // Proper socket shutdown sequence
            std::cout << "Cleaning up client connection..." << std::endl;
            shutdown(client_socket, SD_BOTH);
            closesocket(client_socket);
            client_socket = INVALID_SOCKET;
        }
        cleanup_scene(); // Clean up CUDA resources
        
        // Add a small delay to ensure socket cleanup completes
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    void cleanup_server() {
        std::cout << "Cleaning up server..." << std::endl;
        if (server_socket != INVALID_SOCKET) {
            closesocket(server_socket);
            server_socket = INVALID_SOCKET;
        }
        NetworkUtils::cleanup_networking();
    }
    
    void stop() {
        running = false;
    }
};

int main(int argc, char** argv) {
    std::cout << "CUDA Raytracer Server (Windows)" << std::endl;
    RayTracerServer server;
    
    if (!server.initialize()) {
        std::cerr << "Server initialization failed" << std::endl;
        server.cleanup_server();
        return -1;
    }
    
    int consecutive_failures = 0;
    const int MAX_FAILURES = 5;
    
    std::cout << "Server initialized successfully. Starting main loop..." << std::endl;
    
    while (true) {
        std::cout << "\n=== Waiting for new client connection ===" << std::endl;
        
        if (!server.wait_for_client()) {
            consecutive_failures++;
            std::cerr << "Failed to accept client (attempt " << consecutive_failures 
                     << "/" << MAX_FAILURES << ")" << std::endl;
            
            if (consecutive_failures >= MAX_FAILURES) {
                std::cerr << "Too many consecutive failures, shutting down server" << std::endl;
                break;
            }
            
            // Wait before retrying
            std::cout << "Waiting " << (consecutive_failures * 2) << " seconds before retry..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(consecutive_failures * 2));
            continue;
        }
        
        // Reset failure counter on successful connection
        consecutive_failures = 0;
        
        std::cout << "\n=== Client Connected Successfully ===" << std::endl;
        std::cout << "Controls (handled by client):" << std::endl;
        std::cout << "  WASD - Move around" << std::endl;
        std::cout << "  QE - Move up/down" << std::endl;
        std::cout << "  Hold Left Mouse - Look around" << std::endl;
        std::cout << "  Shift - Move faster" << std::endl;
        std::cout << "  +/- - Zoom in/out" << std::endl;
        std::cout << "  ESC - Exit" << std::endl;
        std::cout << "=======================================" << std::endl;
        
        // Run the server for this client
        server.run();
        
        // After run() returns, clean up client socket but keep server running
        server.cleanup();
        std::cout << "\n=== Client Disconnected ===" << std::endl;
    }
    
    std::cout << "Server shutting down..." << std::endl;
    server.cleanup_server();
    return 0;
}