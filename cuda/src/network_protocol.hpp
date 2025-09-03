#pragma once

#include <cstdint>
#include <zstd.h>
#include <vector>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    typedef int socklen_t;
    #define SOCKET_ERROR_AGAIN WSAEWOULDBLOCK
    #define close closesocket
#else
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <errno.h>
    typedef int SOCKET;
    #define INVALID_SOCKET -1
    #define SOCKET_ERROR -1
    #define SOCKET_ERROR_AGAIN EAGAIN
#endif

// Network constants
constexpr int DEFAULT_PORT = 7890;
constexpr int PIXEL_BUFFER_SIZE = 800 * 600 * sizeof(uint32_t);

// Message types
enum MessageType : uint8_t {
    MSG_FRAME_DATA = 1,      // Server -> Client: pixel data
    MSG_INPUT_DATA = 2,      // Client -> Server: input state
    MSG_DISCONNECT = 3,      // Either: disconnect notification
    MSG_PING = 4,           // Either: keep-alive
    MSG_PONG = 5            // Either: keep-alive response
};

// Network message header
struct MessageHeader {
    MessageType type;
    uint32_t size;          // Size of data following this header
    
    MessageHeader() : type(MSG_PING), size(0) {}
    MessageHeader(MessageType t, uint32_t s) : type(t), size(s) {}
};

// Input state structure (Client -> Server)
struct InputState {
    // Key states (using SDL scancodes)
    bool keys[512];         // SDL_NUM_SCANCODES is typically 512
    
    // Mouse state
    float mouse_x_rel;      // Relative mouse movement
    float mouse_y_rel;
    bool mouse_captured;
    bool left_mouse_pressed;
    bool left_mouse_released;
    
    // Timing
    float delta_time;
    
    InputState() {
        // Initialize all keys as not pressed
        for (int i = 0; i < 512; i++) {
            keys[i] = false;
        }
        mouse_x_rel = 0.0f;
        mouse_y_rel = 0.0f;
        mouse_captured = false;
        left_mouse_pressed = false;
        left_mouse_released = false;
        delta_time = 0.016f; // ~60fps default
    }
};

// Camera state structure (for synchronization)
struct CameraState {
    float position[3];      // x, y, z
    float front[3];         // direction vector
    float up[3];            // up vector
    float yaw, pitch;
    float fov;
    float aperture;
    float focus_distance;
    
    CameraState() {
        position[0] = 13.0f; position[1] = 2.0f; position[2] = 3.0f;
        front[0] = 0.0f; front[1] = 0.0f; front[2] = -1.0f;
        up[0] = 0.0f; up[1] = 1.0f; up[2] = 0.0f;
        yaw = -90.0f;
        pitch = 0.0f;
        fov = 45.0f;
        aperture = 0.0f;
        focus_distance = 4.0f;
    }
};

// Network utility functions
class NetworkUtils {
private:
    static const int COMPRESSION_LEVEL = 3;  // Balance between speed and compression
    static std::vector<char> compression_buffer;  // Reusable buffer for compression

public:
    // Helper function to get compression bound
    static size_t get_compress_bound(size_t size) { return ZSTD_compressBound(size); }

public:
    static bool initialize_networking();
    static void cleanup_networking();
    static bool set_socket_nonblocking(SOCKET sock);
    static bool send_all(SOCKET sock, const void* data, size_t size);
    static bool recv_all(SOCKET sock, void* data, size_t size);
    static bool send_message(SOCKET sock, MessageType type, const void* data, uint32_t size);
    static bool recv_message(SOCKET sock, MessageHeader& header, void* data, uint32_t max_size);
    
    // New compression methods
    static bool compress_data(const void* src, size_t src_size, std::vector<char>& dst);
    static bool decompress_data(const void* src, size_t src_size, void* dst, size_t dst_size);
};