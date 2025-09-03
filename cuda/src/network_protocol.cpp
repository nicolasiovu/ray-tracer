#include "network_protocol.hpp"
#include <iostream>

bool NetworkUtils::initialize_networking() {
#ifdef _WIN32
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0) {
        std::cerr << "WSAStartup failed: " << result << std::endl;
        return false;
    }
    return true;
#else
    return true; // No initialization needed on Unix
#endif
}

void NetworkUtils::cleanup_networking() {
#ifdef _WIN32
    WSACleanup();
#endif
}

bool NetworkUtils::set_socket_nonblocking(SOCKET sock) {
#ifdef _WIN32
    u_long mode = 1;
    return (ioctlsocket(sock, FIONBIO, &mode) == 0);
#else
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags == -1) return false;
    return (fcntl(sock, F_SETFL, flags | O_NONBLOCK) == 0);
#endif
}

bool NetworkUtils::send_all(SOCKET sock, const void* data, size_t size) {
    const char* ptr = static_cast<const char*>(data);
    size_t total_sent = 0;
    
    while (total_sent < size) {
        int sent = send(sock, ptr + total_sent, size - total_sent, 0);
        if (sent == SOCKET_ERROR) {
#ifdef _WIN32
            int error = WSAGetLastError();
            if (error == WSAEWOULDBLOCK) continue;
#else
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
#endif
            std::cerr << "Send error: " << sent << std::endl;
            return false;
        }
        total_sent += sent;
    }
    return true;
}

bool NetworkUtils::recv_all(SOCKET sock, void* data, size_t size) {
    char* ptr = static_cast<char*>(data);
    size_t total_received = 0;
    
    while (total_received < size) {
        int received = recv(sock, ptr + total_received, size - total_received, 0);
        if (received == SOCKET_ERROR) {
#ifdef _WIN32
            int error = WSAGetLastError();
            if (error == WSAEWOULDBLOCK) continue;
#else
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
#endif
            std::cerr << "Recv error: " << received << std::endl;
            return false;
        }
        if (received == 0) {
            std::cerr << "Connection closed by peer" << std::endl;
            return false;
        }
        total_received += received;
    }
    return true;
}

bool NetworkUtils::send_message(SOCKET sock, MessageType type, const void* data, uint32_t size) {
    MessageHeader header(type, size);
    
    // Send header
    if (!send_all(sock, &header, sizeof(header))) {
        return false;
    }
    
    // Send data if any
    if (size > 0 && data != nullptr) {
        if (!send_all(sock, data, size)) {
            return false;
        }
    }
    
    return true;
}

bool NetworkUtils::recv_message(SOCKET sock, MessageHeader& header, void* data, uint32_t max_size) {
    // Receive header
    if (!recv_all(sock, &header, sizeof(header))) {
        return false;
    }
    
    // Receive data if any
    if (header.size > 0) {
        if (header.size > max_size) {
            std::cerr << "Message too large: " << header.size << " > " << max_size << std::endl;
            return false;
        }
        if (!recv_all(sock, data, header.size)) {
            return false;
        }
    }
    
    return true;
}