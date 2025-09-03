#pragma once
#include <SDL2/SDL.h>
#include <string>
#include "vec3.cuh"
#include "camera.cuh"

class InteractiveCamera {
public:
    // Camera position and orientation
    Vec3 position;
    Vec3 front;        // Direction camera is looking
    Vec3 up;           // Up vector
    Vec3 right;        // Right vector (calculated)
    Vec3 world_up;     // World up vector (usually 0,1,0)
    
    float fov;
    float aspect_ratio;
    float aperture;
    float focus_distance;
    
    float movement_speed;
    float mouse_sensitivity;
    
    // Mouse state
    float yaw; 
    float pitch; 
    bool first_mouse;
    float last_x, last_y;
    bool mouse_captured;
    
    // Input state
    bool keys[SDL_NUM_SCANCODES];
    
    InteractiveCamera(Vec3 pos = Vec3(13.0f, 2.0f, 3.0f), 
                     Vec3 up = Vec3(0.0f, 1.0f, 0.0f),
                     float yaw = -90.0f, float pitch = 0.0f)
        : position(pos), world_up(up), yaw(yaw), pitch(pitch),
          fov(45.0f), aspect_ratio(16.0f/9.0f), aperture(0.0f), focus_distance(4.0f),
          movement_speed(2.5f), mouse_sensitivity(0.1f), first_mouse(true),
          last_x(400.0f), last_y(300.0f), mouse_captured(false)
    {
        // Initialize all keys as not pressed
        for (int i = 0; i < SDL_NUM_SCANCODES; i++) {
            keys[i] = false;
        }
        update_camera_vectors();
    }
    
    void process_keyboard(float delta_time) {
        float velocity = movement_speed * delta_time;
        
        if (keys[SDL_SCANCODE_W]) {
            position = position + front * velocity;
        }
        if (keys[SDL_SCANCODE_S]) {
            position = position - front * velocity;
        }
        if (keys[SDL_SCANCODE_A]) {
            position = position - right * velocity;
        }
        if (keys[SDL_SCANCODE_D]) {
            position = position + right * velocity;
        }
        
        if (keys[SDL_SCANCODE_Q]) {
            position = position - world_up * velocity;
        }
        if (keys[SDL_SCANCODE_E]) {
            position = position + world_up * velocity;
        }
        
        if (keys[SDL_SCANCODE_LSHIFT]) {
            // Move faster when shift is held
            float fast_velocity = movement_speed * 3.0f * delta_time;
            if (keys[SDL_SCANCODE_W]) position = position + front * (fast_velocity - velocity);
            if (keys[SDL_SCANCODE_S]) position = position - front * (fast_velocity - velocity);
            if (keys[SDL_SCANCODE_A]) position = position - right * (fast_velocity - velocity);
            if (keys[SDL_SCANCODE_D]) position = position + right * (fast_velocity - velocity);
        }
        
        if (keys[SDL_SCANCODE_EQUALS] || keys[SDL_SCANCODE_KP_PLUS]) {
            fov = fmaxf(1.0f, fov - 30.0f * delta_time);  // Zoom in
        }
        if (keys[SDL_SCANCODE_MINUS] || keys[SDL_SCANCODE_KP_MINUS]) {
            fov = fminf(90.0f, fov + 30.0f * delta_time);  // Zoom out
        }

        if (keys[SDL_SCANCODE_UP]) {
            aperture = fminf(1.0f, aperture + 0.01f * delta_time * 10.0f); 
        }
        if (keys[SDL_SCANCODE_DOWN]) {
            aperture = fmaxf(0.0f, aperture - 0.01f * delta_time * 10.0f); 
        }
        
        if (keys[SDL_SCANCODE_RIGHT]) {
            focus_distance = fminf(100.0f, focus_distance + 0.1f * delta_time * 10.0f);
        }
        if (keys[SDL_SCANCODE_LEFT]) {
            focus_distance = fmaxf(0.1f, focus_distance - 0.1f * delta_time * 10.0f);
        }

    }
    
    void handle_event(const SDL_Event& event) {
        switch (event.type) {
            case SDL_KEYDOWN:
                keys[event.key.keysym.scancode] = true;
                break;
                
            case SDL_KEYUP:
                keys[event.key.keysym.scancode] = false;
                break;
                
            case SDL_MOUSEBUTTONDOWN:
                if (event.button.button == SDL_BUTTON_LEFT && !mouse_captured) {
                    mouse_captured = true;
                    SDL_SetRelativeMouseMode(SDL_TRUE);
                    first_mouse = true;
                }
                break;

            case SDL_MOUSEBUTTONUP:
                if (event.button.button == SDL_BUTTON_LEFT && mouse_captured) {
                    mouse_captured = false;
                    SDL_SetRelativeMouseMode(SDL_FALSE);
                }
                break;

            case SDL_MOUSEMOTION:
                if (mouse_captured) {
                    float x_offset = event.motion.xrel * mouse_sensitivity;
                    float y_offset = -event.motion.yrel * mouse_sensitivity; 
                    
                    yaw += x_offset;
                    pitch += y_offset;
                    
                    if (pitch > 89.0f) pitch = 89.0f;
                    if (pitch < -89.0f) pitch = -89.0f;
                    
                    update_camera_vectors();
                }
        }
    }

    // Calculate front vector from Euler angles
    void update_camera_vectors() {
        // Calculate the new front vector
        Vec3 new_front;
        new_front.x = cos(yaw * PI / 180.0f) * cos(pitch * PI / 180.0f);
        new_front.y = sin(pitch * PI / 180.0f);
        new_front.z = sin(yaw * PI / 180.0f) * cos(pitch * PI / 180.0f);
        front = new_front.unit_vector();
        
        // Also re-calculate the right and up vector
        right = front.cross(world_up).unit_vector();
        up = right.cross(front).unit_vector();
    }
    
    Camera to_cuda_camera() const {
        Camera cam;
        cam.lookfrom = position;
        cam.lookat = position + front;  // Look in the direction of front vector
        cam.vup = up;
        cam.vfov = fov;
        cam.aspect_ratio = aspect_ratio;
        cam.aperture = aperture;
        cam.focus_dist = focus_distance;
        cam.lens_radius = aperture / 2.0f;
        
        float theta = fov * PI / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect_ratio * half_height;
        
        cam.w = (position - (position + front)).unit_vector();
        cam.u = up.cross(cam.w).unit_vector();
        cam.v = cam.w.cross(cam.u);
        
        cam.origin = position;
        cam.lower_left_corner = cam.origin - half_width * focus_distance * cam.u 
                               - half_height * focus_distance * cam.v - focus_distance * cam.w;
        cam.horizontal = 2.0f * half_width * focus_distance * cam.u;
        cam.vertical = 2.0f * half_height * focus_distance * cam.v;
        
        return cam;
    }
    
    std::string get_info_string() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), 
            "Pos: (%.1f, %.1f, %.1f) | Yaw: %.1f | Pitch: %.1f | FOV: %.1f | Aperture: %.4f | Focus Dist: %.1f",
            position.x, position.y, position.z, yaw, pitch, fov, aperture, focus_distance);
        return std::string(buffer);
    }
};