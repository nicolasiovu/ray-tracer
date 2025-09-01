#pragma once
#include <cstdint>

class InteractiveCamera;

// Function declarations
void render_with_camera(uint32_t *pixels, int width, int height, const InteractiveCamera& camera);
void cleanup_scene();