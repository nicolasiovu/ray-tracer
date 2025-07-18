#ifndef COLOR_H
#define COLOR_H

#include "vec3.hpp"

#include <iostream>

using Color = Vec3;

void write_color(std::ostream& out, const Color& pixel_color) {
	double r = pixel_color.x();
	double g = pixel_color.y();
	double b = pixel_color.z();

	int rbyte = int(255.999 * r);
	int gbyte = int(255.999 * g);
	int bbyte = int(255.999 * b);

	out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif
