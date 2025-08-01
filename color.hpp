#ifndef COLOR_H
#define COLOR_H

#include "interval.hpp"

using Color = Vec3;

inline double linear_to_gamma(double linear_component) {
	if (linear_component > 0) {
		return std::sqrt(linear_component);
	}
	return 0;
}

void write_color(std::ostream& out, const Color& pixel_color) {
	double r = pixel_color.x();
	double g = pixel_color.y();
	double b = pixel_color.z();

	r = linear_to_gamma(r);
	g = linear_to_gamma(g);
	b = linear_to_gamma(b);
	
	static const Interval intensity(0.000, 0.999);
	int rbyte = int(256 * intensity.clamp(r));
	int gbyte = int(256 * intensity.clamp(g));
	int bbyte = int(256 * intensity.clamp(b));

	out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif
