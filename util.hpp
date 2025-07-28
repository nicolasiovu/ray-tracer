#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <memory>

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double degress_to_radians(double degrees) {
	return degrees * pi / 180.0;
}

inline double random_double() {
	return std::rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
	return min + (max - min) * random_double();
}

#include "vec3.hpp"
#include "interval.hpp"
#include "color.hpp"
#include "ray.hpp"

#endif
