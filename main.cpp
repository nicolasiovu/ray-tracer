#include <iostream>

#include "color.hpp"
#include "vec3.hpp"
#include "ray.hpp"

Color ray_color(const Ray& r) {
	Vec3 unit_direction = unit_vector(r.direction());
	double a = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
}

int main() {
	// Image
	
	double aspect_ratio = 16.0 / 9.0;

	int image_width = 400;

	int image_height = int(image_width / aspect_ratio);
	image_height = (image_height < 1) ? 1: image_height;

	// Camera
	
	double focal_length = 1.0;
	double viewport_height = 2.0;
	double viewport_width = viewport_height * (double(image_width) / image_height);
	Point3 camera_center = Point3(0, 0, 0);

	// Vectors across the horizontal and own the vertical viewport edges
	Vec3 viewport_u = Vec3(viewport_width, 0, 0);
	Vec3 viewport_v = Vec3(0, -viewport_height, 0);
	
	// these vectors scaled to our actual screen
	Vec3 pixel_delta_u = viewport_u / image_width;
	Vec3 pixel_delta_v = viewport_v / image_height;

	// Location of upper left pixel
	Point3 viewport_upper_left = camera_center - Vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
	Point3 upper_left_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v); // middle of the pixel
 
	// Render
	
	std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

	for (int j = 0; j < image_height; j++) {
		std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
		for (int i = 0; i < image_width; i++) {
			Point3 pixel_center = upper_left_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
			Vec3 ray_direction = pixel_center - camera_center;
			Ray r = Ray(camera_center, ray_direction);

			Color pixel_color = ray_color(r);
			write_color(std::cout, pixel_color);	
		}
	}

	std::clog << "\rDone.             \n";
}
