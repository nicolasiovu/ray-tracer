#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.hpp"
#include "material.hpp"

class Camera {
	public:
		double aspect_ratio = 1.0;
		int image_width = 100;
		int samples_per_pixel = 10;
		int max_depth = 10;

		double vfov = 90;
		Point3 lookfrom = Point3(0, 0, 0);
		Point3 lookat = Point3(0, 0, -1);
		Vec3 vup = Vec3(0, 1, 0);
		
		double defocus_angle = 0;
		double focus_distance = 10;

		void render(const Hittable& world) {
			initialize();

			std::cout << "P3\n" << image_width << ' ' << image_height << "\n225\n";

			for (int j=0; j<image_height; j++) {
				std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
				for (int i=0; i<image_width; i++) {
					Color pixel_color(0, 0, 0);
					for (int sample=0; sample < samples_per_pixel; sample++) {
						Ray r = get_ray(i, j);
						pixel_color += ray_color(r, world, max_depth);
					}
					write_color(std::cout, pixel_samples_scale * pixel_color);
				}
			}

			std::clog << "\r Done.             \n";
		}

	private:
		int image_height;
		double pixel_samples_scale;
		Point3 center;
		Point3 pixel00_loc;
		Vec3 pixel_delta_u;
		Vec3 pixel_delta_v;
		Vec3 u, v, w;
		Vec3 defocus_disk_u;
		Vec3 defocus_disk_v;

		void initialize() {
			image_height = int(image_width / aspect_ratio);
			image_height = (image_height < 1) ? 1: image_height;
			
			pixel_samples_scale = 1.0 / samples_per_pixel;

			center = lookfrom;

			auto theta = degrees_to_radians(vfov);
			auto h = std::tan(theta / 2);
			double viewport_height = 2 * h * focus_distance;
			double viewport_width = viewport_height * (double(image_width) / image_height);
			
			w = unit_vector(lookfrom - lookat);
			u = unit_vector(cross(vup, w));
			v = cross(w, u);

			Vec3 viewport_u = viewport_width * u;
			Vec3 viewport_v = viewport_height * -v;

			pixel_delta_u = viewport_u / image_width;
			pixel_delta_v = viewport_v / image_height;

			Point3 viewport_upper_left = center - (focus_distance * w) - viewport_u / 2 - viewport_v / 2;
			pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

			auto defocus_radius = focus_distance * std::tan(degrees_to_radians(defocus_angle / 2));
			defocus_disk_u = u * defocus_radius;
			defocus_disk_v = v * defocus_radius;
		}

		Ray get_ray(int i, int j) const {
			Vec3 offset = sample_square();
			Point3 pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);

			Point3 ray_origin = defocus_angle <= 0 ? center: defocus_disk_sample();
			Vec3 ray_direction = pixel_sample - ray_origin;

			return Ray(ray_origin, ray_direction);
		}

		Vec3 sample_square() const {
			return Vec3(random_double() - 0.5, random_double() - 0.5, 0);
		}

		Point3 defocus_disk_sample() const {
			auto p = random_in_unit_disk();
			return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
		}

		Color ray_color(const Ray& r, const Hittable& world, int depth) const {
			if (depth <= 0) {
				return Color(0, 0, 0);
			}

			HitRecord rec;

			if (world.hit(r, Interval(0.001, infinity), rec)) {
				Ray scattered;
				Color attenuation;
				if (rec.mat->scatter(r, rec, attenuation, scattered)) {
					return attenuation * ray_color(scattered, world, depth - 1);
				}
				return Color(0, 0, 0);
			}

			Vec3 unit_direction = unit_vector(r.direction());
			double a = 0.5 * (unit_direction.y() + 1.0);
			return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
		}
};

#endif
