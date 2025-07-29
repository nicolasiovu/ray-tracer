#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.hpp"

class Sphere : public Hittable {
	public:
		Sphere(const Point3& center, double radius, std::shared_ptr<Material> mat) 
			: center(center), radius(std::fmax(0, radius)), mat(mat) {}

		bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
			Vec3 oc = center - r.origin();
			double a = r.direction().length_squared();
			double h = dot(r.direction(), oc);
			double c = oc.length_squared() - radius * radius;

			double discriminant = h * h - a * c;
			if (discriminant < 0) {
				return false;
			}

			double sqrtd = std::sqrt(discriminant);

			double root = (h - sqrtd) / a;
			if (!ray_t.surrounds(root)) {
				root = (h + sqrtd) / a;
				if (!ray_t.surrounds(root)) {
					return false;
				}
			}

			rec.t = root;
			rec.p = r.at(rec.t);
			Vec3 outward_normal = (rec.p - center) / radius;
			rec.set_face_normal(r, outward_normal);
			rec.mat = mat;

			return true;
		}

	private:
		Point3 center;
		double radius;
		std::shared_ptr<Material> mat;
};

#endif
