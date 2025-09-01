#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.hpp"

class Sphere : public Hittable {
	public:
		Sphere(const Point3& static_center, double radius, std::shared_ptr<Material> mat) 
			: center(static_center, Vec3(0, 0, 0)), radius(std::fmax(0, radius)), mat(mat) 
		{
			auto rvec = Vec3(radius, radius, radius);
			bbox = BoundingBox(static_center - rvec, static_center + rvec);	
		}

		Sphere(const Point3& center1, const Point3& center2, double radius, std::shared_ptr<Material> mat) 
			: center(center1, center2 - center1), radius(std::fmax(0, radius)), mat(mat) 
		{
			auto rvec = Vec3(radius, radius, radius);
			BoundingBox box1(center.at(0) - rvec, center.at(0) + rvec);
			BoundingBox box2(center.at(1) - rvec, center.at(1) + rvec);
			bbox = BoundingBox(box1, box2);
		}

		bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
			Point3 current_center = center.at(r.time());
			Vec3 oc = current_center - r.origin();
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
			Vec3 outward_normal = (rec.p - current_center) / radius;
			rec.set_face_normal(r, outward_normal);
			rec.mat = mat;

			return true;
		}

		BoundingBox bounding_box() const override {
			return bbox;
		}

	private:
		Ray center;
		double radius;
		std::shared_ptr<Material> mat;
		BoundingBox bbox;
};

#endif
