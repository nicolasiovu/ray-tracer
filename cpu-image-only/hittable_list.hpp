#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.hpp"
#include "boundingbox.hpp"
#include <vector>

class HittableList : public Hittable {
	public:
		std::vector<std::shared_ptr<Hittable>> objects;

		HittableList() {}
		HittableList(std::shared_ptr<Hittable> object) { add(object); }

		void clear() { objects.clear(); }

		void add(std::shared_ptr<Hittable> object) {
			objects.push_back(object);
			bbox = BoundingBox(bbox, object->bounding_box());
		}

		bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
			HitRecord temp_rec;
			bool hit_anything = false;
			double closest = ray_t.max;

			for (const auto& object : objects) {
				if (object->hit(r, Interval(ray_t.min, closest), temp_rec)) {
					hit_anything = true;
					closest = temp_rec.t;
					rec = temp_rec;
				}
			}

			return hit_anything;
		}

		BoundingBox bounding_box() const override {
			return bbox;
		}

	private:
		BoundingBox bbox;
};

#endif
