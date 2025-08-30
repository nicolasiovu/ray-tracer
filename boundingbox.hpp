#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

class BoundingBox {
	public:
		Interval x, y, z;

		BoundingBox() {}

		BoundingBox(const Interval& x, const Interval& y, const Interval& z) : x(x), y(y), z(z) {}

		BoundingBox(const Point3& a, const Point3& b) {
			if (a[0] <= b[0]) {
				x = Interval(a[0], b[0]);
			} else { 
				x = Interval(b[0], a[0]);
			}
			if (a[1] <= b[1]) {
				y = Interval(a[1], b[1]);
			} else {
				y = Interval(b[1], a[1]);
			}
			if (a[2] <= b[2]) {
				z = Interval(a[2], b[2]);
			} else {
				z = Interval(b[2], a[2]);
			}
		}

		BoundingBox(const BoundingBox& box1, const BoundingBox& box2) {
			x = Interval(box1.x, box2.x);
			y = Interval(box1.y, box2.y);
			z = Interval(box1.z, box2.z);
		}

		const Interval& axis_interval(int n) const {
			switch (n) {
				case 0:
					return x;
					
				case 1:
					return y;

				case 2:
					return z;
			}
			return x;
		}

		bool hit(const Ray& r, Interval ray_t) const {
			const Point3& ray_orig = r.origin();
			const Vec3& ray_dir = r.direction();

			for(int axis = 0; axis < 3; axis++) {
				const Interval& ax = axis_interval(axis);
				const double adinv = 1.0 / ray_dir[axis];

				auto t0 = (ax.min - ray_orig[axis]) * adinv;
				auto t1 = (ax.max - ray_orig[axis]) * adinv;
				
				// essentially, clamp the ray_t interval so if it doesnt
				// intersect, it will exit early more often!
				if (t0 < t1) {
					if (t0 > ray_t.min) {
						ray_t.min = t0;
					}
					if (t1 < ray_t.max) {
						ray_t.max = t1;
					}
				} else {
					if (t1 > ray_t.min) {
						ray_t.min = t1;
					}
					if (t0 < ray_t.max) {
						ray_t.max = t0;
					}
				}

				if (ray_t.max <= ray_t.min) {
					return false;
				}
			}
			return true;
		}

		int longest_axis() const {
			if (x.size() > y.size()) {
				return x.size() > z.size() ? 0: 2;
			} else {
				return y.size() > z.size() ? 1: 2;
			}
		}

		static const BoundingBox empty, universe;
};

const BoundingBox BoundingBox::empty = BoundingBox(Interval::empty, Interval::empty, Interval::empty);

const BoundingBox BoundingBox::universe = BoundingBox(Interval::universe, Interval::universe, Interval::universe);


#endif
