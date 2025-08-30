#ifndef INTERVAL_H
#define INTERVAL_H

class Interval {
	public:
		double min, max;

		Interval() : min(+infinity), max(-infinity) {} // empty

		Interval(double min, double max) : min(min), max(max) {}

		Interval(const Interval& a, const Interval&b) {
			min = a.min <= b.min ? a.min: b.min;
			max = a.max >= b.max ? a.max: b.max;
		}

		double size() const {
			return max - min;
		}

		bool contains(double x) const {
			return min <= x && x <= max;
		}

		bool surrounds(double x) const {
			return min < x && x < max;
		}

		Interval expand(double delta) const {
			double padding = delta / 2;
			return Interval(min - padding, max + padding);	
		}

		double clamp(double x) const {
			if (x < min) {
				return min;
			}

			if (x > max) {
				return max;
			}

			return x;
		}

		static const Interval empty, universe;
};

const Interval Interval::empty = Interval(+infinity, -infinity);
const Interval Interval::universe = Interval(-infinity, +infinity);

#endif
