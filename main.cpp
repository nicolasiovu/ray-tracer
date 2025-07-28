#include "util.hpp"

#include "camera.hpp"
#include "hittable.hpp"
#include "hittable_list.hpp"
#include "sphere.hpp"

int main() {
	HittableList world;
	
	world.add(std::make_shared<Sphere>(Point3(0, 0, -1), 0.5));
	world.add(std::make_shared<Sphere>(Point3(0, -100.5, -1), 100));
	
	Camera cam;

	cam.aspect_ratio = 16.0 / 9.0;
	cam.image_width = 400;
	cam.samples_per_pixel = 100;
	cam.max_depth = 50;
	
	cam.render(world);	
}
