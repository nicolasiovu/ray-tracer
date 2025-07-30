#include "util.hpp"

#include "camera.hpp"
#include "hittable.hpp"
#include "hittable_list.hpp"
#include "material.hpp"
#include "sphere.hpp"

int main() {
	HittableList world;
	
	/*
	auto material_ground = std::make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
	auto material_center = std::make_shared<Lambertian>(Color(0.1, 0.2, 0.5));
	auto material_left = std::make_shared<Dielectric>(1.50);
	auto material_bubble = std::make_shared<Dielectric>(1.00 / 1.50);
	auto material_right = std::make_shared<Metal>(Color(0.8, 0.6, 0.2), 1.0);

	world.add(std::make_shared<Sphere>(Point3(0.0, -100.5, -1.0), 100.0, material_ground));
	world.add(std::make_shared<Sphere>(Point3(0.0, 0.0, -1.2), 0.5, material_center));
	world.add(std::make_shared<Sphere>(Point3(-1.0, 0.0, -1.0), 0.5, material_left));
	world.add(std::make_shared<Sphere>(Point3(-1.0, 0.0, -1.0), 0.4, material_bubble));
	world.add(std::make_shared<Sphere>(Point3(1.0, 0.0, -1.0), 0.5, material_right));
	*/

	auto ground_material = std::make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    world.add(std::make_shared<Sphere>(Point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            Point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
		    std::shared_ptr<Material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = Color::random() * Color::random();
                    sphere_material = std::make_shared<Lambertian>(albedo);
                    world.add(std::make_shared<Sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = Color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = std::make_shared<Metal>(albedo, fuzz);
                    world.add(std::make_shared<Sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = std::make_shared<Dielectric>(1.5);
                    world.add(std::make_shared<Sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = std::make_shared<Dielectric>(1.5);
    world.add(std::make_shared<Sphere>(Point3(0, 1, 0), 1.0, material1));

    auto material2 = std::make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
    world.add(std::make_shared<Sphere>(Point3(-4, 1, 0), 1.0, material2));

    auto material3 = std::make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
    world.add(std::make_shared<Sphere>(Point3(4, 1, 0), 1.0, material3));

	Camera cam;

	cam.aspect_ratio = 16.0 / 9.0;
	cam.image_width = 1200;
	cam.samples_per_pixel = 100;
	cam.max_depth = 50;

	cam.vfov = 20;
	cam.lookfrom = Point3(13, 2, 3);
	cam.lookat = Point3(0, 0, 0);
	cam.vup = Vec3(0, 1, 0);

	cam.defocus_angle = 0.6;
	cam.focus_distance = 10.0;
	
	cam.render(world);	
}
