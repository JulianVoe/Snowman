// This file is distributed under the MIT license.
// See the LICENSE file for details.


#ifndef RAYTRACER_HPP
#define RAYTRACER_HPP

#include <vector>
#include <string>
#include "utils.hpp"
#include "scene.hpp"

class RayTracer {
public:
    RayTracer(int width, int height);
    void set_scene(Scene* scene);
    void render(int rank, int size, std::vector<Color>& out_pixels);
    void save_image(const std::string& filename, const std::vector<Color>& pixels);

private:
    int width, height;
    Scene* scene;

	template<bool ray_dir_normalized>
    bool intersect_sphere(const Vec3& ray_orig, const Vec3& ray_dir,
                          const Sphere& sphere, double& t);
    bool intersect_plane(const Vec3& ray_orig, const Vec3& ray_dir,
                         const Plane& plane, double& t);
};

#endif

