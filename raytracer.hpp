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
    // Render a rectangular tile given its top-left corner (x0,y0) and size (w,h).
    // `out` will be resized to w*h and filled row-major.
    // `seed` is used to initialize any RNG for deterministic overlays per tile.
    void renderTile(int x0, int y0, int w, int h, unsigned int seed, std::vector<Color>& out);
    void save_image(const std::string& filename, const std::vector<Color>& pixels);

private:
    int width, height;
    Scene* scene;

    bool intersect_sphere(const Vec3& ray_orig, const Vec3& ray_dir,
                          const Sphere& sphere, double& t);
    bool intersect_plane(const Vec3& ray_orig, const Vec3& ray_dir,
                         const Plane& plane, double& t);
};

#endif

