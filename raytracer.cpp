// This file is distributed under the MIT license.
// See the LICENSE file for details.

/*
  SNOWMAN is currently under active development.
  Features, functionality, and output may change frequently.

  It is created for teaching purposes as part of an HPC (High Performance Computing) course.

  If you encounter any issues feel free to reach out:

  Contact: kmanda@uni-bonn.de.com
*/

#include "raytracer.hpp"
#include <fstream>
#include <limits>
#include <iostream>
#include <random>
#include <cmath>
#include <mpi.h>
#include <algorithm>

RayTracer::RayTracer(int w, int h) : width(w), height(h), scene(nullptr) {}

void RayTracer::set_scene(Scene* s) {
    scene = s;
}

template<bool ray_dir_normalized>
bool RayTracer::intersect_sphere(const Vec3& ray_orig, const Vec3& ray_dir,
                                 const Sphere& sphere, double& t) {
	//Ideas left to optimize this:
	// - Spheres store squared radius
	// - Manual vectorization (either with std::array<Sphere, 4> or directly with intrinsics)

    Vec3 oc = ray_orig - sphere.center;

    double half_b = oc.dot(ray_dir);
    double c = oc.dot(oc) - sphere.radius * sphere.radius;

    if (half_b * half_b < c) return false; //This really is discriminant < 0
    double sqrt_disc = std::sqrt(half_b * half_b - c);
	
	double t_near = -half_b - sqrt_disc;
	double t_far  = -half_b + sqrt_disc;
	if constexpr (!ray_dir_normalized) {
		double inv_a = 1.0 / ray_dir.dot(ray_dir);
		t_near *= inv_a;
		t_far  *= inv_a;
	}

	double t_candidate = (t_near > 1e-4) ? t_near : t_far;
    if (t_candidate > 1e-4) {
        t = t_candidate;
        return true;
    }

    return false;
}

template<bool ray_dir_normalized>
__m256d RayTracer::intersect_sphere_vectorized(const Vec3& ray_orig, const Vec3& ray_dir,
                                               const double* center_x, const double* center_y,
                                               const double* center_z, const double* radius) {
	//Load ray into registers. This will be optimized away to only happen once, hopefully
	const __m256d orig_x = _mm256_set1_pd(ray_orig.x);
	const __m256d orig_y = _mm256_set1_pd(ray_orig.y);
	const __m256d orig_z = _mm256_set1_pd(ray_orig.z);

	const __m256d dir_x = _mm256_set1_pd(ray_dir.x);
	const __m256d dir_y = _mm256_set1_pd(ray_dir.y);
	const __m256d dir_z = _mm256_set1_pd(ray_dir.z);

	const __m256d zero  = _mm256_setzero_pd();
	const __m256d eps   = _mm256_set1_pd(1e-4);
	const __m256d infty = _mm256_set1_pd(std::numeric_limits<double>::infinity());

	const __m256d cx = _mm256_load_pd(center_x);
	const __m256d cy = _mm256_load_pd(center_y);
	const __m256d cz = _mm256_load_pd(center_z);
	const __m256d r  = _mm256_load_pd(radius);

	const __m256d ocx = _mm256_sub_pd(orig_x, cx);
	const __m256d ocy = _mm256_sub_pd(orig_y, cy);
	const __m256d ocz = _mm256_sub_pd(orig_z, cz);


	const __m256d half_b = _mm256_fmadd_pd(ocx, dir_x, _mm256_fmadd_pd(ocy, dir_y, _mm256_mul_pd(ocz, dir_z)));
	const __m256d oc_dot = _mm256_fmadd_pd(ocx, ocx, _mm256_fmadd_pd(ocy, ocy, _mm256_mul_pd(ocz, ocz)));
	const __m256d c = _mm256_fnmadd_pd(r, r, oc_dot);
	//const __m256d c = _mm256_sub_pd(oc_dot, _mm256_mul_pd(r, r));

	const __m256d discriminant = _mm256_fmsub_pd(half_b, half_b, c);
	//const __m256d discriminant = _mm256_sub_pd(_mm256_mul_pd(half_b, half_b), c);
//	__m256d result_mask = _mm256_cmp_pd(discriminant, zero, _CMP_GT_OQ); //Not needed: NaN entries work out with carefully chosen comparisons down below

#if defined __AVX512VL__ && defined __AVX512F__
	__mmask8 any_intersection = _mm256_cmp_pd_mask(discriminant, zero, _CMP_GT_OQ);
	if (any_intersection == 0)
		return infty;
#else
	__m256d any_intersection = _mm256_cmp_pd(discriminant, zero, _CMP_GT_OQ);
	int mask = _mm256_movemask_pd(any_intersection);
	if (mask == 0)
		return infty;
#endif

	__m256d sqrt_disc = _mm256_sqrt_pd(discriminant);  //This will contain -NaN for entries indicated in result_mask
	const __m256d neg_half_b = _mm256_sub_pd(zero, half_b);
	__m256d t_near = _mm256_sub_pd(neg_half_b, sqrt_disc);
	__m256d t_far  = _mm256_add_pd(neg_half_b, sqrt_disc);

	if constexpr (!ray_dir_normalized) {
		const __m256d inv_a = _mm256_set1_pd(1.0 / ray_dir.dot(ray_dir));
		t_near = _mm256_mul_pd(t_near, inv_a);
		t_far  = _mm256_mul_pd(t_far , inv_a);
	}

	//For the output, we want to create a vector that has the closest t > 1e-4 and
	//infinity if this doesn't exist (i.e. if both t_near,t_far<=1e-4 or discriminant <0)

	__m256d t_candidate;
#if defined __AVX512VL__ && defined __AVX512F__
	__mmask8 near_gt_eps = _mm256_cmp_pd_mask(t_near, eps, _CMP_GT_OQ);
	__mmask8  far_gt_eps = _mm256_cmp_pd_mask(t_far , eps, _CMP_GT_OQ);

	t_candidate = _mm256_mask_mov_pd(t_far, near_gt_eps, t_near);
	__mmask8 any_gt_eps = near_gt_eps | near_gt_eps;
	t_candidate = _mm256_mask_mov_pd(infty, any_gt_eps, t_candidate);
#else
#warning Old processor
	t_candidate = _mm256_blendv_pd(t_far, t_near     , _mm256_cmp_pd(t_near     , eps, _CMP_GT_OQ));
	t_candidate = _mm256_blendv_pd(infty, t_candidate, _mm256_cmp_pd(t_candidate, eps, _CMP_GT_OQ));
#endif

	return t_candidate;
}

template<bool ray_dir_normalized>
bool RayTracer::does_intersect_sphere_vectorized(const Vec3& ray_orig, const Vec3& ray_dir,
                                               const double* center_x, const double* center_y,
                                               const double* center_z, const double* radius) {
	//Load ray into registers. This will be optimized away to only happen once, hopefully
	const __m256d orig_x = _mm256_set1_pd(ray_orig.x);
	const __m256d orig_y = _mm256_set1_pd(ray_orig.y);
	const __m256d orig_z = _mm256_set1_pd(ray_orig.z);

	const __m256d dir_x = _mm256_set1_pd(ray_dir.x);
	const __m256d dir_y = _mm256_set1_pd(ray_dir.y);
	const __m256d dir_z = _mm256_set1_pd(ray_dir.z);

	const __m256d zero  = _mm256_setzero_pd();
	const __m256d eps   = _mm256_set1_pd(1e-4);
	const __m256d infty = _mm256_set1_pd(std::numeric_limits<double>::infinity());

	const __m256d cx = _mm256_load_pd(center_x);
	const __m256d cy = _mm256_load_pd(center_y);
	const __m256d cz = _mm256_load_pd(center_z);
	const __m256d r  = _mm256_load_pd(radius);

	const __m256d ocx = _mm256_sub_pd(orig_x, cx);
	const __m256d ocy = _mm256_sub_pd(orig_y, cy);
	const __m256d ocz = _mm256_sub_pd(orig_z, cz);


	const __m256d half_b = _mm256_fmadd_pd(ocx, dir_x, _mm256_fmadd_pd(ocy, dir_y, _mm256_mul_pd(ocz, dir_z)));
	const __m256d oc_dot = _mm256_fmadd_pd(ocx, ocx, _mm256_fmadd_pd(ocy, ocy, _mm256_mul_pd(ocz, ocz)));
	const __m256d c = _mm256_fnmadd_pd(r, r, oc_dot);
	//const __m256d c = _mm256_sub_pd(oc_dot, _mm256_mul_pd(r, r));

	const __m256d discriminant = _mm256_fmsub_pd(half_b, half_b, c);
	//const __m256d discriminant = _mm256_sub_pd(_mm256_mul_pd(half_b, half_b), c);
//	__m256d result_mask = _mm256_cmp_pd(discriminant, zero, _CMP_GT_OQ); //Not needed: NaN entries work out with carefully chosen comparisons down below

#if defined __AVX512VL__ && defined __AVX512F__
	__mmask8 any_intersection = _mm256_cmp_pd_mask(discriminant, zero, _CMP_GT_OQ);
	if (any_intersection == 0)
		return false;
#else
	__m256d any_intersection = _mm256_cmp_pd(discriminant, zero, _CMP_GT_OQ);
	int mask = _mm256_movemask_pd(any_intersection);
	if (mask == 0)
		return false;
#endif

	__m256d sqrt_disc = _mm256_sqrt_pd(discriminant);  //This will contain -NaN for entries indicated in result_mask

	//We want to check (-half_b + sqrt_disc) / ray_dir.dot(ray_dir) >= eps. 
	//To make computation dependency chains shorted, instead we test sqrt_disc >= eps * ray_dir.dot(ray_dir) + half_b
	__m256d changed_eps = eps;
	if constexpr (!ray_dir_normalized) 
		changed_eps = _mm256_mul_pd(changed_eps, _mm256_set1_pd(ray_dir.dot(ray_dir)));
	changed_eps = _mm256_add_pd(changed_eps, half_b);

	//For the output, we want to create a vector that has the closest t > 1e-4 and
	//infinity if this doesn't exist (i.e. if both t_near,t_far<=1e-4 or discriminant <0)

#if defined __AVX512VL__ && defined __AVX512F__
	return _mm256_cmp_pd_mask(sqrt_disc , changed_eps, _CMP_GT_OQ);
#else
#warning Old processor
	return _mm256_movemask_pd(_mm256_cmp_pd(sqrt_disc, changed_eps, _CMP_GT_OQ));
#endif
}


bool RayTracer::intersect_plane(const Vec3& ray_orig, const Vec3& ray_dir, const Plane& plane, double& t) {
    double denom = plane.normal.dot(ray_dir);
    if (std::fabs(denom) > 1e-6) {
        double t_temp = (plane.point - ray_orig).dot(plane.normal) / denom;
        if (t_temp >= 1e-4) {
            t = t_temp;
            return true;
        }
    }
    return false;
}

void RayTracer::render(int rank, int size, std::vector<Color>& out_pixels) {
	//Make sure we have a scene set
    if (!scene) {
        if (rank == 0) std::cerr << "Scene not set!\n";
        return;
    }

	//Set up camera
    Vec3 camera_pos(0, 2, 5); // Camera position
    Vec3 camera_lookat(0, 1, 0); // Point camera is looking at
    Vec3 camera_dir = (camera_lookat - camera_pos).normalize();
    Vec3 up(0, 1, 0); // World up vector
    Vec3 right = camera_dir.cross(up).normalize(); // Camera's right vector
    Vec3 cam_up = right.cross(camera_dir).normalize(); // Camera's actual up vector

    double fov = 60.0;
    double aspect_ratio = double(width) / height;
    double scale = tan((fov * 0.5) * M_PI / 180.0);

    Vec3 sunlight_dir = Vec3(-1, -1, -1).normalize(); // Direction of sunlight
    double ambient = 0.3; // Base ambient light in the scene

    //Find floor plane (normal y ~1 and point.y ~0)
    const Plane* floor_plane = nullptr;
    for (const auto& plane : scene->planes) {
        if (plane.normal.y > 0.99 && std::abs(plane.point.y) < 1e-3) {
            floor_plane = &plane;
            break;
        }
    }

	//Generate snowflakes
	// This is done by only one thread and then broadcasted instead of by all threads simultaneously.
	// The reason is that only this way we get a consistent image: snowflakes are circles which could
	// extend over tile-borders. If the workers of these tiles have a different snowflake there, it 
	// would cause inconsistencies.
	//   Note: This wouldn't happen in original implementation, since all workers use the same seed 
	//   for the rng. But this is the more conceptually correct way that allows runtime seeds. Also,
	//   it is unclear whether this actually changes the runtime at all: first of all, the runtime of 
	//   this is negligent. Second of all, we do introduce one more broadcast, but get rid of any load
	//   imbalances due to some threads being faster at computing the snowflakes array.
    const int snowflake_count = 75000;
	const double snowflake_radius = 0.008;
	const double max_ray_distance = 8.0;

	struct BoundedSnowflake {
		int x_min;
		int x_max;
		Vec3 snowflake;

		BoundedSnowflake(int x_min_, int x_max_, const Vec3& snowflake_) :
			x_min(x_min_), x_max(x_max_), snowflake(snowflake_)
		{}
	};
	std::vector<std::vector<BoundedSnowflake>> snowflakes(height);

	std::mt19937 rng(12345);
	std::normal_distribution<double> dist_xz(0.0, 6.0);
	std::uniform_real_distribution<double> dist_y(-1.0, 25.0);

	for (int i = 0; i < snowflake_count; ++i) {
		double x_rand = dist_xz(rng);
		double y_rand = dist_y(rng);
		double z_rand = dist_xz(rng);

		// Clamping bounds: covering a wide, deep area
		if (x_rand < -25.0) x_rand = -25.0;
		else if (x_rand > 25.0) x_rand = 25.0;

		if (z_rand < -25.0) z_rand = -25.0;
		else if (z_rand > 25.0) z_rand = 25.0;

		Vec3 snowflake = Vec3(x_rand, y_rand, z_rand);

		//Shift camera to origin and rotate so that cam_dir is (0,0,1)
		Vec3 snowflake_shifted = snowflake - camera_pos;
		double x_transf = snowflake_shifted.dot(right);
		double y_transf = snowflake_shifted.dot(cam_up);
		double z_transf = snowflake_shifted.dot(camera_dir);

		//Step 1: Figure out if this is in correct distance range. If not, reject it.
		//By triangle inequality and under the assumption that the ray hits the sphere:
		//proj \in [(snowflake - camera_pos).length() - snowflake_radius , (snowflake - camera_pos).length() + snowflake_radius]. So if this range doesn't intersect [0, max_ray_distance], this snowflake will never be rendered
		if (snowflake_shifted.length() > max_ray_distance + snowflake_radius || snowflake_shifted.length() + snowflake_radius < 0)
			continue;

		//Step 2: Determine bounding box
		if (x_transf * x_transf + z_transf * z_transf - snowflake_radius * snowflake_radius < 0 || y_transf * y_transf + z_transf * z_transf - snowflake_radius * snowflake_radius < 0)
			continue;  //We are somehow in the sphere, so there will be no tangent cone
		
		double x_virt_min = (x_transf * z_transf - snowflake_radius * std::sqrt(x_transf * x_transf + z_transf * z_transf - snowflake_radius * snowflake_radius)) / (z_transf * z_transf - snowflake_radius * snowflake_radius);
		double x_virt_max = (x_transf * z_transf + snowflake_radius * std::sqrt(x_transf * x_transf + z_transf * z_transf - snowflake_radius * snowflake_radius)) / (z_transf * z_transf - snowflake_radius * snowflake_radius);
		int x_bound_min =  x_virt_min * ((double)width / (double)(2. * aspect_ratio * scale)) + ((double)width - 1.) / 2.;
		int x_bound_max = (x_virt_max * ((double)width / (double)(2. * aspect_ratio * scale)) + ((double)width - 1.) / 2.) + 1;

		double y_virt_min = (y_transf * z_transf - snowflake_radius * std::sqrt(y_transf * y_transf + z_transf * z_transf - snowflake_radius * snowflake_radius)) / (z_transf * z_transf - snowflake_radius * snowflake_radius);
		double y_virt_max = (y_transf * z_transf + snowflake_radius * std::sqrt(y_transf * y_transf + z_transf * z_transf - snowflake_radius * snowflake_radius)) / (z_transf * z_transf - snowflake_radius * snowflake_radius);
		int y_bound_min =  (((double)height - 1.) / 2.) - y_virt_max * ((double)height / (double)(2. * scale));
		int y_bound_max = ((((double)height - 1.) / 2.) - y_virt_min * ((double)height / (double)(2. * scale))) + 1.;

		//Step 3: Make sure that bounding box intersects viewing window
		if (x_bound_max < 0 || x_bound_min >= width || y_bound_max < 0 || y_bound_min >= height)
			continue;
			
		//Step 4: Accept the snowflake
		for (int row = std::max(0, y_bound_min); row <= std::min(y_bound_max, height-1); row++)
			snowflakes[row].emplace_back(x_bound_min, x_bound_max, snowflake);
	}
	
	//TODO: Test the following and data oriented: split snowflakes into three vectors
	for (auto& row : snowflakes) {
 		std::sort(row.begin(), row.end(), [](const BoundedSnowflake& a, const BoundedSnowflake& b) {
			return a.x_min < b.x_min;
		});
	}


	//Bring spheres in more advantages data structure for vectorization
	//These could be std::vector with custom allocator, but that seems to be a bit overboard. Also std::vector<__m256> would be possible, but that also doesn't seem nice.
	//The best alternative would be std::aligned_alloc, but that is more of a C-style way so we stick to this
	double* sphere_centers_x = (double*)::operator new[](((scene->spheres.size() + 3)) * sizeof(double), std::align_val_t(32));
	double* sphere_centers_y = (double*)::operator new[](((scene->spheres.size() + 3)) * sizeof(double), std::align_val_t(32));
	double* sphere_centers_z = (double*)::operator new[](((scene->spheres.size() + 3)) * sizeof(double), std::align_val_t(32));
	double* sphere_radii     = (double*)::operator new[](((scene->spheres.size() + 3)) * sizeof(double), std::align_val_t(32));

	for(int i = 0; i != scene->spheres.size(); i++) {
		const auto& sphere = scene->spheres[i];
		sphere_centers_x[i] = sphere.center.x;
		sphere_centers_y[i] = sphere.center.y;
		sphere_centers_z[i] = sphere.center.z;
		sphere_radii    [i] = sphere.radius;
	}
	for(int i = scene->spheres.size(); i != scene->spheres.size()+3; i++){
		//Add dummy spheres behind camera
		sphere_centers_x[i] = camera_pos.x - camera_dir.x;
		sphere_centers_y[i] = camera_pos.y - camera_dir.y;
		sphere_centers_z[i] = camera_pos.z - camera_dir.z;
		sphere_radii    [i] = 0;
	}


	//This lambda will actually do the raytracing to compute the pixel values of a tile. 
	//The output format is a flattened RGB array. We do it this way instead of "Color"-structs, as 
	//this format allows direct sending/receiving using MPI and thus gets rid of the unnecessary 
	//conversion inbetween
	auto compute_tile_flat = [&](int start_row, int row_count, unsigned char* buffer) {
        for (int y = start_row; y < start_row + row_count; ++y) {
            for (int x = 0; x < width; ++x) {
				//Theoretically, we could pull computation of py out of this loop and precompute the px and store them. Perf shows that this uses up basically no runtime anyways, so let's keep it this way for readability sake
                double ndc_x = (x + 0.5) / width;
                double ndc_y = (y + 0.5) / height;
                double px = (2 * ndc_x - 1) * aspect_ratio * scale;
                double py = (1 - 2 * ndc_y) * scale;

                Vec3 ray_dir = (camera_dir + right * px + cam_up * py).normalize();
                Vec3 ray_orig = camera_pos;

                double closest_t = std::numeric_limits<double>::max();
                const Sphere* hit_sphere = nullptr;
                const Plane * hit_plane = nullptr;

                // Find closest sphere hit
				__m256d closest_t_vec = _mm256_set1_pd(std::numeric_limits<double>::infinity());
				__m256i closest_t_idx = _mm256_setzero_si256();  //Initial value doesn't matter: we detect that nothing was found via "closest_t_vec"
				__m256i current_idx   = _mm256_setr_epi64x(0,1,2,3);
				__m256i increment     = _mm256_set1_epi64x(4);
				for(int i = 0; i < scene->spheres.size(); i+=4) {
					__m256d t_canidate = intersect_sphere_vectorized<true>(ray_orig, ray_dir, sphere_centers_x + i, sphere_centers_y + i, sphere_centers_z + i, sphere_radii + i);
				
#if defined __AVX512VL__ && defined __AVX512F__
					__mmask8 better = _mm256_cmp_pd_mask(t_canidate, closest_t_vec, _CMP_LT_OQ);

					closest_t_vec = _mm256_mask_mov_pd   (closest_t_vec, better, t_canidate);
					closest_t_idx = _mm256_mask_mov_epi64(closest_t_idx, better, current_idx);
#else
#warning Old processor
					__m256d better = _mm256_cmp_pd(t_canidate, closest_t_vec, _CMP_LT_OQ);

					closest_t_vec = _mm256_blendv_pd   (closest_t_vec, t_canidate , better);
					closest_t_idx = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(closest_t_idx), _mm256_castsi256_pd(current_idx), better));
#endif

					current_idx = _mm256_add_epi64(current_idx, increment);
				}

				//The following is dependent on the standard library version one uses. It works with gcc and clang. For others, one might explicitely need to store "closest_t_vec" and "closest_t_idx"
				for(int i = 0; i != 4; i++) {
					if (closest_t_vec[i] < closest_t) {
						closest_t = closest_t_vec[i];
						hit_sphere = scene->spheres.data() + closest_t_idx[i];
						//hit_plane = nullptr;
					}
				}

                // Find closest plane hit
                for (const auto& plane : scene->planes) {
                    double t;
                    if (intersect_plane(ray_orig, ray_dir, plane, t) && t < closest_t) {
                        closest_t = t;
                        hit_plane = &plane;
                        hit_sphere = nullptr;
                    }
                }
				
                Color pixel_color;

                if (hit_sphere) {
                    Vec3 hit_point = ray_orig + ray_dir * closest_t;
                    Vec3 normal = (hit_point - hit_sphere->center).normalize();

                    Vec3 shadow_origin = hit_point + normal * 1e-4;
                    bool in_shadow = false;
                    Vec3 shadow_dir = -sunlight_dir;

					for(int i = 0; i < scene->spheres.size(); i+=4) {
						if (does_intersect_sphere_vectorized<true>(shadow_origin, shadow_dir, sphere_centers_x + i, sphere_centers_y + i, sphere_centers_z + i, sphere_radii + i)) {
							in_shadow = true;
							break;
						}
                    }
                    if (!in_shadow && floor_plane) {
                        double t_shadow_floor;
                        if (intersect_plane(shadow_origin, shadow_dir, *floor_plane, t_shadow_floor)) {
                            if (t_shadow_floor > 1e-4) {
                                in_shadow = true;
                            }
                        }
                    }

                    double diffuse = in_shadow ? 0.0 : std::max(0.0, normal.dot(-sunlight_dir));
                    double brightness = ambient + (1.0 - ambient) * diffuse;

                    pixel_color.r = std::min(255, int(hit_sphere->color.r * brightness));
                    pixel_color.g = std::min(255, int(hit_sphere->color.g * brightness));
                    pixel_color.b = std::min(255, int(hit_sphere->color.b * brightness));
                }
                else if (hit_plane) {
                    Vec3 hit_point = ray_orig + ray_dir * closest_t;
                    Vec3 normal = hit_plane->normal;

                    Vec3 shadow_origin = hit_point + normal * 1e-4;
                    bool in_shadow = false;

                    Vec3 shadow_dir = -sunlight_dir;

					
					for(int i = 0; i < scene->spheres.size(); i+=4) {
						if (does_intersect_sphere_vectorized<true>(shadow_origin, shadow_dir, sphere_centers_x + i, sphere_centers_y + i, sphere_centers_z + i, sphere_radii + i)) {
							in_shadow = true;
							break;
						}
                    }

                    if (floor_plane && hit_plane == floor_plane) {
                        // Base color for the floor is pure white
                        pixel_color = Color(255, 255, 255);

                        // Apply shadow effect to the white floor
                        if (in_shadow) {
                            // This makes the shadow appear as a darker shade of white.
                            double shadow_brightness_factor = 0.6; // Adjust for desired shadow intensity
                            pixel_color.r = (unsigned char)(pixel_color.r * shadow_brightness_factor);
                            pixel_color.g = (unsigned char)(pixel_color.g * shadow_brightness_factor);
                            pixel_color.b = (unsigned char)(pixel_color.b * shadow_brightness_factor);
                        }
                    } else {
                        // Other planes with their original lighting
                        double diffuse = in_shadow ? 0.0 : std::max(0.0, normal.dot(-sunlight_dir));
                        double brightness = ambient + (1.0 - ambient) * diffuse;
                        pixel_color.r = std::min(255, int(hit_plane->color.r * brightness));
                        pixel_color.g = std::min(255, int(hit_plane->color.g * brightness));
                        pixel_color.b = std::min(255, int(hit_plane->color.b * brightness));
                    }
                }
                else {
                    // Sky gradient (top blue, bottom white)
                    double t = 0.5 * (ray_dir.y + 1.0);
                    Color top(135, 206, 235);
                    Color bottom(255, 255, 255);
                    pixel_color.r = (1 - t) * bottom.r + t * top.r;
                    pixel_color.g = (1 - t) * bottom.g + t * top.g;
                    pixel_color.b = (1 - t) * bottom.b + t * top.b;
                }

                // SNOWFLAKE OVERLAY
                // Overlay snowflakes as tiny white dots
                for (const auto& [x_min, x_max, flake_pos] : snowflakes[y]) {
					if (x < x_min)
						break;
		  			if (x > x_max)
						continue;

                    Vec3 to_flake = flake_pos - ray_orig;
                    double proj = to_flake.dot(ray_dir);

                    // Check if snowflake is in front of the camera, within max_ray_distance,
                    // AND not behind any other scene object (closest_t)
                    if (proj < 0 || proj > max_ray_distance || proj > closest_t) continue;

                    Vec3 closest_point_on_ray = ray_orig + ray_dir * proj;
                    double dist_sq = (closest_point_on_ray.x - flake_pos.x)*(closest_point_on_ray.x - flake_pos.x) +
                                     (closest_point_on_ray.y - flake_pos.y)*(closest_point_on_ray.y - flake_pos.y) +
                                     (closest_point_on_ray.z - flake_pos.z)*(closest_point_on_ray.z - flake_pos.z);

                    if (dist_sq < snowflake_radius * snowflake_radius) {
                        pixel_color = Color(255, 255, 255); // Pure white snowflake dot
                        break;
                    }
                }

                size_t idx = ((y - start_row) * width + x) * 3;
                buffer[idx + 0] = pixel_color.r;
                buffer[idx + 1] = pixel_color.g;
                buffer[idx + 2] = pixel_color.b;
            }
        }
    };

	const int tiles_per_worker = [&](){ if (size < 32) { return 8; } return 16; }();
    const int tile_height = std::max(1, height / (size * tiles_per_worker));
    const int master_tile_height = [&](){ if (size < 32) { return tile_height / 16; } return tile_height / 4; }(); // Rank 0 uses smaller tiles to offset communication overhead
	enum class Tag : int { WORK = 1, RESULT = 2 };

    if (rank == 0) {
		//Initialize memory
		std::vector<unsigned char> master_buffer(static_cast<size_t>(width) * height * 3);

		//1.: If there is only one MPI thread, this has to do all work and no communication is required
        if (size == 1) {
			compute_tile_flat(0, height, master_buffer.data());
			out_pixels.resize(width * height);
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
					size_t idx = (static_cast<size_t>(row) * width + col) * 3;
					out_pixels[row * width + col] = Color(master_buffer[idx], master_buffer[idx + 1], master_buffer[idx + 2]);
                }
            }
            return;
        }


		//2.: Distribute initial work
        int next_row = 0;
        int active_workers = std::min(size - 1, (height + tile_height - 1) / tile_height);

		//The following lambda just sends the next tile to "worker_rank"
        auto send_work = [&](int worker_rank) {
            if (next_row >= height) {  //All work was distributed already. Send termination signal
                int term[2] = {-1, 0}; //This is termination header
                MPI_Send(term, 2, MPI_INT, worker_rank, static_cast<int>(Tag::WORK), MPI_COMM_WORLD);
                --active_workers;      //When worker receives "term", it will return and thus become inactive
                return;
            }

			//If we are here, there is still work left to distribute
            int rows = std::min(tile_height, height - next_row); //Last tile could be smaller
            int msg[2] = {next_row, rows}; //We send start row and number of rows
            MPI_Send(msg, 2, MPI_INT, worker_rank, static_cast<int>(Tag::WORK), MPI_COMM_WORLD);
            next_row += rows;
        };

        //Send initial tiles
        for (int r = 1; r <= active_workers; ++r)
            send_work(r);

		//3.: Main loop: poll for finished work from any worker before doing our own work
		while (active_workers > 0) {
            MPI_Status status{};
            int flag = 0;

            // Drain all available results to keep workers busy before doing local work.
            while (true) {
                MPI_Iprobe(MPI_ANY_SOURCE, static_cast<int>(Tag::RESULT), MPI_COMM_WORLD, &flag, &status);
                if (!flag) break;

                int header[2];
                MPI_Recv(header, 2, MPI_INT, status.MPI_SOURCE, static_cast<int>(Tag::RESULT), MPI_COMM_WORLD, &status);

                const int start = header[0];
                const int rows = header[1];
				MPI_Recv(master_buffer.data() + static_cast<size_t>(start) * width * 3, static_cast<size_t>(rows) * width * 3, MPI_UNSIGNED_CHAR, status.MPI_SOURCE, static_cast<int>(Tag::RESULT), MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Distribute new tile to worker (if still possible)
                send_work(status.MPI_SOURCE);
            }

			//Do some work yourself using slightly smaller tiles to compensate for communication overhead
            if (master_tile_height > 0 && next_row < height) {
                int rows_to_compute = std::min(master_tile_height, height - next_row);
                compute_tile_flat(next_row, rows_to_compute, master_buffer.data() + static_cast<size_t>(next_row) * width * 3);
                next_row += rows_to_compute;
                continue;
            }

            // If there is no local work left, block for the next result to avoid spinning.
            int header[2];
            MPI_Recv(header, 2, MPI_INT, MPI_ANY_SOURCE, static_cast<int>(Tag::RESULT), MPI_COMM_WORLD, &status);
            const int start = header[0];
            const int rows = header[1];
            MPI_Recv(master_buffer.data() + static_cast<size_t>(start) * width * 3, rows * width * 3, MPI_UNSIGNED_CHAR, status.MPI_SOURCE, static_cast<int>(Tag::RESULT), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            send_work(status.MPI_SOURCE);
        }
       
		out_pixels.resize(static_cast<size_t>(width) * height);
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                size_t idx = (static_cast<size_t>(row) * width + col) * 3;
                out_pixels[static_cast<size_t>(row) * width + col] = Color(master_buffer[idx], master_buffer[idx + 1], master_buffer[idx + 2]);
            }
        }
    } else {
		//Worker threads
		std::vector<unsigned char> local_buf;
        while (true) {
            int msg[2];
            MPI_Recv(msg, 2, MPI_INT, 0, static_cast<int>(Tag::WORK), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int start = msg[0];
            int rows = msg[1];
            if (start < 0 || rows <= 0) break;

            local_buf.resize(static_cast<size_t>(rows) * width * 3);
            compute_tile_flat(start, rows, local_buf.data());

            int header[2] = {start, rows};
            MPI_Send(header, 2, MPI_INT, 0, static_cast<int>(Tag::RESULT), MPI_COMM_WORLD);
            MPI_Send(local_buf.data(), local_buf.size(), MPI_UNSIGNED_CHAR, 0, static_cast<int>(Tag::RESULT), MPI_COMM_WORLD);
        }
        out_pixels.clear(); //TODO: This seems unnecessary
    }
}

void RayTracer::save_image(const std::string& filename, const std::vector<Color>& pixels) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (auto& c : pixels) {
        ofs << (unsigned char)c.r << (unsigned char)c.g << (unsigned char)c.b;
    }
    ofs.close();
}
