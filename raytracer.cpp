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

RayTracer::RayTracer(int w, int h) : width(w), height(h), scene(nullptr) {}

void RayTracer::set_scene(Scene* s) {
    scene = s;
}

bool RayTracer::intersect_sphere(const Vec3& ray_orig, const Vec3& ray_dir,
                                 const Sphere& sphere, double& t) {
    Vec3 oc = ray_orig - sphere.center;
    double a = ray_dir.dot(ray_dir);
    double b = 2.0 * oc.dot(ray_dir);
    double c = oc.dot(oc) - sphere.radius * sphere.radius;
    double discriminant = b*b - 4*a*c;

    if (discriminant < 0) return false;

    double sqrt_disc = std::sqrt(discriminant);
    double t0 = (-b - sqrt_disc) / (2*a);
    double t1 = (-b + sqrt_disc) / (2*a);

    if (t0 > 1e-4) {
        t = t0;
        return true;
    } else if (t1 > 1e-4) {
        t = t1;
        return true;
    }
    return false;
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
    std::vector<Vec3> snowflakes(snowflake_count);
    if (rank == 0) {
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

            snowflakes[i] = Vec3(x_rand, y_rand, z_rand);
        }
    }
    MPI_Bcast(snowflakes.data(), snowflake_count * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//This lambda will actually do the raytracing to compute the pixel values of a tile. 
	//The output format is a flattened RGB array. We do it this way instead of "Color"-structs, as 
	//this format allows direct sending/receiving using MPI and thus gets rid of the unnecessary 
	//conversion inbetween
    auto compute_tile_flat = [&](int start_row, int row_count, std::vector<unsigned char>& buffer) {
        buffer.resize(row_count * width * 3);
        for (int y = start_row; y < start_row + row_count; ++y) {
            for (int x = 0; x < width; ++x) {
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
                for (const auto& sphere : scene->spheres) {
                    double t;
                    if (intersect_sphere(ray_orig, ray_dir, sphere, t) && t < closest_t) {
                        closest_t = t;
                        hit_sphere = &sphere;
                        hit_plane = nullptr;
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
                    for (const auto& s : scene->spheres) {
                        double t_shadow;
                        if (&s != hit_sphere && intersect_sphere(shadow_origin, shadow_dir, s, t_shadow)) {
                            if (t_shadow > 1e-4) {
                                in_shadow = true;
                                break;
                            }
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
                    for (const auto& s : scene->spheres) {
                        double t_shadow;
                        if (intersect_sphere(shadow_origin, shadow_dir, s, t_shadow)) {
                            if (t_shadow > 1e-4) {
                                in_shadow = true;
                                break;
                            }
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
                const double snowflake_radius = 0.008;
                const double max_ray_distance = 8.0;

                for (const auto& flake_pos : snowflakes) {
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

        constexpr int tiles_per_worker = 4;
    const int tile_height = std::max(1, height / (size * tiles_per_worker));
    const int master_tile_height = std::max(1, tile_height / 2); // Rank 0 uses smaller tiles to offset communication overhead
        enum class Tag : int { WORK = 1, RESULT = 2 };

    if (rank == 0) {
		//1.: Initialize memory
		
		//Initialize output vector
        out_pixels.assign(width * height, Color()); //TODO: It is standard practice to initialize memory, but on other hand this is wasted time
		
		//If there is only one MPI thread, this has to do all work and no communication is required
        if (size == 1) {
            std::vector<unsigned char> full_buf;
            compute_tile_flat(0, height, full_buf);
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    size_t idx = (row * width + col) * 3;
                    out_pixels[row * width + col] = Color(full_buf[idx], full_buf[idx + 1], full_buf[idx + 2]);
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
        std::vector<unsigned char> recv_buf; //This will receive pixels of workers
        std::vector<unsigned char> local_buf;
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
                recv_buf.resize(rows * width * 3);
                MPI_Recv(recv_buf.data(), recv_buf.size(), MPI_UNSIGNED_CHAR, status.MPI_SOURCE, static_cast<int>(Tag::RESULT), MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Immediately dispatch more work so the worker does not idle while we copy data.
                send_work(status.MPI_SOURCE);

                for (int row = 0; row < rows; ++row) {
                    const size_t base = static_cast<size_t>(row * width) * 3;
                    for (int col = 0; col < width; ++col) {
                        size_t idx = base + static_cast<size_t>(col) * 3;
                        out_pixels[(start + row) * width + col] = Color(recv_buf[idx], recv_buf[idx + 1], recv_buf[idx + 2]);
                    }
                }
            }

                        //Do some work yourself using slightly smaller tiles to compensate for communication overhead
            if (next_row < height) {
                int rows_to_compute = std::min(master_tile_height, height - next_row);
                compute_tile_flat(next_row, rows_to_compute, local_buf);
                for (int row = 0; row < rows_to_compute; ++row) {
                    for (int col = 0; col < width; ++col) {
                        size_t idx = (static_cast<size_t>(row) * width + col) * 3;
                        out_pixels[(next_row + row) * width + col] = Color(local_buf[idx], local_buf[idx + 1], local_buf[idx + 2]);
                    }
                }
                next_row += rows_to_compute;
                continue;
            }

            // If there is no local work left, block for the next result to avoid spinning.
            int header[2];
            MPI_Recv(header, 2, MPI_INT, MPI_ANY_SOURCE, static_cast<int>(Tag::RESULT), MPI_COMM_WORLD, &status);
            const int start = header[0];
            const int rows = header[1];
            recv_buf.resize(rows * width * 3);
            MPI_Recv(recv_buf.data(), recv_buf.size(), MPI_UNSIGNED_CHAR, status.MPI_SOURCE, static_cast<int>(Tag::RESULT), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            send_work(status.MPI_SOURCE);
            for (int row = 0; row < rows; ++row) {
                const size_t base = static_cast<size_t>(row * width) * 3;
                for (int col = 0; col < width; ++col) {
                    size_t idx = base + static_cast<size_t>(col) * 3;
                    out_pixels[(start + row) * width + col] = Color(recv_buf[idx], recv_buf[idx + 1], recv_buf[idx + 2]);
                }
            }
        }
    } else {
		//Worker threads
        while (true) {
            int msg[2];
            MPI_Recv(msg, 2, MPI_INT, 0, static_cast<int>(Tag::WORK), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int start = msg[0];
            int rows = msg[1];
            if (start < 0 || rows <= 0) break;

            std::vector<unsigned char> send_buf;
            compute_tile_flat(start, rows, send_buf);

            int header[2] = {start, rows};
            MPI_Send(header, 2, MPI_INT, 0, static_cast<int>(Tag::RESULT), MPI_COMM_WORLD);
            MPI_Send(send_buf.data(), send_buf.size(), MPI_UNSIGNED_CHAR, 0, static_cast<int>(Tag::RESULT), MPI_COMM_WORLD);
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
