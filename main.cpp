// This file is distributed under the MIT license.
// See the LICENSE file for details.

/*
  SNOWMAN is currently under active development.
  Features, functionality, and output may change frequently.

  It is created for teaching purposes as part of an HPC (High Performance Computing) course.

  If you encounter any issues feel free to reach out:

  Contact: kmanda@uni-bonn.de.com
*/

#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "raytracer.hpp"
#include "scene.hpp"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            std::cout << "Usage: " << argv[0] << " <image_size> <num_snowmen>\n";
        }
        MPI_Finalize();
        return 1;
    }

    int image_size = std::stoi(argv[1]);
    int num_snowmen = std::stoi(argv[2]);

    // Scene generation and RayTracer setup
    Scene scene;
    scene.generate_snowmen(num_snowmen);

    RayTracer raytracer(image_size, image_size);
    raytracer.set_scene(&scene);

    std::vector<Color> pixels;

    // Local Computation Time using std::chrono
    auto compute_start_time = std::chrono::high_resolution_clock::now();
    raytracer.render(rank, size, pixels);
    auto compute_end_time = std::chrono::high_resolution_clock::now();

    // Calculate duration in seconds
    std::chrono::duration<double> local_compute_duration = compute_end_time - compute_start_time;
    double local_compute_time = local_compute_duration.count();

    // Image saving
    if (rank == 0) {
        raytracer.save_image("output.ppm", pixels);
        std::cout << "Image saved to output.ppm\n";
    }

    // Reporting Performance Metrics for Computation
    double max_local_compute_time;
    MPI_Reduce(&local_compute_time, &max_local_compute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double min_local_compute_time;
    MPI_Reduce(&local_compute_time, &min_local_compute_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    double sum_local_compute_time;
    MPI_Reduce(&local_compute_time, &sum_local_compute_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_local_compute_time = sum_local_compute_time / size;

        std::cout << "\n--- Computational Performance Metrics ---\n";
        std::cout << "Image Size: " << image_size << ", Num Snowmen: " << num_snowmen << ", MPI Processes: " << size << "\n";
        std::cout << "Max Local Computation Time (across all ranks): " << max_local_compute_time << " seconds\n";
        std::cout << "Min Local Computation Time (across all ranks): " << min_local_compute_time << " seconds\n";
        std::cout << "Avg Local Computation Time (across all ranks): " << avg_local_compute_time << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
