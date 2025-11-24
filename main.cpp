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
#include <cstring>
#include "raytracer.hpp"
#include "scene.hpp"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            std::cout << "Usage: " << argv[0] << " <image_size> <num_snowmen> <tile_size>\n";
        }
        MPI_Finalize();
        return 1;
    }

    int image_size = std::stoi(argv[1]);
    int num_snowmen = std::stoi(argv[2]);
    int tile_size = std::stoi(argv[3]);

    // Scene generation and RayTracer setup
    Scene scene;
    scene.generate_snowmen(num_snowmen);

    RayTracer raytracer(image_size, image_size);
    raytracer.set_scene(&scene);

    // accumulate local compute time (sum of tile times) per rank
    double local_compute_time = 0.0;

    // --- Master/Worker Tile-based rendering ---
    if (size == 1) {
        // Single-process fallback: render whole image as before
        std::vector<Color> pixels;
        auto t0 = std::chrono::high_resolution_clock::now();
        raytracer.render(0,1,pixels);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = t1 - t0;
        std::cout << "Single-rank render time: " << dur.count() << " s\n";
        raytracer.save_image("output.ppm", pixels);
        if (rank == 0) std::cout << "Image saved to output.ppm\n";
        local_compute_time = dur.count();
    } else {
        const int TILE_SIZE = tile_size;
        struct Tile { int id; int x0; int y0; int w; int h; };

        std::vector<Tile> tiles;
        int id = 0;
        for (int y = 0; y < image_size; y += TILE_SIZE) {
            for (int x = 0; x < image_size; x += TILE_SIZE) {
                int w = std::min(TILE_SIZE, image_size - x);
                int h = std::min(TILE_SIZE, image_size - y);
                tiles.push_back(Tile{id, x, y, w, h});
                ++id;
            }
        }

        int num_tiles = (int)tiles.size();

        if (rank == 0) {
            // Master: coordinate work and gather results
            std::vector<unsigned char> full_buf(image_size * image_size * 3);

            int next_tile = 0;
            // send initial tiles to workers
            for (int worker = 1; worker < size; ++worker) {
                if (next_tile < num_tiles) {
                    int meta[5] = {tiles[next_tile].id, tiles[next_tile].x0, tiles[next_tile].y0, tiles[next_tile].w, tiles[next_tile].h};
                    MPI_Send(meta, 5, MPI_INT, worker, 1, MPI_COMM_WORLD);
                    ++next_tile;
                } else {
                    MPI_Send(nullptr, 0, MPI_INT, worker, 2, MPI_COMM_WORLD); // done
                }
            }

            int tiles_received = 0;
            while (tiles_received < num_tiles) {
                MPI_Status status;
                int header[3]; // tile_id, w, h
                MPI_Recv(header, 3, MPI_INT, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &status);
                int src = status.MPI_SOURCE;
                int tile_id = header[0];
                int w = header[1];
                int h = header[2];
                int bufsize = w * h * 3;
                std::vector<unsigned char> buf(bufsize);
                MPI_Recv(buf.data(), bufsize, MPI_UNSIGNED_CHAR, src, 5, MPI_COMM_WORLD, &status);
                // receive elapsed time for this tile
                double elapsed = 0.0;
                MPI_Recv(&elapsed, 1, MPI_DOUBLE, src, 6, MPI_COMM_WORLD, &status);

                // place buffer into full_buf at correct offset
                Tile t = tiles[tile_id];
                for (int row = 0; row < t.h; ++row) {
                    int dest_row = t.y0 + row;
                    int dest_off = (dest_row * image_size + t.x0) * 3;
                    int src_off = row * t.w * 3;
                    std::memcpy(&full_buf[dest_off], &buf[src_off], t.w * 3);
                }

                ++tiles_received;

                // (elapsed time received; master does not aggregate per-worker here)

                // send next tile to this worker or done
                if (next_tile < num_tiles) {
                    int meta[5] = {tiles[next_tile].id, tiles[next_tile].x0, tiles[next_tile].y0, tiles[next_tile].w, tiles[next_tile].h};
                    MPI_Send(meta, 5, MPI_INT, src, 1, MPI_COMM_WORLD);
                    ++next_tile;
                } else {
                    MPI_Send(nullptr, 0, MPI_INT, src, 2, MPI_COMM_WORLD);
                }
            }

            // (master will compute standard MPI-reduced metrics after workers finish)

            // all tiles received -> save image
            std::vector<Color> full_pixels(image_size * image_size);
            for (int i = 0; i < image_size * image_size; ++i) {
                full_pixels[i] = Color(full_buf[3*i], full_buf[3*i + 1], full_buf[3*i + 2]);
            }
            raytracer.save_image("output.ppm", full_pixels);
            std::cout << "Master: Image saved to output.ppm\n";
        } else {
            // Worker loop
            while (true) {
                MPI_Status status;
                int meta[5];
                MPI_Recv(meta, 5, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (status.MPI_TAG == 2) {
                    break; // done
                }
                int tile_id = meta[0];
                int x0 = meta[1];
                int y0 = meta[2];
                int w = meta[3];
                int h = meta[4];

                unsigned int seed = static_cast<unsigned int>(tile_id * 10007u) ^ static_cast<unsigned int>(rank + 12345);
                double t0 = MPI_Wtime();
                std::vector<Color> out;
                raytracer.renderTile(x0, y0, w, h, seed, out);
                double t1 = MPI_Wtime();
                double elapsed = t1 - t0;

                // convert to bytes
                int bufsize = w * h * 3;
                std::vector<unsigned char> buf(bufsize);
                for (int i = 0; i < w * h; ++i) {
                    buf[3*i + 0] = out[i].r;
                    buf[3*i + 1] = out[i].g;
                    buf[3*i + 2] = out[i].b;
                }

                int header[3] = {tile_id, w, h};
                MPI_Send(header, 3, MPI_INT, 0, 4, MPI_COMM_WORLD);
                MPI_Send(buf.data(), bufsize, MPI_UNSIGNED_CHAR, 0, 5, MPI_COMM_WORLD);
                MPI_Send(&elapsed, 1, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);

                // accumulate local compute time for this worker
                local_compute_time += elapsed;

                // lightweight instrumentation to stderr
                std::cerr << "Rank " << rank << " rendered tile " << tile_id << " (" << w << "x" << h << ") in " << elapsed << " s\n";
            }
        }
    }

    // --- Report original-style performance metrics (max/min/avg local compute time) ---
    double max_local_compute_time = 0.0;
    double min_local_compute_time = 0.0;
    double sum_local_compute_time = 0.0;
    MPI_Reduce(&local_compute_time, &max_local_compute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute_time, &min_local_compute_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute_time, &sum_local_compute_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Gather all local compute times to rank 0 for per-rank output
    std::vector<double> all_local_compute_times(size);
    MPI_Gather(&local_compute_time, 1, MPI_DOUBLE, all_local_compute_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_local_compute_time = sum_local_compute_time / size;
        std::cout << "\n--- Computational Performance Metrics ---\n";
        std::cout << "Image Size: " << image_size << ", Num Snowmen: " << num_snowmen << ", MPI Processes: " << size << "\n";
        std::cout << "Max Local Computation Time (across all ranks): " << max_local_compute_time << " seconds\n";
        std::cout << "Min Local Computation Time (across all ranks): " << min_local_compute_time << " seconds\n";
        std::cout << "Avg Local Computation Time (across all ranks): " << avg_local_compute_time << " seconds\n";
        
        std::cout << "\n--- Per-Rank Computation Time ---\n";
        for (int i = 0; i < size; ++i) {
            std::cout << "Rank " << i << ": " << all_local_compute_times[i] << " seconds\n";
        }
    }

    MPI_Finalize();
    return 0;
}
