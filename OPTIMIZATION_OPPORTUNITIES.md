# Raytracer optimization opportunities

The notes below summarize code paths that appear on the critical rendering path and should provide noticeable performance improvements once optimized.

## Hot loops and per-pixel work
- **Snowflake overlay:** Every pixel iterates over 75k snowflakes and performs multiple vector operations before breaking out, even when the snowflake is far outside the ray. Spatial binning (uniform grid/tiles), reduced particle count per tile, or precomputing screen-space bounds would dramatically cut the O(pixels * flakes) cost while preserving visuals.
- **Primitive intersections:** For each pixel we scan every sphere and plane separately, even though the scene mostly consists of snowmen. Grouping planes (floor) separately and building a simple BVH or spatial grid for spheres would reduce the per-pixel primitive count and improve cache locality.
- **Shadow rays:** Shadow tests re-iterate through all spheres/planes for every hit, duplicating work already done for the primary ray. Caching the closest blocker per pixel or reusing sphere radii-squared to avoid extra sqrt operations would lower the per-pixel arithmetic.
- **Math reuse:** Row-level constants (`ndc_y`, `py`) and camera basis vectors are recomputed inside inner loops. Hoisting row-specific terms and precomputing `radius^2` for spheres can reduce FLOPs in the tightest loops.

## Data layout
- **Snowflakes and scene primitives use `double` vectors.** Converting snowflake positions and possibly sphere centers to `float` (or struct-of-arrays) would halve bandwidth and allow SIMD-friendly loads while retaining visual fidelity.
- **Color conversion overhead:** Tiles are produced as `unsigned char` buffers and immediately converted back to `Color`. Keeping a single contiguous `unsigned char` framebuffer on rank 0 and only materializing `Color` for final output would avoid repeated per-pixel structs and copies.

## Scheduling and communication
- **Tile sizing:** Rank 0 uses half-height tiles and sends work to `size-1` workers, causing frequent small messages at higher core counts. Increasing `tiles_per_worker` or adopting larger tiles for the master would reduce MPI traffic and better amortize latency.
- **Snowflake broadcast:** Snowflake positions are regenerated and broadcast on every render. Generating once and caching on rank 0 (then rebroadcasting only when the RNG seed changes) would avoid repeated large broadcasts.
- **Master participation:** Rank 0 interleaves probing and rendering small tiles. Deferring its local work until the job queue empties (or dedicating it to coordination only) would reduce cache churn and allow more predictable progress on large node counts.

## Memory
- **Repeated `std::vector` reallocations** in worker loops (`send_buf`, `recv_buf`, `local_buf`) occur every tile. Reserving buffers once per rank and reusing them would eliminate allocator overhead and improve cache reuse.

## Low-level tweaks
- Use `restrict`-like semantics (compiler flags) and inline-friendly helpers for `Vec3` to enable auto-vectorization in `compute_tile_flat`.
- Avoid `std::pow`/`std::sqrt` where a squared comparison suffices (e.g., snowflake radius checks) to reduce transcendental cost in the innermost loop.
