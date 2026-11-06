# FAR-LIO-CUDA

## Device-resident LiDAR pipeline

`SparseVoxel` uses a CUDA/cuCollections backend inspired by FAR-LIO's
[cuVoxelMap](https://github.com/TUMFTM/FAR-LIO/blob/main/lib_cpp/odometry_cpp/modules/map_handler/include/map_handler/voxel_hash_map.cuh).
The current compact implementation runs sparse insertion, covariances, local
1-NN/kNN, GICP point transforms, residuals, Jacobians and normal-equation reduction
on CUDA. The CPU solves the reduced 6x6 system, updates the pose and controls
convergence. Point arrays and correspondence arrays are not downloaded.

Implementation: [GICP.cu](source/GICP/GICP.cu),
[SparseVoxel.cu](source/types/SparseVoxel.cu), and
[SparseVoxel.cuh](include/localization/types/SparseVoxel.cuh).

### SparseVoxel device API

`SparseVoxel` exposes its device methods directly, without a separate device view.
Allocate it with `new` or `std::make_unique<SparseVoxel>()`: its allocation operator
places the object in CUDA unified memory. Do not use a stack object or
`std::make_shared` for GPU access.

```cpp
__global__ void query(const SparseVoxel* map, SparseVoxel::Neighbor* result)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *result = map->nearestNeighbor(Eigen::Vector3f::Zero(), 1.0f);
}

// Host:
auto map = std::make_unique<SparseVoxel>();
bool added = map->insert(cloud);
query<<<1, 1>>>(map.get(), device_result);
```

Only construction, destruction and `insert()` run on the host. `insert()` uploads
the cloud, then groups voxels, filters points and updates affected covariances on
the GPU. It blocks until completion and returns whether any point was added.
Input coordinates must be finite and in `[-16384, 16384)` m; an invalid batch is
rejected before the map changes. Hash occupancy is limited to half of its
`2^23`-slot budget.

`empty()`, `points()`, `nearestNeighbor()` and `nearestNeighbors()` are device-only.
`points()` returns a non-owning device span. `nearestNeighbors(query, k, output)`
writes up to `k` neighbors, sorted by squared distance, and returns their count;
provide space for `k` results (`k == 0` permits a null output). Queries must be
finite and within the coordinate range; `k` must be nonnegative and the 1-NN
distance limit must be positive. Both searches are local to 27 voxels, with
lexicographic voxel/insertion ordering for ties and a strict 1-NN distance limit.

Point addresses survive subsequent insertions; the span's pointer array may be
reallocated, so acquire a new span after insertion. Copies and moves are disabled.
Readers must finish before insertion; destruction waits for pending GPU work.
This API is also used by the compact GICP alignment implementation below.

### GICP CUDA alignment

`GICP::insertTargetPoints()` maintains a device-resident target map.
`GICP::align()` builds a device source map, then uses a fused CUDA
`transform_reduce` for point transformation, local correspondence search,
covariance weighting and double-precision normal equations. Only a 240-byte
aggregate is returned per iteration; Eigen's CPU LDLT and Sophus perform the
small solve and left pose update. There is no separate correspondence buffer or
CPU nearest-neighbor search.

The original covariance fallback, strict distance gate and iteration semantics
are preserved. `converged` retains the legacy meaning of at least one finite
completed iteration, including reaching the iteration limit. Quality thresholds
`min_correspondences` and `max_fitness_score` remain caller-side acceptance gates.
Copies are disabled; moves preserve ownership of the managed target map.

On an RTX 5080 / CUDA 13, a seeded rigid-transform benchmark compared this design
with a cooperative all-GPU loop. Both paths took three iterations; timings are
median wall-clock milliseconds over seven warmed calls, including source-map
construction, covariance estimation and result download:

| Input points | Cooperative GPU loop | Final hybrid implementation |
|---|---:|---:|
| 600 | 2.28 ms | 2.54 ms |
| 6,000 | 3.28 ms | 3.02 ms |
| 30,000 | 3.85 ms | 3.46 ms |

These are synthetic-workload measurements, not guarantees for every scan.
The simpler hybrid implementation was retained for its lower latency on the
larger scans; the cooperative loop was faster on the smallest input.

### Historical ROS input/output boundaries

The prepared-frame and preprocessing APIs in this section describe an earlier
implementation; they are not exposed by the current compact `GICP` interface.
The ROS node integration has not been restored or validated by these changes.

The default LO node uses `GICP::processFrame`: one raw-scan upload followed by GPU
preprocessing, registration, and conditional map updates, with one endpoint
synchronization for the compact result. Point clouds are downloaded only when
their visualization topics have subscribers; scan and map downloads share that
endpoint synchronization.

The LIO node uses `prepareSource` / `alignPrepared` and then CPU ESKF fusion.
`insertPreparedTarget` transforms and inserts the already-resident scan using
the posterior CPU pose, without uploading the points again or waiting on the
host. An optional `downloadPrepared` is an additional, explicit visualization
boundary after fusion. Asynchronous errors are reported by the next blocking
operation. Invalid input is rejected explicitly; fatal device/numerical errors
require recreating the pipeline rather than silently falling back to the CPU.

Workspaces, pinned staging buffers, and source/target allocations are reused.
Growth uses stream-ordered allocation; independent map copies use CUDA events
to order access across streams. Allocation growth and explicit compatibility
snapshots can still cost time; not all synchronization can be removed.

### Preserved numerical behavior

- The existing 0.5 m voxels, 36-point capacity, 0.1 m same-voxel minimum spacing,
  27-voxel search window, strict distance gate, and 8/12 covariance neighbors are
  unchanged. Insertion order within each voxel and 1-NN tie ordering are preserved.
- Covariances retain the existing formula:
  `C = (sample_covariance + 1e-6 I) * ||(sample_covariance + 1e-6 I)^-1||_F`.
  This is not PCL's fixed-spectrum covariance model.
- Covariances use float; GICP linearization, reduction, LDLT, and pose updates
  use double. No new damping or robust loss is introduced.
- Voxel keys store three signed 16-bit coordinates. With the default
  voxel size, positions must be in `[-16384, 16384)` m on every axis. Nonfinite
  and out-of-range positions throw instead of aliasing another voxel.
- Queries run directly inside caller kernels; there are no host query wrappers or
  host point snapshots. Returned point pointers address device memory and remain
  valid until map destruction. Concurrent read-only kernels are supported, but
  insertion must not overlap readers. CUDA failures throw; there is no CPU fallback.
- GPU preprocessing reports dropped nonfinite input points. Legacy XYZ map
  insertion/query APIs reject nonfinite input instead of filtering it.
- The visualization map preserves the initial processed scan and subsequently
  merges/downsamples centroids at the configured visualization resolution.
  No new spatial pruning or adaptive-density policy is added.

## Build and test

Requires ROS 2, PCL, Sophus, TBB, a CUDA toolkit with CCCL's CMake package, and a
CUDA-capable GPU. CUDA code uses C++20 and the ROS/C++ code uses C++23. CMake fetches
a pinned cuCollections revision including its CCCL 3/CUDA 13 compatibility fix;
for an offline build, pass
`-DFETCHCONTENT_SOURCE_DIR_CUCO_HEADERS=/path/to/cuCollections`.

```bash
source /opt/ros/jazzy/setup.bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sparse_voxel_gpu_test gicp_cuda_test -j2
ctest --test-dir build --output-on-failure -R '^(sparse_voxel_gpu_test|gicp_cuda_test)$'
```

The default CUDA architecture is `native`. For cross-compilation, set
`-DCMAKE_CUDA_ARCHITECTURES=<compute capability without a dot>`, e.g. `120`.
The focused regression compares device searches with local brute force and
covariances with the CPU formula. It also checks capacity/spacing, point lifetimes,
incremental covariance updates, empty maps, key boundaries, invalid input and
neighbor counts larger than the covariance neighborhood.
GICP tests compare against a CPU Eigen/Sophus implementation and a known rigid
transform, including empty targets, zero/limited iterations, no correspondences,
invalid input, covariance fallbacks, singular systems and move ownership.

Optional checks:

```bash
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 build/sparse_voxel_gpu_test
compute-sanitizer --tool initcheck --error-exitcode 1 build/sparse_voxel_gpu_test
compute-sanitizer --tool synccheck --error-exitcode 1 build/sparse_voxel_gpu_test
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 build/gicp_cuda_test
compute-sanitizer --tool initcheck --error-exitcode 1 build/gicp_cuda_test
compute-sanitizer --tool synccheck --error-exitcode 1 build/gicp_cuda_test
```

### Historical synchronization measurements

On an RTX 5080 / CUDA 13, the same seeded 6,000-point alignment (four successful
iterations, 40-iteration budget, no visualization) was measured before and after:

| Steady-state operation | Previous hybrid path | Device-resident path |
|---|---:|---:|
| Stream synchronizations | 33 | 1 |
| Device-to-host transfers | 11 | 1 |
| Device-to-host bytes | 1,544,776 | 448 |
| Synchronous allocations/frees | 21 / 21 | 0 / 0 |
| Unprofiled median alignment, 5 runs | 4.108 ms | 0.782 ms |

Nsight also verified one scan upload, one compact-result download, one endpoint
synchronization, and no allocations on the warmed full `processFrame` path.
That synthetic preprocessing/registration/map-update path measured 1.515 ms
median without visualization or ROS serialization. These are local synthetic
measurements, not a real-sensor trajectory-accuracy or latency guarantee.
