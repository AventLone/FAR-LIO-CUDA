# FAR-LIO-CUDA

## Device-resident LiDAR pipeline

`SparseVoxel` uses a CUDA/cuCollections backend inspired by FAR-LIO's
[cuVoxelMap](https://github.com/TUMFTM/FAR-LIO/blob/main/lib_cpp/odometry_cpp/modules/map_handler/include/map_handler/voxel_hash_map.cuh).
The numerical hot path stays on one CUDA stream:

- Intensity/footprint filtering, centroid voxel downsampling, and sensor transforms.
- Per-point motion compensation using the CPU-propagated IMU trajectory.
- Sparse map insertion, incremental covariances, and local 1-NN/kNN searches.
- GICP transforms, residuals, Jacobians, packed Hessian/gradient reduction,
  pivoted LDLT solves, SE(3) updates, and convergence/acceptance decisions.
- Registration-map updates and the separate centroid map used for visualization.

The CPU handles ROS decoding/publishing, first-frame intensity statistics, IMU
trajectory propagation, and ESKF state fusion. Point arrays, source/target maps,
and intermediate normal equations are not downloaded during registration.
The host queues a fixed iteration budget; device flags stop numerical work after
convergence without polling or synchronizing each iteration.

Implementation: [GICP.cu](source/GICP/GICP.cu),
[SparseVoxel.cu](source/types/SparseVoxel.cu), and
[SparseVoxelDevice.cuh](include/localization/types/SparseVoxelDevice.cuh).

### Input/output boundaries

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
- Voxel keys pack three signed 21-bit coordinates into 63 bits. With the default
  voxel size, positions must be in `[-524288, 524288)` m on every axis. Nonfinite
  and out-of-range positions throw instead of aliasing another voxel.
- Scalar queries remain available; use `nearestNeighborBatch` and
  `nearestNeighborsBatch` for compatibility queries. Each batch preserves query order.
  These host-facing APIs explicitly download results; the registration path uses
  device views instead. Host point snapshots are lazy. Once pointers have been
  exposed, standalone insertion refreshes the snapshot to preserve their values.
  Returned host point pointers remain valid across insertions, but not after
  clearing, assignment, or destruction of their owning map.
- Map copies are independent; moves and clear/reinsert are supported. Concurrent
  read-only queries are serialized on the map's CUDA stream. Insertion/clearing
  must not overlap readers. CUDA failures throw; there is no silent CPU fallback.
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
cmake --build build -j2
ctest --test-dir build --output-on-failure -R '^gpu_resident_regression$'
```

The default CUDA architecture is `native`. For cross-compilation, set
`-DCMAKE_CUDA_ARCHITECTURES=<compute capability without a dot>`, e.g. `120`.
The focused regression compares search with local brute force, covariances with
the CPU formula, and GPU registration with an independent CPU Eigen/Sophus solver.
It also checks capacity/spacing, copies and point lifetimes, concurrent queries,
rank deficiency, zero iterations, invalid input, centroid-map updates, and deskew
including trajectories extending beyond the scan end.

Optional checks:

```bash
compute-sanitizer --tool memcheck --track-stream-ordered-races all --leak-check full --error-exitcode 1 build/gpu_resident_regression
compute-sanitizer --tool initcheck --error-exitcode 1 build/gpu_resident_regression
compute-sanitizer --tool synccheck --error-exitcode 1 build/gpu_resident_regression
nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop -o gpu-align build/gpu_resident_regression --profile
nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop -o gpu-frame build/gpu_resident_regression --profile-frame
```

### Measured synchronization reduction

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
