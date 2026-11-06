#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <Eigen/Core>
#include <cuda_runtime.h>
#include <cuda/std/span>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using CovarianceMatrix = Eigen::Matrix<float, 3, 3, Eigen::ColMajor | Eigen::DontAlign>;

struct PointWithCovariance
{
    Eigen::Vector3f position;
    CovarianceMatrix covariance{CovarianceMatrix::Identity()};
    std::uint32_t covariance_valid{0};
};

struct Correspondence
{
    const PointWithCovariance* target{nullptr};
    Eigen::Vector3d transformed_position{Eigen::Vector3d::Zero()};
};

class SparseVoxel
{
public:
    // NVCC-generated kernel stubs need these names to be public.
    struct VoxelKey;
    struct VoxelCell;
    struct Storage;

    struct Neighbor
    {
        const PointWithCovariance* point{nullptr};
        float squared_distance{std::numeric_limits<float>::infinity()};
    };

    static constexpr float VOXEL_SIZE = 0.5f;
    static constexpr float MIN_DIST = 0.1f;
    static constexpr int MIN_COVARIANCE_NEIGHBORS = 8;
    static constexpr int MAX_COVARIANCE_NEIGHBORS = 12;

    // Allocate with new/make_unique so the object itself is GPU-accessible.
    static void* operator new(std::size_t size);
    static void operator delete(void* pointer) noexcept;
    SparseVoxel();
    ~SparseVoxel();
    SparseVoxel(const SparseVoxel&) = delete;
    SparseVoxel& operator=(const SparseVoxel&) = delete;
    SparseVoxel(SparseVoxel&&) = delete;
    SparseVoxel& operator=(SparseVoxel&&) = delete;

    // Blocking host entry: true iff at least one point was added.
    bool insert(const pcl::PointCloud<pcl::PointXYZ>& cloud);

    __device__
    bool empty() const noexcept
    {
        return mPointCount == 0;
    }

    __device__
    cuda::std::span<const PointWithCovariance* const> points() const noexcept
    {
        return {mPoints, mPointCount};
    }

    __device__
    Neighbor nearestNeighbor(const Eigen::Vector3f& query, float max_distance) const;

    // output has room for max_neighbors; only the returned prefix is populated.
    __device__
    int nearestNeighbors(const Eigen::Vector3f& query, int max_neighbors, Neighbor* output) const;

private:
    static constexpr std::size_t MAX_MAP_SIZE = 1 << 23;

    __device__ static VoxelKey voxelKey(const Eigen::Vector3f& position);
    __device__ VoxelCell* findCell(const VoxelKey& key) const;
    __device__ bool estimateCovariance(PointWithCovariance& point) const;

    Storage* mStorage;
    void* mIndex;
    const PointWithCovariance* const* mPoints{nullptr};
    std::size_t mPointCount{0};
    unsigned long long mEpoch{0};
};
