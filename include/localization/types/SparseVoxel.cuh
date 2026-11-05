#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <Eigen/Core>
#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cuda/stream_ref>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
// remove/remove_if are intentionally not used in the asynchronous pipeline:
// their returned end iterator requires the compacted item count on the host.
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cuco/static_map.cuh>
// static_set is available for set-only indices. The primary index below must
// remain a static_map because every occupied key maps to a voxel ID.
#include <cuco/static_set.cuh>
#include <cuda/std/atomic>
#include <cuda/std/tuple>
#include <cuda/std/array>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

using CovarianceMatrix = Eigen::Matrix<float, 3, 3, Eigen::ColMajor | Eigen::DontAlign>;

struct PointWithCovariance
{
    Eigen::Vector3f position;
    CovarianceMatrix covariance{CovarianceMatrix::Identity()};
    std::uint32_t covariance_valid{0};
};

struct Correspondence
{
    const thrust::device_ptr<PointWithCovariance> target{nullptr};
    Eigen::Vector3d transformed_position{Eigen::Vector3d::Zero()};
};

class SparseVoxel
{
    struct alignas(8) VoxelKey
    {
        std::int16_t x{};
        std::int16_t y{};
        std::int16_t z{};
        std::int16_t padding{0};

        __host__ __device__
        bool operator==(const VoxelKey& rhs) const noexcept
        {
            return x == rhs.x && y == rhs.y && z == rhs.z;
        }
    };

    static_assert(sizeof(VoxelKey) == 8);
    static_assert(alignof(VoxelKey) == 8);
    static_assert(cuco::is_bitwise_comparable_v<VoxelKey>);

    struct VoxelHash
    {
        __host__ __device__
        uint32_t operator()(const VoxelKey& k) const noexcept
        {
            return static_cast<uint32_t>(k.x) * 73856093 ^
                   static_cast<uint32_t>(k.y) * 19349669 ^
                   static_cast<uint32_t>(k.z) * 83492791;
        }
    };

    struct VoxelKeyEqual
    {
        __host__ __device__
        bool operator()(const VoxelKey& lhs, const VoxelKey& rhs) const noexcept
        {
            return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.padding == rhs.padding;
        }
    };

public:
    struct Neighbor
    {
        const thrust::device_ptr<PointWithCovariance> point{nullptr};
        float squared_distance{std::numeric_limits<float>::infinity()};
    };

    static constexpr float VOXEL_SIZE = 0.5f;

    static constexpr int MIN_COVARIANCE_NEIGHBORS = 8;
    static constexpr int MAX_COVARIANCE_NEIGHBORS = 12;

    SparseVoxel(const SparseVoxel&) = delete;
    SparseVoxel& operator=(const SparseVoxel&) = delete;
    SparseVoxel(SparseVoxel&&) = delete;
    SparseVoxel& operator=(SparseVoxel&&) = delete;
    ~SparseVoxel() = default;

    SparseVoxel() = default;

    bool insert(const pcl::PointCloud<pcl::PointXYZ>& cloud);

    void clear() noexcept
    {

    }

    bool empty() const noexcept
    {

    }

    std::size_t pointCount() const noexcept
    {

    }

    std::size_t voxelCount() const noexcept
    {

    }


    thrust::device_vector<const PointWithCovariance*> points() const;


    Neighbor nearestNeighbor(const Eigen::Vector3f& query, float max_distance) const;


    thrust::device_vector<Neighbor> nearestNeighbors(const Eigen::Vector3f& query, int max_neighbors) const;

private:

    static constexpr float MIN_DIST = 0.1f;
    static constexpr std::size_t MAX_MAP_SIZE = 1 << 23;  // 2^23 = 8388608

    struct VoxelCell
    {
        static constexpr int CAPACITY = 36;
        static constexpr float MIN_DIST_SQUARE = MIN_DIST * MIN_DIST;

        using Points = cuda::std::array<PointWithCovariance, CAPACITY>;

        Points points{};
        int count{0};

        __device__ PointWithCovariance* add(const Eigen::Vector3f& position)
        {
            if (count >= CAPACITY)
            {
                return nullptr;
            }

            const bool too_close = thrust::any_of(thrust::seq, points.begin(), points.begin() + count,
                [&position] __device__(const PointWithCovariance& point)
                {
                    return (point.position - position).squaredNorm() < MIN_DIST_SQUARE;
                });

            if (too_close)
            {
                return nullptr;
            }

            PointWithCovariance& point = points[count];
            point.position = position;
            point.covariance.setIdentity();
            point.covariance_valid = 0;
            ++count;

            return &point;
        }
    };

    using OccupidVoxel = cuco::static_map<VoxelKey, VoxelCell*, cuco::extent<std::size_t, MAX_MAP_SIZE>,
                                           cuda::thread_scope_device, VoxelKeyEqual, cuco::linear_probing<1, VoxelHash>>;


    static constexpr int VOXEL_CAPACITY = 36;
    static constexpr int NEIGHBOR_VOXEL_COUNT = 27;

    __device__
    void estimateCovariances(const std::vector<PointWithCovariance*>& points);
};
