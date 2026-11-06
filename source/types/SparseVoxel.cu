#include "localization/types/SparseVoxel.cuh"
#include <cstdio>
#include <deque>
#include <stdexcept>
#include <string>
#include <Eigen/LU>
#include <cuco/static_map.cuh>
#include <cuda/std/array>
#include <cuda/std/atomic>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace
{
void check(const cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

void report(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
}
}

struct alignas(8) SparseVoxel::VoxelKey
{
    std::int16_t x, y, z, padding{0};

    __host__ __device__ VoxelKey() = default;
    __host__ __device__ VoxelKey(std::int16_t x, std::int16_t y, std::int16_t z, std::int16_t padding = 0)
        : x(x), y(y), z(z), padding(padding)
    {
    }

    __host__ __device__ bool operator==(const VoxelKey& rhs) const noexcept
    {
        return x == rhs.x && y == rhs.y && z == rhs.z && padding == rhs.padding;
    }

    __host__ __device__ bool operator<(const VoxelKey& rhs) const noexcept
    {
        if (x != rhs.x) return x < rhs.x;
        if (y != rhs.y) return y < rhs.y;
        return z < rhs.z;
    }
};

struct SparseVoxel::VoxelCell
{
    static constexpr int CAPACITY = 36;
    cuda::std::array<PointWithCovariance, CAPACITY> points;
    int count{0};
    unsigned long long covariance_epoch{0};

    __device__ void add(const Eigen::Vector3f& position)
    {
        if (count == CAPACITY) return;
        for (int i = 0; i < count; ++i)
            if ((points[i].position - position).squaredNorm() < MIN_DIST * MIN_DIST) return;

        auto& point = points[count++];
        point.position = position;
        point.covariance.setIdentity();
        point.covariance_valid = 0;
    }
};

struct SparseVoxel::Storage
{
    struct Hash
    {
        __host__ __device__ std::uint32_t operator()(const VoxelKey& key) const noexcept
        {
            return (static_cast<std::uint32_t>(key.x) * 73856093u) ^
                   (static_cast<std::uint32_t>(key.y) * 19349669u) ^
                   (static_cast<std::uint32_t>(key.z) * 83492791u);
        }
    };

    struct Equal
    {
        __host__ __device__ bool operator()(const VoxelKey& lhs, const VoxelKey& rhs) const noexcept
        {
            return lhs == rhs;
        }
    };

    using Probing = cuco::linear_probing<1, Hash>;
    using Extent = cuco::extent<std::size_t, MAX_MAP_SIZE>;
    using Map = cuco::static_map<VoxelKey, VoxelCell*, Extent, cuda::thread_scope_device, Equal, Probing>;
    using FindRef = Map::ref_type<cuco::find_tag>;

    static_assert(sizeof(VoxelKey) == 8 && alignof(VoxelKey) == 8);
    static_assert(cuco::is_bitwise_comparable_v<VoxelKey>);

    Map map{Extent{}, cuco::empty_key{VoxelKey{0, 0, 0, -1}},
            cuco::empty_value{static_cast<VoxelCell*>(nullptr)}};
    thrust::device_vector<FindRef> index{1, map.ref(cuco::find)};
    std::deque<thrust::device_vector<VoxelCell>> batches;
    thrust::device_vector<PointWithCovariance*> points;
    std::size_t voxel_count{0};

};

void* SparseVoxel::operator new(std::size_t size)
{
    void* pointer;
    check(cudaMallocManaged(&pointer, size), "allocate SparseVoxel");
    return pointer;
}

void SparseVoxel::operator delete(void* pointer) noexcept
{
    report(cudaFree(pointer), "free SparseVoxel");
}

SparseVoxel::SparseVoxel() : mStorage(new Storage), mIndex(mStorage->index.data().get())
{
}

SparseVoxel::~SparseVoxel()
{
    report(cudaDeviceSynchronize(), "finish SparseVoxel readers");
    delete mStorage;
}

__device__ SparseVoxel::VoxelKey SparseVoxel::voxelKey(const Eigen::Vector3f& position)
{
    return {static_cast<std::int16_t>(floorf(position.x() / VOXEL_SIZE)),
            static_cast<std::int16_t>(floorf(position.y() / VOXEL_SIZE)),
            static_cast<std::int16_t>(floorf(position.z() / VOXEL_SIZE))};
}

__device__ SparseVoxel::VoxelCell* SparseVoxel::findCell(const VoxelKey& key) const
{
    const auto& index = *static_cast<const Storage::FindRef*>(mIndex);
    const auto found = index.find(key);
    return found == index.end() ? nullptr : found->second;
}

__device__ int SparseVoxel::nearestNeighbors(const Eigen::Vector3f& query, int max_neighbors,
                                             Neighbor* output) const
{
    if (max_neighbors == 0 || empty()) return 0;
    const VoxelKey center = voxelKey(query);
    int count = 0;
    for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz)
            {
                const int x = center.x + dx, y = center.y + dy, z = center.z + dz;
                if (x < -32768 || x > 32767 || y < -32768 || y > 32767 || z < -32768 || z > 32767)
                    continue;
                const VoxelCell* cell = findCell({static_cast<std::int16_t>(x),
                                                static_cast<std::int16_t>(y),
                                                static_cast<std::int16_t>(z)});
                if (!cell) continue;
                for (int i = 0; i < cell->count; ++i)
                {
                    const auto* point = &cell->points[i];
                    const Eigen::Vector3f delta = point->position - query;
                    const float distance = delta.squaredNorm();
                    if (count == max_neighbors && distance >= output[count - 1].squared_distance) continue;
                    int at = count < max_neighbors ? count++ : count - 1;
                    while (at > 0 && distance < output[at - 1].squared_distance)
                    {
                        output[at] = output[at - 1];
                        --at;
                    }
                    output[at] = {point, distance};
                }
            }
    return count;
}

__device__ SparseVoxel::Neighbor SparseVoxel::nearestNeighbor(const Eigen::Vector3f& query, const float max_distance) const
{
    Neighbor result;
    if (nearestNeighbors(query, 1, &result) == 0 ||
        result.squared_distance >= max_distance * max_distance)
        return {};
    return result;
}

__device__ bool SparseVoxel::estimateCovariance(PointWithCovariance& point) const
{
    Neighbor neighbors[MAX_COVARIANCE_NEIGHBORS];
    const int count = nearestNeighbors(point.position, MAX_COVARIANCE_NEIGHBORS, neighbors);
    point.covariance.setIdentity();
    point.covariance_valid = 0;
    if (count < MIN_COVARIANCE_NEIGHBORS) return true;

    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (int i = 0; i < count; ++i) mean += neighbors[i].point->position;
    mean /= static_cast<float>(count);
    CovarianceMatrix covariance = CovarianceMatrix::Zero();
    for (int i = 0; i < count; ++i)
    {
        const Eigen::Vector3f delta = neighbors[i].point->position - mean;
        covariance.noalias() += delta * delta.transpose();
    }
    covariance /= static_cast<float>(count - 1);
    covariance.diagonal().array() += 1e-6f;
    const CovarianceMatrix inverse = covariance.inverse();
    covariance *= inverse.norm();
    // if (!covariance.allFinite()) return false;
    point.covariance = covariance;
    point.covariance_valid = 1;
    return true;
}

bool SparseVoxel::insert(const pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    if (cloud.empty()) return false;
    thrust::device_vector<pcl::PointXYZ> input(cloud.begin(), cloud.end());
    const bool valid = thrust::transform_reduce(input.begin(), input.end(),
        [] __device__(const pcl::PointXYZ& point) -> bool
    {
        return isfinite(point.x) && isfinite(point.y) && isfinite(point.z) &&
               point.x >= -16384.0f && point.x < 16384.0f &&
               point.y >= -16384.0f && point.y < 16384.0f &&
               point.z >= -16384.0f && point.z < 16384.0f;
    }, true, thrust::logical_and<bool>{});
    if (!valid) throw std::invalid_argument("SparseVoxel positions must be finite and in [-16384, 16384) m");

    const std::size_t size = input.size();
    thrust::device_vector<VoxelKey> keys(size);
    thrust::device_vector<std::size_t> order(size), starts(size + 1), new_groups(size);
    thrust::transform(input.begin(), input.end(), keys.begin(), [] __device__(const pcl::PointXYZ& point)
    {
        return voxelKey({point.x, point.y, point.z});
    });
    thrust::sequence(order.begin(), order.end());
    thrust::stable_sort_by_key(keys.begin(), keys.end(), order.begin());

    const auto first = thrust::make_counting_iterator<std::size_t>(0);
    const auto* key_data = keys.data().get();
    const auto groups_end = thrust::copy_if(thrust::device, first, first + size, starts.begin(),
        [=] __device__(std::size_t i) { return i == 0 || !(key_data[i] == key_data[i - 1]); });
    const std::size_t groups = groups_end - starts.begin();
    starts[groups] = size;
    const auto* boundaries = starts.data().get();
    const auto find = mStorage->map.ref(cuco::find);
    const auto new_end = thrust::copy_if(thrust::device, first, first + groups, new_groups.begin(),
        [=] __device__(std::size_t i) { return find.find(key_data[boundaries[i]]) == find.end(); });
    const std::size_t new_count = new_end - new_groups.begin();
    if (mStorage->voxel_count + new_count > MAX_MAP_SIZE / 2)
        throw std::length_error("SparseVoxel hash table reached its voxel budget");

    const std::size_t previous_count = mPointCount;
    mStorage->points.resize(previous_count + size);
    mPoints = mStorage->points.data().get();
    if (new_count != 0)
    {
        mStorage->batches.emplace_back(new_count);
        auto* cells = mStorage->batches.back().data().get();
        const auto* groups_data = new_groups.data().get();
        auto index = mStorage->map.ref(cuco::insert);
        thrust::for_each(thrust::device, first, first + new_count, [=] __device__(std::size_t i) mutable
        {
            index.insert(cuco::pair{key_data[boundaries[groups_data[i]]], cells + i});
        });
        mStorage->voxel_count += new_count;
    }

    const auto* input_data = input.data().get();
    const auto* indices = order.data().get();
    auto* output = mStorage->points.data().get();
    const auto epoch = ++mEpoch;
    thrust::for_each(thrust::device, first, first + groups, [=, this] __device__(const std::size_t group)
    {
        const VoxelKey key = key_data[boundaries[group]];
        VoxelCell* cell = findCell(key);
        const int old_count = cell->count;
        for (std::size_t i = boundaries[group]; i < boundaries[group + 1]; ++i)
        {
            const auto& point = input_data[indices[i]];
            cell->add({point.x, point.y, point.z});
        }
        const int added = cell->count - old_count;
        if (added == 0) return;
        const auto offset = cuda::atomic_ref<std::size_t, cuda::thread_scope_device>{mPointCount}
                                .fetch_add(added, cuda::memory_order_relaxed);
        for (int i = 0; i < added; ++i) output[offset + i] = &cell->points[old_count + i];

        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    const int x = key.x + dx, y = key.y + dy, z = key.z + dz;
                    if (x < -32768 || x > 32767 || y < -32768 || y > 32767 || z < -32768 || z > 32767)
                    {
                        continue;
                    }
                    VoxelCell* neighbor = findCell({static_cast<std::int16_t>(x),
                                                    static_cast<std::int16_t>(y),
                                                    static_cast<std::int16_t>(z)});
                    if (neighbor)
                    {
                        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>{neighbor->covariance_epoch}.store(epoch, cuda::memory_order_relaxed);
                    }
                }
            }
        }
    });

    thrust::for_each(thrust::device, first, first + previous_count + size,
        [=, this] __device__(std::size_t i)
        {
            if (i >= mPointCount) return;
            auto* point = output[i];
            if (findCell(voxelKey(point->position))->covariance_epoch == epoch) estimateCovariance(*point);
        });
    check(cudaDeviceSynchronize(), "insert SparseVoxel points");
    mStorage->points.resize(mPointCount);
    return mPointCount != previous_count;
}
