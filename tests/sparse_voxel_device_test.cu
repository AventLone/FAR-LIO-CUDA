#include "localization/types/SparseVoxel.cuh"
#include <Eigen/LU>
#include <thrust/host_vector.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using Key = std::array<int, 3>;

static_assert(std::is_trivially_copyable_v<SparseVoxel::DeviceView>);
static_assert(!std::is_copy_constructible_v<SparseVoxel>);
static_assert(!std::is_move_constructible_v<SparseVoxel>);

void check(cudaError_t error)
{
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void requireThrows(Function function, const std::string& message)
{
    try { function(); }
    catch (const Exception&) { return; }
    throw std::runtime_error(message);
}

template <typename T>
T* data(thrust::device_vector<T>& values)
{
    return thrust::raw_pointer_cast(values.data());
}

struct Summary
{
    std::size_t points{};
    std::size_t voxels{};
    SparseVoxel::DeviceStatus status{};
    bool empty{};
};

struct Sample
{
    const PointWithCovariance* address;
    PointWithCovariance point;
};

__global__ void inspect(SparseVoxel::DeviceView map, Summary* summary)
{
    *summary = {map.pointCount(), map.voxelCount(), map.status(), map.empty()};
}

__global__ void snapshot(SparseVoxel::DeviceView map, Sample* output)
{
    const auto points = map.points();
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < points.size; i += blockDim.x * gridDim.x)
    {
        output[i].address = points.data[i];
        output[i].point = *points.data[i];
    }
}

__global__ void clearMap(SparseVoxel::DeviceView map)
{
    map.clear();
}

__global__ void query(SparseVoxel::DeviceView map, const Eigen::Vector3f* queries, std::size_t count,
                      float radius, int k, SparseVoxel::Neighbor* nearest, SparseVoxel::Neighbor* neighbors, int* sizes)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
    {
        nearest[i] = map.nearestNeighbor(queries[i], radius);
        sizes[i] = map.nearestNeighbors(queries[i], k, k > 0 ? neighbors + i * k : nullptr);
    }
}

__global__ void invalidQueries(SparseVoxel::DeviceView map, int mode)
{
    SparseVoxel::Neighbor result[1];
    if (mode == 0) map.nearestNeighbor(Eigen::Vector3f::Zero(), -1.0f);
    else if (mode == 1) map.nearestNeighbors(Eigen::Vector3f::Zero(), 1, nullptr);
    else map.nearestNeighbors(Eigen::Vector3f(std::numeric_limits<float>::infinity(), 0.0f, 0.0f), 1, result);
}

Summary summary(const SparseVoxel& map)
{
    thrust::device_vector<Summary> output(1);
    inspect<<<1, 1>>>(map.device, data(output));
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    const thrust::host_vector<Summary> host = output;
    return host[0];
}

std::vector<Sample> samples(const SparseVoxel& map)
{
    const auto count = summary(map).points;
    thrust::device_vector<Sample> output(count);
    if (count != 0)
    {
        snapshot<<<32, 128>>>(map.device, data(output));
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
    }
    const thrust::host_vector<Sample> host = output;
    return {host.begin(), host.end()};
}

void clear(SparseVoxel& map)
{
    clearMap<<<1, 128>>>(map.device);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
}

Key key(const Eigen::Vector3f& point)
{
    return {static_cast<int>(std::floor(point.x() / SparseVoxel::VOXEL_SIZE)),
            static_cast<int>(std::floor(point.y() / SparseVoxel::VOXEL_SIZE)),
            static_cast<int>(std::floor(point.z() / SparseVoxel::VOXEL_SIZE))};
}

void checkSearch(const SparseVoxel& map, const Cloud& input)
{
    auto points = samples(map);
    std::stable_sort(points.begin(), points.end(), [](const Sample& lhs, const Sample& rhs)
    {
        return key(lhs.point.position) < key(rhs.point.position);
    });
    thrust::host_vector<Eigen::Vector3f> host_queries;
    for (const auto& point : input) host_queries.push_back(point.getVector3fMap());
    thrust::device_vector<Eigen::Vector3f> queries = host_queries;
    thrust::device_vector<SparseVoxel::Neighbor> nearest(input.size());
    thrust::device_vector<int> sizes(input.size());
    for (const int k : {-1, 0, 1, 8, 12, 36, 1000})
    {
        thrust::device_vector<SparseVoxel::Neighbor> neighbors(input.size() * static_cast<std::size_t>(std::max(k, 0)));
        for (const float radius : {0.0f, 0.1f, 0.5f, 1.5f})
        {
            query<<<8, 128>>>(map.device, data(queries), input.size(), radius, k, data(nearest), data(neighbors), data(sizes));
            check(cudaGetLastError());
            check(cudaDeviceSynchronize());
            const thrust::host_vector<SparseVoxel::Neighbor> actual_nearest = nearest;
            const thrust::host_vector<SparseVoxel::Neighbor> actual_neighbors = neighbors;
            const thrust::host_vector<int> actual_sizes = sizes;
            for (std::size_t i = 0; i < input.size(); ++i)
            {
                std::vector<std::pair<float, const PointWithCovariance*>> expected;
                const Eigen::Vector3f position = input[i].getVector3fMap();
                const Key query_key = key(position);
                for (const auto& point : points)
                {
                    const Key point_key = key(point.point.position);
                    if (std::abs(point_key[0] - query_key[0]) <= 1 && std::abs(point_key[1] - query_key[1]) <= 1 &&
                        std::abs(point_key[2] - query_key[2]) <= 1)
                        expected.emplace_back((position - point.point.position).squaredNorm(), point.address);
                }
                std::stable_sort(expected.begin(), expected.end(), [](const auto& lhs, const auto& rhs)
                {
                    return lhs.first < rhs.first;
                });
                const bool found = !expected.empty() && expected.front().first < radius * radius;
                require(thrust::raw_pointer_cast(actual_nearest[i].point) == (found ? expected.front().second : nullptr),
                        "Device 1-NN point, tie ordering, or strict radius gate differs from brute force");
                require(actual_nearest[i].squared_distance == (found ? expected.front().first : radius * radius),
                        "Device 1-NN squared distance differs from brute force");
                const std::size_t count = k <= 0 ? 0 : std::min<std::size_t>(k, expected.size());
                require(actual_sizes[i] == static_cast<int>(count), "Device kNN returned the wrong count");
                std::vector<float> distances;
                std::vector<const PointWithCovariance*> addresses;
                for (std::size_t j = 0; j < count; ++j)
                {
                    const auto& value = actual_neighbors[i * k + j];
                    distances.push_back(value.squared_distance);
                    addresses.push_back(thrust::raw_pointer_cast(value.point));
                }
                std::sort(distances.begin(), distances.end());
                for (std::size_t j = 0; j < count; ++j)
                    require(distances[j] == expected[j].first, "Device kNN distances differ from brute force");
                std::sort(addresses.begin(), addresses.end(), std::less<const PointWithCovariance*>{});
                require(std::adjacent_find(addresses.begin(), addresses.end()) == addresses.end(), "Device kNN duplicated a point");
            }
        }
    }
    require(summary(map).status == SparseVoxel::DeviceStatus::success, "A valid search set an error");
}

void checkCovariances(const SparseVoxel& map)
{
    const auto points = samples(map);
    if (points.empty()) return;
    constexpr int k = SparseVoxel::MAX_COVARIANCE_NEIGHBORS;
    thrust::host_vector<Eigen::Vector3f> positions;
    for (const auto& point : points) positions.push_back(point.point.position);
    thrust::device_vector<Eigen::Vector3f> queries = positions;
    thrust::device_vector<SparseVoxel::Neighbor> nearest(points.size());
    thrust::device_vector<SparseVoxel::Neighbor> neighbors(points.size() * k);
    thrust::device_vector<int> sizes(points.size());
    query<<<32, 128>>>(map.device, data(queries), points.size(), 0.5f, k, data(nearest), data(neighbors), data(sizes));
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    const thrust::host_vector<SparseVoxel::Neighbor> host_neighbors = neighbors;
    const thrust::host_vector<int> host_sizes = sizes;
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto& point = points[i].point;
        const int count = host_sizes[i];
        if (count < SparseVoxel::MIN_COVARIANCE_NEIGHBORS)
        {
            require(point.covariance_valid == 0 && point.covariance.isIdentity(), "Sparse covariance fallback changed");
            continue;
        }
        std::vector<Eigen::Vector3f> neighborhood;
        for (int j = 0; j < count; ++j)
        {
            const auto* address = thrust::raw_pointer_cast(host_neighbors[i * k + j].point);
            const auto found = std::find_if(points.begin(), points.end(), [address](const Sample& value)
            {
                return value.address == address;
            });
            require(found != points.end(), "kNN returned an address outside the GPU map");
            neighborhood.push_back(found->point.position);
        }
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (const auto& position : neighborhood) mean += position;
        mean /= static_cast<float>(count);
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const auto& position : neighborhood)
        {
            const Eigen::Vector3f offset = position - mean;
            covariance.noalias() += offset * offset.transpose();
        }
        covariance /= static_cast<float>(count - 1);
        covariance.diagonal().array() += 1e-6f;
        const Eigen::Matrix3f expected = covariance * covariance.inverse().norm();
        require(point.covariance_valid != 0 && point.covariance.isApprox(expected, 2e-5f),
                "GPU covariance differs from the CPU Frobenius formula");
    }
}

void regression()
{
    SparseVoxel map;
    const auto initial = summary(map);
    require(initial.empty && initial.points == 0 && initial.voxels == 0, "A new GPU map is not empty");
    require(!map.insert(Cloud{}), "Empty insert must return false");
    Cloud boundaries;
    for (const float face : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f})
        for (const float x : {std::nextafter(face, -INFINITY), face, std::nextafter(face, INFINITY)})
            boundaries.emplace_back(x, face, -face);
    checkSearch(map, boundaries);
    map.insert(boundaries);
    checkSearch(map, boundaries);
    clear(map);
    Cloud crowded;
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y)
            for (int z = 0; z < 4; ++z)
                crowded.emplace_back(0.02f + 0.11f * x, 0.02f + 0.11f * y, 0.02f + 0.11f * z);
    map.insert(crowded);
    require(summary(map).points == 36 && summary(map).voxels == 1, "The per-voxel capacity changed");
    const auto original = samples(map);
    for (const auto& point : original)
    {
        bool expected = false;
        for (std::size_t i = 0; i < 36; ++i)
            expected = expected || point.point.position == crowded[i].getVector3fMap();
        require(expected, "Insertion did not preserve the first accepted points of a voxel");
    }
    map.insert(crowded);
    require(summary(map).points == original.size(), "Duplicate insertion retained new points");
    Cloud growth;
    for (int i = 1; i < 400; ++i) growth.emplace_back(0.5f * i + 0.02f, 3.0f, -2.0f);
    map.insert(growth);
    require(summary(map).points == original.size() + growth.size(), "Pool growth lost accepted points");
    const auto grown = samples(map);
    for (const auto& point : original)
    {
        const auto found = std::find_if(grown.begin(), grown.end(), [&](const Sample& candidate)
        {
            return candidate.address == point.address;
        });
        require(found != grown.end() && found->point.position == point.point.position,
                "Pool growth invalidated an existing device point address");
    }
    checkCovariances(map);
    clear(map);
    Cloud first, rest;
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 3; ++y)
            (first.size() < 7 ? first : rest).emplace_back(0.11f * x, 0.11f * y, 0.0f);
    map.insert(first);
    checkCovariances(map);
    map.insert(rest);
    checkCovariances(map);
    for (const auto& point : samples(map)) require(point.point.covariance_valid != 0, "Old covariance was not refreshed");
    const auto before = summary(map);
    Cloud invalid;
    invalid.emplace_back(2.0f, 2.0f, 2.0f);
    invalid.emplace_back(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f);
    requireThrows<std::invalid_argument>([&] { map.insert(invalid); }, "Nonfinite input must be rejected");
    require(summary(map).points == before.points && summary(map).voxels == before.voxels,
            "Rejected insertion partially changed the map");
    invalid.back().x = 16384.0f;
    requireThrows<std::out_of_range>([&] { map.insert(invalid); }, "A voxel coordinate must not wrap its int16 key");
    map.insert(rest);
    for (int mode = 0; mode < 3; ++mode)
    {
        invalidQueries<<<1, 1>>>(map.device, mode);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        require(summary(map).status != SparseVoxel::DeviceStatus::success, "Invalid device query failed silently");
        clear(map);
        require(summary(map).status == SparseVoxel::DeviceStatus::success && summary(map).empty,
                "Device clear did not reset counts/status");
    }
    Cloud limits;
    limits.emplace_back(-16384.0f, 0.0f, 0.0f);
    limits.emplace_back(16383.5f, 0.0f, 0.0f);
    map.insert(limits);
    checkSearch(map, limits);
    clear(map);
    Cloud line;
    for (int i = 0; i < 8; ++i) line.emplace_back(0.11f * (i - 4), 0.0f, 0.0f);
    map.insert(line);
    checkCovariances(map);
    clear(map);
    std::mt19937 random(17);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    Cloud cloud;
    for (int i = 0; i < 1500; ++i)
        cloud.emplace_back(3.0f * unit(random), 1.5f * unit(random), 0.2f * std::sin(i * 0.1f));
    map.insert(cloud);
    Cloud queries;
    for (int i = 0; i < 60; ++i) queries.push_back(cloud[static_cast<std::size_t>(i) * 23]);
    checkSearch(map, queries);
    checkCovariances(map);
}
}

int main()
{
    try
    {
        regression();
        std::cout << "GPU-only SparseVoxel query/count/clear and insertion/covariance checks passed\n";
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "SparseVoxel device regression failed: " << exception.what() << '\n';
        return 1;
    }
}
