#include "localization/types/SparseVoxel.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <Eigen/LU>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void synchronize()
{
    const auto error = cudaDeviceSynchronize();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct Sample
{
    const PointWithCovariance* address;
    PointWithCovariance value;
};

struct Summary
{
    std::size_t count;
    bool empty;
};

__global__ void snapshot(const SparseVoxel* map, Sample* output, Summary* summary)
{
    const auto points = map->points();
    if (threadIdx.x == 0 && blockIdx.x == 0) *summary = {points.size(), map->empty()};
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < points.size();
         i += blockDim.x * gridDim.x)
        output[i] = {points[i], *points[i]};
}

__global__ void query(const SparseVoxel* map, const Eigen::Vector3f* queries, int count,
                      float radius, int k, SparseVoxel::Neighbor* nearest,
                      SparseVoxel::Neighbor* neighbors, int* sizes)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    nearest[i] = map->nearestNeighbor(queries[i], radius);
    sizes[i] = map->nearestNeighbors(queries[i], k, k == 0 ? nullptr : neighbors + i * k);
}

auto key(const Eigen::Vector3f& point)
{
    return std::tuple{static_cast<int>(std::floor(point.x() / SparseVoxel::VOXEL_SIZE)),
                      static_cast<int>(std::floor(point.y() / SparseVoxel::VOXEL_SIZE)),
                      static_cast<int>(std::floor(point.z() / SparseVoxel::VOXEL_SIZE))};
}

std::vector<Sample> read(const SparseVoxel* map, std::size_t upper_bound)
{
    thrust::device_vector<Sample> output(upper_bound);
    thrust::device_vector<Summary> summary(1);
    snapshot<<<2, 128>>>(map, output.data().get(), summary.data().get());
    synchronize();
    const Summary metadata = summary[0];
    require(metadata.empty == (metadata.count == 0), "empty() disagrees with points()");
    const thrust::host_vector<Sample> host(output.begin(), output.begin() + metadata.count);
    std::vector<Sample> result(host.begin(), host.end());
    std::stable_sort(result.begin(), result.end(), [](const Sample& lhs, const Sample& rhs)
    {
        return key(lhs.value.position) < key(rhs.value.position);
    });
    return result;
}

std::vector<SparseVoxel::Neighbor> brute(const std::vector<Sample>& samples, const Eigen::Vector3f& query)
{
    std::vector<SparseVoxel::Neighbor> result;
    const auto [qx, qy, qz] = key(query);
    for (const auto& sample : samples)
    {
        const auto [x, y, z] = key(sample.value.position);
        if (std::abs(x - qx) <= 1 && std::abs(y - qy) <= 1 && std::abs(z - qz) <= 1)
            result.push_back({sample.address, (sample.value.position - query).squaredNorm()});
    }
    std::stable_sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs)
    {
        return lhs.squared_distance < rhs.squared_distance;
    });
    return result;
}

void checkQueries(const SparseVoxel* map, const std::vector<Sample>& samples,
                  const std::vector<Eigen::Vector3f>& queries, int k, float radius)
{
    const int count = static_cast<int>(queries.size());
    thrust::device_vector<Eigen::Vector3f> input(queries.begin(), queries.end());
    thrust::device_vector<SparseVoxel::Neighbor> nearest(count), neighbors(count * k);
    thrust::device_vector<int> sizes(count);
    query<<<(count + 127) / 128, 128>>>(map, input.data().get(), count, radius, k,
        nearest.data().get(), neighbors.data().get(), sizes.data().get());
    synchronize();
    const thrust::host_vector<SparseVoxel::Neighbor> host_nearest = nearest, host_neighbors = neighbors;
    const thrust::host_vector<int> host_sizes = sizes;
    for (int i = 0; i < count; ++i)
    {
        const auto expected = brute(samples, queries[i]);
        const auto* closest = !expected.empty() && expected[0].squared_distance < radius * radius
                                  ? expected[0].point : nullptr;
        if (host_nearest[i].point != closest)
        {
            std::cerr << "Query " << queries[i].transpose() << ", radius=" << radius
                      << ", GPU=" << host_nearest[i].point << ", CPU=" << closest
                      << ", GPU distance=" << host_nearest[i].squared_distance
                      << ", CPU distance=" << (expected.empty() ? -1.0f : expected[0].squared_distance) << '\n';
            throw std::runtime_error("nearestNeighbor differs from local brute force");
        }
        if (!closest)
            require(std::isinf(host_nearest[i].squared_distance), "missing neighbor distance is not infinity");
        const int expected_size = std::min(k, static_cast<int>(expected.size()));
        require(host_sizes[i] == expected_size, "nearestNeighbors returned an incorrect count");
        for (int j = 0; j < expected_size; ++j)
        {
            const auto& actual = host_neighbors[i * k + j];
            require(actual.point == expected[j].point, "kNN point or tie ordering differs from brute force");
            require(std::abs(actual.squared_distance - expected[j].squared_distance) < 1e-5f,
                    "kNN distance differs from brute force");
        }
    }
}

void checkCovariances(const std::vector<Sample>& samples)
{
    for (const auto& sample : samples)
    {
        auto neighbors = brute(samples, sample.value.position);
        neighbors.resize(std::min(neighbors.size(), std::size_t{SparseVoxel::MAX_COVARIANCE_NEIGHBORS}));
        if (neighbors.size() < SparseVoxel::MIN_COVARIANCE_NEIGHBORS)
        {
            require(!sample.value.covariance_valid && sample.value.covariance.isIdentity(),
                    "insufficient neighbors must leave an invalid identity covariance");
            continue;
        }
        std::vector<Eigen::Vector3f> positions;
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (const auto& neighbor : neighbors)
        {
            const auto found = std::find_if(samples.begin(), samples.end(), [&](const Sample& candidate)
            {
                return candidate.address == neighbor.point;
            });
            positions.push_back(found->value.position);
            mean += positions.back();
        }
        mean /= static_cast<float>(positions.size());
        CovarianceMatrix expected = CovarianceMatrix::Zero();
        for (const auto& position : positions)
        {
            const Eigen::Vector3f delta = position - mean;
            expected.noalias() += delta * delta.transpose();
        }
        expected /= static_cast<float>(positions.size() - 1);
        expected.diagonal().array() += 1e-6f;
        expected *= expected.inverse().norm();
        if (!sample.value.covariance_valid || !sample.value.covariance.isApprox(expected, 2e-4f))
        {
            std::cerr << "Covariance at " << sample.value.position.transpose()
                      << ", neighbors=" << neighbors.size() << ", valid=" << sample.value.covariance_valid
                      << ", address=" << sample.address
                      << "\nGPU:\n" << sample.value.covariance << "\nCPU:\n" << expected << '\n';
            throw std::runtime_error("GPU covariance differs from the CPU formula");
        }
    }
}

void run()
{
    auto map = std::make_unique<SparseVoxel>();
    require(read(map.get(), 0).empty(), "new map is not empty");
    checkQueries(map.get(), {}, {Eigen::Vector3f::Zero()}, 12, 1.0f);
    pcl::PointCloud<pcl::PointXYZ> cloud;
    require(!map->insert(cloud), "empty insertion reports added points");

    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
                cloud.emplace_back(0.0625f + x * 0.125f, 0.0625f + y * 0.125f, 0.0625f + z * 0.125f);
    const auto eighth = cloud.back();
    cloud.resize(cloud.size() - 1);
    require(map->insert(cloud), "first insertion failed");
    auto samples = read(map.get(), 7);
    require(samples.size() == 7, "unexpected initial point count");
    checkCovariances(samples);
    const auto* original = samples.front().address;
    require(!map->insert(cloud), "duplicate points were accepted");
    pcl::PointCloud<pcl::PointXYZ> extra;
    extra.push_back(eighth);
    require(map->insert(extra), "incremental insertion failed");
    samples = read(map.get(), 8);
    require(samples.size() == 8 && samples.front().address == original, "insertion invalidated a point pointer");
    checkCovariances(samples);
    require(samples.front().value.covariance_valid, "old points were not updated after adding a neighbor");

    extra.clear();
    extra.emplace_back(0.5625f, 0.0625f, 0.0625f);
    extra.emplace_back(3.99f, 0.0625f, 0.0625f);
    extra.emplace_back(4.01f, 0.0625f, 0.0625f);
    extra.emplace_back(-0.25f, -0.25f, -0.25f);
    extra.emplace_back(-16384.0f, 0.0f, 0.0f);
    extra.emplace_back(16383.875f, 0.0f, 0.0f);
    require(map->insert(extra), "new voxel insertion failed");
    samples = read(map.get(), 14);
    require(samples.size() == 14, "cross-voxel spacing or key boundaries discarded points");
    checkCovariances(samples);
    const std::vector<Eigen::Vector3f> queries{
        {0.0f, 0.0f, 0.0f}, {0.125f, 0.0625f, 0.0625f}, {20.0f, 20.0f, 20.0f},
        {4.0f, 0.0625f, 0.0625f}, {-16384.0f, 0.0f, 0.0f}, {16383.875f, 0.0f, 0.0f}};
    for (const int k : {0, 1, 12, 40}) checkQueries(map.get(), samples, queries, k, 100000.0f);
    checkQueries(map.get(), samples, {{0.125f, 0.0625f, 0.0625f}}, 1, 0.0625f);

    for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(), 16384.0f, -16385.0f})
    {
        extra.clear();
        extra.emplace_back(100.0f, 100.0f, 100.0f);
        extra.emplace_back(invalid, 0.0f, 0.0f);
        bool rejected = false;
        try { map->insert(extra); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && read(map.get(), 14).size() == 14, "invalid batch changed the map");
    }

    extra.clear();
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y)
            for (int z = 0; z < 4; ++z)
            {
                extra.emplace_back(8.03125f + x * 0.125f, 0.03125f + y * 0.125f, 0.03125f + z * 0.125f);
                extra.emplace_back(extra.back().x + 0.01f, extra.back().y, extra.back().z);
            }
    require(map->insert(extra), "capacity insertion failed");
    samples = read(map.get(), 50);
    require(samples.size() == 50, "per-voxel capacity or minimum spacing is incorrect");
    require(!map->insert(extra), "full voxel accepted more points");
    checkCovariances(samples);

    std::mt19937 random(17);
    std::uniform_real_distribution<float> coordinate(-2.0f, 2.0f);
    cloud.clear();
    for (int i = 0; i < 200; ++i) cloud.emplace_back(coordinate(random), coordinate(random), coordinate(random));
    require(map->insert(cloud), "random insertion failed");
    samples = read(map.get(), 250);
    std::vector<Eigen::Vector3f> random_queries;
    for (int i = 0; i < 50; ++i)
        random_queries.emplace_back(coordinate(random), coordinate(random), coordinate(random));
    for (const int k : {1, 12, 40}) checkQueries(map.get(), samples, random_queries, k, 0.5f);
    checkCovariances(samples);
}
}

int main()
{
    try
    {
        run();
        std::cout << "SparseVoxel GPU tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
