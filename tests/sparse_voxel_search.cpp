#include "localization/types/SparseVoxel.h"
#include <Eigen/SVD>
#include <pcl/search/kdtree.h>
#include <tbb/global_control.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace
{
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using Key = std::array<int, 3>;
using Points = std::vector<const PointWithCovariance*>;
using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

Key key(const Eigen::Vector3f& point)
{
    const Eigen::Vector3f scaled = point / SparseVoxel::VOXEL_SIZE;
    return {static_cast<int>(std::floor(scaled.x())), static_cast<int>(std::floor(scaled.y())),
            static_cast<int>(std::floor(scaled.z()))};
}

Cloud makeCloud(std::size_t count, std::uint32_t seed)
{
    Cloud result;
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.005f);
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        float x = 4.0f * unit(random);
        float y = 1.3f * unit(random);
        float z = 1.4f * unit(random);
        switch (i % 5)
        {
        case 0: z = -1.4f; break;
        case 1: z = 1.4f; break;
        case 2: y = -1.3f; break;
        case 3: y = 1.3f; break;
        default: z = 0.25f * std::sin(1.7f * x) + 0.1f * y; break;
        }
        result.emplace_back(x + noise(random), y + noise(random), z + noise(random));
    }
    return result;
}

Cloud::Ptr retained(const SparseVoxel& index)
{
    Cloud::Ptr result(new Cloud);
    result->reserve(index.pointCount());
    for (const auto* point : index.points())
    {
        result->emplace_back(point->position.x(), point->position.y(), point->position.z());
    }
    return result;
}

std::vector<SparseVoxel::Neighbor> bruteNeighbors(const Points& points, const Eigen::Vector3f& query)
{
    const Key query_key = key(query);
    std::vector<SparseVoxel::Neighbor> result;
    for (const auto* point : points)
    {
        const Key point_key = key(point->position);
        if (std::abs(point_key[0] - query_key[0]) <= 1 &&
            std::abs(point_key[1] - query_key[1]) <= 1 &&
            std::abs(point_key[2] - query_key[2]) <= 1)
        {
            result.push_back({point, (query - point->position).squaredNorm()});
        }
    }
    std::stable_sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs)
    {
        return lhs.squared_distance < rhs.squared_distance;
    });
    return result;
}

void checkQueries(const SparseVoxel& index, const Cloud& queries)
{
    Points points = index.points();
    std::stable_sort(points.begin(), points.end(), [](const auto* lhs, const auto* rhs)
    {
        return key(lhs->position) < key(rhs->position);
    });
    for (const auto& query_point : queries)
    {
        const Eigen::Vector3f query = query_point.getVector3fMap();
        const auto expected = bruteNeighbors(points, query);
        for (const float radius : {0.0f, 0.001f, 0.1f, 0.125f, 0.5f, 1.5f})
        {
            const auto actual = index.nearestNeighbor(query, radius);
            const bool found = !expected.empty() && expected.front().squared_distance < radius * radius;
            require(actual.point == (found ? expected.front().point : nullptr),
                    "1-NN differs from local brute force, including its original tie order");
            require(actual.squared_distance == (found ? expected.front().squared_distance : radius * radius),
                    "1-NN distance or strict distance gate changed");
        }
        for (const int count : {-1, 0, 1, 2, 8, 20, 36, 1000})
        {
            auto actual = index.nearestNeighbors(query, count);
            const std::size_t expected_count = count <= 0 ? 0
                : std::min(static_cast<std::size_t>(count), expected.size());
            require(actual.size() == expected_count,
                    "Local kNN returned the wrong number of neighbors");
            std::sort(actual.begin(), actual.end(), [](const auto& lhs, const auto& rhs)
            {
                return lhs.squared_distance < rhs.squared_distance;
            });
            for (std::size_t i = 0; i < actual.size(); ++i)
            {
                require(actual[i].squared_distance == expected[i].squared_distance,
                        "Local kNN distances differ from brute force");
            }
            std::sort(actual.begin(), actual.end(), [](const auto& lhs, const auto& rhs)
            {
                return std::less<const PointWithCovariance*>{}(lhs.point, rhs.point);
            });
            require(std::adjacent_find(actual.begin(), actual.end(), [](const auto& lhs, const auto& rhs)
            {
                return lhs.point == rhs.point;
            }) == actual.end(), "Local kNN returned a point twice");
        }
    }
}

void checkCovariances(const SparseVoxel& index)
{
    for (const auto* point : index.points())
    {
        const auto neighbors = index.nearestNeighbors(point->position, SparseVoxel::MAX_COVARIANCE_NEIGHBORS);
        if (neighbors.size() < static_cast<std::size_t>(SparseVoxel::MIN_COVARIANCE_NEIGHBORS))
        {
            require(!point->covariance_valid && point->covariance.isIdentity(), "Sparse fallback changed");
            continue;
        }
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (const auto& neighbor : neighbors)
        {
            mean += neighbor.point->position;
        }
        mean /= static_cast<float>(neighbors.size());
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const auto& neighbor : neighbors)
        {
            const Eigen::Vector3f offset = neighbor.point->position - mean;
            covariance.noalias() += offset * offset.transpose();
        }
        covariance /= static_cast<float>(neighbors.size() - 1);
        const Eigen::JacobiSVD<Eigen::Matrix3f> decomposition(covariance, Eigen::ComputeFullU);
        const Eigen::Matrix3f expected = decomposition.matrixU() *
            Eigen::Vector3f(1.0f, 1.0f, 0.001f).asDiagonal() * decomposition.matrixU().transpose();
        require(point->covariance_valid && point->covariance.isApprox(expected, 1e-6f),
                "Shared/incremental covariance differs from an independent point query");
    }
}

void regression()
{
    SparseVoxel empty;
    Cloud queries;
    for (const float boundary : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f})
    {
        for (const float x : {std::nextafter(boundary, -INFINITY), boundary,
                              std::nextafter(boundary, INFINITY)})
        {
            queries.emplace_back(x, boundary, -boundary);
            queries.emplace_back(boundary, x, -boundary);
            queries.emplace_back(boundary, -boundary, x);
        }
    }
    checkQueries(empty, queries);
    SparseVoxel faces;
    require(faces.insert(queries), "Voxel face fixture insertion failed");
    checkQueries(faces, queries);
    Cloud line;
    for (int i = 0; i < SparseVoxel::MIN_COVARIANCE_NEIGHBORS; ++i)
    {
        line.emplace_back(0.11f * (static_cast<float>(i) -
            0.5f * static_cast<float>(SparseVoxel::MIN_COVARIANCE_NEIGHBORS - 1)), 0.0f, 0.0f);
    }
    SparseVoxel collinear;
    require(collinear.insert(line) && collinear.pointCount() == line.size(), "Collinear fixture insertion failed");
    for (const auto* point : collinear.points())
    {
        require(point->covariance_valid, "Regularization must support a rank-deficient collinear neighborhood");
    }
    checkCovariances(collinear);
    Cloud ties;
    ties.emplace_back(0.125f, 0.0f, 0.0f);
    ties.emplace_back(-0.125f, 0.0f, 0.0f);
    ties.emplace_back(0.0f, -0.125f, 0.0f);
    ties.emplace_back(0.0f, 0.0f, -0.125f);
    SparseVoxel tied;
    require(tied.insert(ties), "Tie fixture insertion failed");
    Cloud origin;
    origin.emplace_back(0.0f, 0.0f, 0.0f);
    checkQueries(tied, origin);
    const Cloud cloud = makeCloud(1800, 321);
    Cloud random_queries = makeCloud(160, 654);
    queries += random_queries;
    SparseVoxel index;
    Cloud first;
    Cloud remaining;
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        (i < cloud.size() / 2 ? first : remaining).push_back(cloud[i]);
    }
    require(index.insert(first), "Initial insertion failed");
    checkQueries(index, queries);
    checkCovariances(index);
    require(index.insert(remaining), "Incremental insertion failed");
    checkQueries(index, queries);
    checkCovariances(index);
    for (const auto& point : queries)
    {
        origin.emplace_back(point.x + 65536.0f, point.y - 65536.0f, point.z + 32768.0f);
    }
    Cloud shifted;
    for (const auto& point : cloud)
    {
        shifted.emplace_back(point.x + 65536.0f, point.y - 65536.0f, point.z + 32768.0f);
    }
    index.clear();
    require(index.empty() && index.insert(shifted), "Clear/reinsert failed");
    checkQueries(index, origin);
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values.at(values.size() / 2);
}

void benchmark()
{
    std::cout << "raw_points,used_points,insert_ms,voxel_1nn_us,pcl_1nn_us,voxel_knn_us,pcl_knn_us,k\n";
    for (const std::size_t count : {3000U, 12000U, 40000U})
    {
        const Cloud cloud = makeCloud(count, 1000);
        std::vector<double> insert_times;
        SparseVoxel index;
        for (int trial = -1; trial < 7; ++trial)
        {
            index.clear();
            const auto start = Clock::now();
            require(index.insert(cloud), "Benchmark insertion failed");
            const auto end = Clock::now();
            if (trial >= 0)
            {
                insert_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }
        }
        const auto target = retained(index);
        SparseVoxel source;
        require(source.insert(makeCloud(index.pointCount(), 1001)), "Query insertion failed");
        const auto independent = retained(source);
        pcl::search::KdTree<pcl::PointXYZ> tree;
        tree.setInputCloud(target);
        std::array<double, 4> timings{};
        double checksum = 0.0;
        for (int kind = 0; kind < 2; ++kind)
        {
            const int k = kind == 0 ? 1 : SparseVoxel::MAX_COVARIANCE_NEIGHBORS;
            const Cloud& queries = kind == 0 ? *independent : *target;
            pcl::Indices indices(static_cast<std::size_t>(k));
            std::vector<float> distances(static_cast<std::size_t>(k));
            std::array<std::vector<double>, 2> samples;
            for (int trial = -1; trial < 7; ++trial)
            {
                for (int position = 0; position < 2; ++position)
                {
                    const int method = (position + trial + 2) % 2;
                    const auto start = Clock::now();
                    for (int pass = 0; pass < 4; ++pass)
                    {
                        for (const auto& query : queries)
                        {
                            if (method == 0 && kind == 0)
                            {
                                const auto nearest = index.nearestNeighbor(query.getVector3fMap(), 0.5f);
                                checksum += nearest.squared_distance;
                            }
                            else if (method == 0)
                            {
                                const auto neighbors = index.nearestNeighbors(query.getVector3fMap(), k);
                                for (const auto& neighbor : neighbors)
                                {
                                    checksum += neighbor.squared_distance;
                                }
                            }
                            else
                            {
                                require(tree.nearestKSearch(query, k, indices, distances) == k, "PCL search failed");
                                checksum += kind == 0 ? std::min(distances[0], 0.25f)
                                    : std::accumulate(distances.begin(), distances.end(), 0.0);
                            }
                        }
                    }
                    const auto end = Clock::now();
                    if (trial >= 0)
                    {
                        samples[method].push_back(std::chrono::duration<double, std::micro>(end - start).count() /
                                                  (4.0 * static_cast<double>(queries.size())));
                    }
                }
            }
            timings[2 * kind] = median(samples[0]);
            timings[2 * kind + 1] = median(samples[1]);
        }
        require(std::isfinite(checksum) && checksum > 0.0, "Invalid benchmark checksum");
        std::cout << count << ',' << index.pointCount() << ',' << median(insert_times);
        for (const double time : timings)
        {
            std::cout << ',' << time;
        }
        std::cout << ',' << SparseVoxel::MAX_COVARIANCE_NEIGHBORS << std::endl;
    }
}
}

int main(int argc, char** argv)
{
    try
    {
        const bool measure = argc == 2 && std::string(argv[1]) == "--benchmark";
        require(argc == 1 || measure, "Usage: sparse_voxel_search [--benchmark]");
        if (measure)
        {
            tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);
            std::cout << std::fixed << std::setprecision(6);
            benchmark();
        }
        else
        {
            for (const std::size_t threads : {1U, 8U})
            {
                tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
                regression();
            }
            std::cout << "SparseVoxel local search and shared covariance checks passed\n";
        }
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "SparseVoxel search check failed: " << exception.what() << '\n';
        return 1;
    }
}
