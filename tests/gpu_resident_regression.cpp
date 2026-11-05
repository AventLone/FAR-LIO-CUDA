#include "localization/GICP/GICP.h"
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

namespace
{
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using Key = std::array<int, 3>;

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

Key key(const Eigen::Vector3f& point)
{
    return {static_cast<int>(std::floor(point.x() / SparseVoxel::VOXEL_SIZE)),
            static_cast<int>(std::floor(point.y() / SparseVoxel::VOXEL_SIZE)),
            static_cast<int>(std::floor(point.z() / SparseVoxel::VOXEL_SIZE))};
}

struct Dataset
{
    Cloud target;
    Cloud source;
    Eigen::Isometry3d truth{Eigen::Isometry3d::Identity()};
};

Dataset dataset()
{
    Dataset data;
    std::mt19937 random(7);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    data.truth.linear() = Eigen::AngleAxisd(0.03, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    data.truth.translation() = Eigen::Vector3d(0.1, -0.04, 0.02);
    const Eigen::Isometry3d inverse = data.truth.inverse();
    for (int i = 0; i < 6000; ++i)
    {
        Eigen::Vector3f point(4.0f * unit(random), 2.0f * unit(random), 1.0f + 2.0f * unit(random));
        switch (i % 5)
        {
        case 0: point.z() = -1.0f; break;
        case 1: point.z() = 3.0f; break;
        case 2: point.y() = -2.0f; break;
        case 3: point.y() = 2.0f; break;
        default: point.z() = 0.2f * std::sin(point.x()) + 0.1f * point.y(); break;
        }
        data.target.emplace_back(point.x(), point.y(), point.z());
        const Eigen::Vector3d transformed = inverse * point.cast<double>();
        data.source.emplace_back(transformed.x(), transformed.y(), transformed.z());
    }
    return data;
}

std::vector<SparseVoxel::Neighbor> bruteNeighbors(const SparseVoxel& map, const Eigen::Vector3f& query)
{
    const Key query_key = key(query);
    auto points = map.points();
    std::stable_sort(points.begin(), points.end(), [](const auto* lhs, const auto* rhs)
    {
        return key(lhs->position) < key(rhs->position);
    });
    std::vector<SparseVoxel::Neighbor> result;
    for (const auto* point : points)
    {
        const Key point_key = key(point->position);
        if (std::abs(point_key[0] - query_key[0]) <= 1 && std::abs(point_key[1] - query_key[1]) <= 1 &&
            std::abs(point_key[2] - query_key[2]) <= 1)
            result.push_back({point, (query - point->position).squaredNorm()});
    }
    std::stable_sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs)
    {
        return lhs.squared_distance < rhs.squared_distance;
    });
    return result;
}

void checkVoxels(const Dataset& data)
{
    SparseVoxel map;
    require(!map.insert(Cloud{}), "Empty insert changed");
    require(map.insert(data.target), "Voxel insertion failed");
    require(map.pointCount() == 3970, "Voxel filtering changed the baseline retained point count");
    const auto points = map.points();
    std::vector<Eigen::Vector3f> positions;
    for (const auto* point : points) positions.push_back(point->position);
    const auto neighborhoods = map.nearestNeighborsBatch(positions, SparseVoxel::MAX_COVARIANCE_NEIGHBORS);
    require(neighborhoods.size() == points.size(), "kNN batch shape changed");
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto& neighbors = neighborhoods[i];
        if (neighbors.size() < static_cast<std::size_t>(SparseVoxel::MIN_COVARIANCE_NEIGHBORS))
        {
            require(!points[i]->covariance_valid && points[i]->covariance.isIdentity(), "Sparse covariance fallback changed");
            continue;
        }
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (const auto& neighbor : neighbors) mean += neighbor.point->position;
        mean /= static_cast<float>(neighbors.size());
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const auto& neighbor : neighbors)
        {
            const Eigen::Vector3f offset = neighbor.point->position - mean;
            covariance.noalias() += offset * offset.transpose();
        }
        covariance /= static_cast<float>(neighbors.size() - 1);
        covariance.diagonal().array() += 1e-6f;
        const Eigen::Matrix3f expected = covariance * covariance.inverse().norm();
        require(points[i]->covariance_valid && points[i]->covariance.isApprox(expected, 2e-5f),
                "Device covariance differs from the original CPU formula");
    }
    for (std::size_t i = 0; i < data.source.size(); i += 83)
    {
        const Eigen::Vector3f query = data.source[i].getVector3fMap();
        const auto expected = bruteNeighbors(map, query);
        for (const float radius : {0.0f, 0.1f, 0.5f, 1.5f})
        {
            const auto actual = map.nearestNeighbor(query, radius);
            const bool found = !expected.empty() && expected.front().squared_distance < radius * radius;
            require(actual.point == (found ? expected.front().point : nullptr), "GPU nearest point or tie ordering changed");
            require(actual.squared_distance == (found ? expected.front().squared_distance : radius * radius),
                    "GPU nearest squared distance changed");
        }
    }
    const auto* first = points.front();
    const Eigen::Vector3f original = first->position;
    require(map.insert(data.target) && map.nearestNeighbor(original, 0.01f).point == first,
            "Incremental insertion invalidated an exposed point pointer");
    SparseVoxel copy(map);
    map.clear();
    require(copy.pointCount() == points.size() && copy.nearestNeighbor(original, 0.01f).point != nullptr,
            "GPU map copies must own independent storage");
    Cloud invalid;
    invalid.emplace_back(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f);
    const auto count = copy.pointCount();
    requireThrows<std::invalid_argument>([&] { copy.insert(invalid); }, "Nonfinite map input must be rejected");
    require(copy.pointCount() == count && copy.insert(data.target), "Rejected input modified or poisoned the voxel map");
    std::vector<std::future<void>> readers;
    for (int thread = 0; thread < 4; ++thread)
        readers.push_back(std::async(std::launch::async, [&]
        {
            for (int i = 0; i < 12; ++i)
                require(copy.nearestNeighbor(original, 0.01f).point != nullptr && copy.pointCount() == count,
                        "Concurrent read-only voxel access changed the map");
        }));
    for (int i = 0; i < 1000; ++i) require(copy.pointCount() == count, "Read-only point counts changed");
    for (auto& reader : readers) reader.get();
    Cloud first_part, remainder;
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 3; ++y)
            (first_part.size() < 7 ? first_part : remainder).emplace_back(0.11f * x, 0.11f * y, 0.0f);
    SparseVoxel incremental;
    incremental.insert(first_part);
    const auto* old_point = incremental.points().front();
    require(!old_point->covariance_valid, "Seven neighbors must not produce a valid covariance");
    incremental.insert(remainder);
    require(old_point->covariance_valid, "An exposed old point was not refreshed after asynchronous insertion");
    SparseVoxel single;
    Cloud one;
    one.emplace_back(1.0f, 0.0f, 0.0f);
    single.insert(one);
    require(single.nearestNeighbor(Eigen::Vector3f(1.0f, 0.0f, 0.0f), 0.1f).point != nullptr,
            "Single-item radix sorting lost a nonzero voxel key");
    Cloud crowded;
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y)
            for (int z = 0; z < 4; ++z)
                crowded.emplace_back(0.02f + 0.11f * x, 0.02f + 0.11f * y, 0.02f + 0.11f * z);
    SparseVoxel capped;
    capped.insert(crowded);
    require(capped.voxelCount() == 1 && capped.pointCount() == SparseVoxel::MAX_POINTS_PER_VOXEL,
            "The 36-point voxel capacity changed");
    const auto accepted = capped.points();
    for (std::size_t i = 0; i < accepted.size(); ++i)
        require(accepted[i]->position == crowded[i].getVector3fMap(), "Stable voxel insertion order changed");
    SparseVoxel spacing;
    Cloud distances;
    for (const float x : {0.0f, 0.1f, 0.099f, 0.2f}) distances.emplace_back(x, 0.0f, 0.0f);
    spacing.insert(distances);
    require(spacing.pointCount() == 3, "The strict 0.1 m same-voxel spacing gate changed");
}

GICP::Result cpuReference(const Dataset& data, const GICP::Config& config)
{
    SparseVoxel source;
    SparseVoxel target;
    source.insert(data.source);
    target.insert(data.target);
    const auto source_points = source.points();
    std::map<Key, std::vector<const PointWithCovariance*>> cells;
    for (const auto* point : target.points()) cells[key(point->position)].push_back(point);
    GICP::Result result;
    result.num_source_points = source.pointCount();
    result.num_target_points = target.pointCount();
    Sophus::SE3d pose;
    for (int iteration = 0; iteration < config.max_iterations; ++iteration)
    {
        Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
        double error = 0.0;
        std::size_t count = 0;
        for (const auto* point : source_points)
        {
            const Eigen::Vector3d transformed = pose * point->position.cast<double>();
            const Eigen::Vector3f query = transformed.cast<float>();
            const Key query_key = key(query);
            float distance = config.max_correspondence_distance * config.max_correspondence_distance;
            const PointWithCovariance* match = nullptr;
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz)
                    {
                        const auto found = cells.find({query_key[0] + dx, query_key[1] + dy, query_key[2] + dz});
                        if (found == cells.end()) continue;
                        for (const auto* candidate : found->second)
                        {
                            const float candidate_distance = (query - candidate->position).squaredNorm();
                            if (candidate_distance < distance)
                            {
                                distance = candidate_distance;
                                match = candidate;
                            }
                        }
                    }
            if (match == nullptr) continue;
            const Eigen::Matrix3d rotation = pose.rotationMatrix();
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
            if (point->covariance_valid)
            {
                if (match->covariance_valid) covariance = match->covariance.cast<double>();
                covariance.noalias() += rotation * point->covariance.cast<double>() * rotation.transpose();
            }
            const Eigen::LDLT<Eigen::Matrix3d> factor(covariance);
            if (factor.info() != Eigen::Success) continue;
            const Eigen::Vector3d residual = transformed - match->position.cast<double>();
            const Eigen::Vector3d precision_residual = factor.solve(residual);
            if (!precision_residual.allFinite()) continue;
            Eigen::Matrix<double, 3, 6> jacobian;
            jacobian.leftCols<3>().setIdentity();
            jacobian.rightCols<3>() = -Sophus::SO3d::hat(transformed);
            const Eigen::Matrix<double, 3, 6> precision_jacobian = factor.solve(jacobian);
            hessian.noalias() += jacobian.transpose() * precision_jacobian;
            gradient.noalias() += jacobian.transpose() * precision_residual;
            error += residual.squaredNorm();
            ++count;
        }
        if (count == 0) break;
        const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> factor(hessian);
        if (factor.info() != Eigen::Success) break;
        const Sophus::SE3d::Tangent increment = factor.solve(-gradient);
        if (!increment.allFinite()) break;
        pose = Sophus::SE3d::exp(increment) * pose;
        result.num_correspondences = count;
        result.fitness_score = error / static_cast<double>(count);
        ++result.iterations;
        if (increment.head<3>().norm() < config.convergence_translation &&
            increment.tail<3>().norm() < config.convergence_rotation) break;
    }
    result.transform = Eigen::Isometry3d(pose.matrix());
    result.converged = result.iterations > 0 && std::isfinite(result.fitness_score);
    return result;
}

void checkRegistration(const Dataset& data)
{
    GICP::Config config;
    config.max_iterations = 40;
    const auto expected = cpuReference(data, config);
    GICP registration(config);
    registration.insertTargetPoints(data.target);
    for (int repeat = 0; repeat < 3; ++repeat)
    {
        const auto actual = registration.align(data.source, Eigen::Isometry3d::Identity());
        require(actual.converged && actual.iterations == expected.iterations &&
                actual.num_source_points == expected.num_source_points &&
                actual.num_target_points == expected.num_target_points &&
                actual.num_correspondences == expected.num_correspondences, "GPU iteration/count results differ from the CPU solver");
        require(actual.transform.matrix().isApprox(expected.transform.matrix(), 1e-5) &&
                std::abs(actual.fitness_score - expected.fitness_score) < 1e-8,
                "GPU LDLT/SE(3) result differs from the independent CPU solver");
        require((actual.transform.translation() - data.truth.translation()).norm() < 0.005 &&
                Eigen::AngleAxisd(actual.transform.rotation().transpose() * data.truth.rotation()).angle() < 0.002,
                "Synthetic registration exceeded 5 mm or 0.002 rad");
    }
    config.max_iterations = 0;
    registration.setConfig(config);
    const auto zero = registration.align(data.source, data.truth);
    require(zero.iterations == 0 && !zero.converged && zero.transform.matrix().isApprox(data.truth.matrix(), 1e-12),
            "Zero-iteration registration changed the initial pose");
    GICP empty;
    const auto absent = empty.align(data.source, data.truth);
    require(!absent.converged && absent.num_target_points == 0 && absent.transform.matrix().isApprox(data.truth.matrix(), 1e-12),
            "Empty-target registration changed behavior");
    requireThrows<std::invalid_argument>([&] { empty.align(Cloud{}, data.truth); }, "Empty source must be rejected");
    Cloud invalid = data.source;
    invalid[3].x = std::numeric_limits<float>::infinity();
    requireThrows<std::invalid_argument>([&] { empty.align(invalid, data.truth); }, "Nonfinite source must be rejected");
    GICP degenerate;
    Cloud one;
    one.emplace_back(1.0f, 0.0f, 0.0f);
    degenerate.insertTargetPoints(one);
    const auto rank_deficient = degenerate.align(one, Eigen::Isometry3d::Identity());
    require(rank_deficient.converged && rank_deficient.iterations == 1 &&
            rank_deficient.transform.matrix().isIdentity(1e-12), "Rank-deficient LDLT zero-step handling changed");
}

Cloud voxelGrid(const Cloud& input, float size)
{
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setLeafSize(size, size, size);
    filter.setInputCloud(input.makeShared());
    Cloud result;
    filter.filter(result);
    return result;
}

void compareClouds(Cloud actual, Cloud expected, const std::string& message)
{
    const auto less = [](const pcl::PointXYZ& lhs, const pcl::PointXYZ& rhs)
    {
        return std::array<float, 3>{lhs.x, lhs.y, lhs.z} < std::array<float, 3>{rhs.x, rhs.y, rhs.z};
    };
    std::sort(actual.begin(), actual.end(), less);
    std::sort(expected.begin(), expected.end(), less);
    require(actual.size() == expected.size(), message + ": wrong point count");
    for (std::size_t i = 0; i < actual.size(); ++i)
        require((actual[i].getVector3fMap() - expected[i].getVector3fMap()).norm() < 2e-5f, message + ": wrong centroid/pose");
}

void checkPreprocessing()
{
    GICP::PreprocessConfig config;
    config.intensity_threshold = 0.5f;
    config.footprint_x = config.footprint_y = 0.8f;
    config.sensor_to_output.linear() = Eigen::AngleAxisf(0.1f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    config.sensor_to_output.translation() = Eigen::Vector3f(0.02f, -0.01f, 0.03f);
    pcl::PointCloud<pcl::PointXYZI> raw;
    Cloud filtered;
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 3; ++j)
        {
            pcl::PointXYZI point;
            point.x = 1.0f + 0.12f * i + 0.005f * j;
            point.y = -0.8f + 0.003f * j;
            point.z = 0.2f;
            point.intensity = 1.0f;
            raw.push_back(point);
            filtered.emplace_back(point.x, point.y, point.z);
        }
    pcl::PointXYZI rejected;
    rejected.x = rejected.y = rejected.z = 0.8f;
    rejected.intensity = 1.0f;
    raw.push_back(rejected);
    rejected.x = 4.0f;
    rejected.intensity = 0.1f;
    raw.push_back(rejected);
    rejected.x = std::numeric_limits<float>::quiet_NaN();
    raw.push_back(rejected);
    rejected.x = 4.0f;
    rejected.intensity = std::numeric_limits<float>::infinity();
    raw.push_back(rejected);
    Cloud expected = voxelGrid(filtered, config.voxel_size);
    pcl::transformPointCloud(expected, expected, config.sensor_to_output);
    GICP registration;
    registration.prepareSource(pcl::PointCloud<pcl::PointXYZI>{}, config);
    require(!registration.initializePreparedTarget(Eigen::Isometry3d::Identity(), 0),
            "An empty frame must not report successful initialization");
    registration.prepareSource(raw, config);
    require(registration.initializePreparedTarget(Eigen::Isometry3d::Identity(), 1), "Prepared map initialization failed");
    const auto first = registration.downloadPrepared(Eigen::Isometry3d::Identity(), true, true);
    require(first.num_invalid_points == 2 && first.num_processed_points == expected.size(),
            "GPU preprocessing did not report invalid/filter counts");
    compareClouds(first.scan_in_map, expected, "GPU filtered scan differs from PCL");
    compareClouds(first.map, expected, "Initial visualization map changed");
    registration.insertPreparedTarget(Eigen::Isometry3d::Identity());
    const auto second = registration.downloadPrepared(Eigen::Isometry3d::Identity(), true, true);
    Cloud merged = expected;
    merged += expected;
    compareClouds(second.map, voxelGrid(merged, config.map_voxel_size), "GPU visualization map merge differs from PCL");
    pcl::PointCloud<pcl::PointXYZI> no_points;
    const auto empty = registration.processFrame(no_points, Eigen::Isometry3d::Identity(), config, true, true);
    require(empty.num_processed_points == 0 && !empty.map_updated && empty.scan_in_map.empty(), "Empty frame was not skipped");
    compareClouds(empty.map, second.map, "An empty frame modified the map");
}

void checkDeskew()
{
    StampedCloud cloud{0.0, 1.0, {}};
    const std::vector<TimedPose> trajectory{
        {0.0, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()},
        {1.0, Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ())), Eigen::Vector3d(0.1, 0.0, 0.0)}};
    Eigen::Isometry3d extrinsic = Eigen::Isometry3d::Identity();
    extrinsic.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()).toRotationMatrix();
    extrinsic.translation() = Eigen::Vector3d(0.2, -0.1, 0.05);
    Eigen::Isometry3d end = Eigen::Isometry3d::Identity();
    end.linear() = trajectory.back().rotation.toRotationMatrix();
    end.translation() = trajectory.back().translation;
    const Eigen::Isometry3d end_inverse = (end * extrinsic).inverse();
    Cloud expected;
    int index = 0;
    for (const double offset : {-0.5, 0.0, 0.25, 0.75, 1.0, 1.5})
    {
        const Eigen::Vector3f point(1.0f + 0.3f * index++, 0.1f, 0.2f);
        cloud.points.push_back({point, 1.0f, offset});
        const double alpha = std::clamp(offset, 0.0, 1.0);
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.linear() = Eigen::AngleAxisd(0.2 * alpha, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        pose.translation() = Eigen::Vector3d(0.1 * alpha, 0.0, 0.0);
        const Eigen::Vector3d deskewed = end_inverse * pose * extrinsic * point.cast<double>();
        expected.emplace_back(deskewed.x(), deskewed.y(), deskewed.z());
    }
    GICP::PreprocessConfig config;
    config.voxel_size = 0.02f;
    config.sensor_to_output = extrinsic.cast<float>();
    expected = voxelGrid(expected, config.voxel_size);
    pcl::transformPointCloud(expected, expected, config.sensor_to_output);
    GICP registration;
    registration.prepareSource(cloud, config, trajectory, extrinsic);
    require(registration.initializePreparedTarget(Eigen::Isometry3d::Identity(), 1), "Deskewed map initialization failed");
    const auto result = registration.downloadPrepared(Eigen::Isometry3d::Identity(), true, false);
    compareClouds(result.scan_in_map, expected, "Device deskew differs from the CPU motion model");
    const std::vector<TimedPose> extended{
        {-1.0, Eigen::Quaterniond(Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitZ())), Eigen::Vector3d(-0.1, 0.0, 0.0)},
        {2.0, Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ())), Eigen::Vector3d(0.2, 0.0, 0.0)}};
    GICP wider;
    wider.prepareSource(cloud, config, extended, extrinsic);
    wider.initializePreparedTarget(Eigen::Isometry3d::Identity(), 1);
    compareClouds(wider.downloadPrepared(Eigen::Isometry3d::Identity(), true, false).scan_in_map, expected,
                  "Deskew must interpolate the pose at scan end, not use the final trajectory sample");
}

pcl::PointCloud<pcl::PointXYZI> withIntensity(const Cloud& input)
{
    pcl::PointCloud<pcl::PointXYZI> result;
    for (const auto& point : input)
    {
        pcl::PointXYZI value;
        value.x = point.x;
        value.y = point.y;
        value.z = point.z;
        value.intensity = 1.0f;
        result.push_back(value);
    }
    return result;
}

void profile(const Dataset& data, bool full_frame)
{
    GICP::Config config;
    config.max_iterations = 40;
    GICP registration(config);
    GICP::PreprocessConfig preprocessing;
    const auto source = withIntensity(data.source);
    const auto target = withIntensity(data.target);
    if (full_frame) registration.processFrame(target, Eigen::Isometry3d::Identity(), preprocessing);
    else registration.insertTargetPoints(data.target);
    for (int i = 0; i < 2; ++i)
    {
        if (full_frame) registration.processFrame(source, Eigen::Isometry3d::Identity(), preprocessing);
        else registration.align(data.source, Eigen::Isometry3d::Identity());
    }
    require(cudaProfilerStart() == cudaSuccess, "Unable to start profiling");
    const auto start = std::chrono::steady_clock::now();
    const auto result = full_frame
        ? registration.processFrame(source, Eigen::Isometry3d::Identity(), preprocessing).registration
        : registration.align(data.source, Eigen::Isometry3d::Identity());
    const auto end = std::chrono::steady_clock::now();
    require(cudaProfilerStop() == cudaSuccess, "Unable to stop profiling");
    require(result.converged, "Profiled registration did not converge");
    std::cout << std::setprecision(17) << "iterations=" << result.iterations
              << " source=" << result.num_source_points << " target=" << result.num_target_points
              << " correspondences=" << result.num_correspondences << " fitness=" << result.fitness_score
              << " frame_ms=" << std::chrono::duration<double, std::milli>(end - start).count()
              << '\n' << result.transform.matrix() << '\n';
}
}

int main(int argc, char** argv)
{
    try
    {
        const Dataset data = dataset();
        if (argc == 2)
        {
            const std::string option = argv[1];
            require(option == "--profile" || option == "--profile-frame", "Unknown profiling option");
            profile(data, option == "--profile-frame");
        }
        else
        {
            require(argc == 1, "Usage: gpu_resident_regression [--profile|--profile-frame]");
            checkVoxels(data);
            checkRegistration(data);
            checkPreprocessing();
            checkDeskew();
            std::cout << "Device-resident voxel, GICP, preprocessing, map, and deskew checks passed\n";
        }
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "GPU resident regression failed: " << exception.what() << '\n';
        return 1;
    }
}
