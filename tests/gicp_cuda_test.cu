#include "localization/GICP/GICP.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <Eigen/Cholesky>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t error)
{
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__global__ void snapshot(const SparseVoxel* map, PointWithCovariance* output, std::size_t* count)
{
    const auto points = map->points();
    if (threadIdx.x == 0) *count = points.size();
    for (std::size_t i = threadIdx.x; i < points.size(); i += blockDim.x) output[i] = *points[i];
}

auto key(const Eigen::Vector3f& point)
{
    return std::tuple{static_cast<int>(std::floor(point.x() / SparseVoxel::VOXEL_SIZE)),
                      static_cast<int>(std::floor(point.y() / SparseVoxel::VOXEL_SIZE)),
                      static_cast<int>(std::floor(point.z() / SparseVoxel::VOXEL_SIZE))};
}

std::vector<PointWithCovariance> points(const pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    auto map = std::make_unique<SparseVoxel>();
    map->insert(cloud);
    thrust::device_vector<PointWithCovariance> output(cloud.size());
    thrust::device_vector<std::size_t> count(1);
    snapshot<<<1, 128>>>(map.get(), output.data().get(), count.data().get());
    check(cudaDeviceSynchronize());
    const std::size_t size = count[0];
    const thrust::host_vector<PointWithCovariance> host(output.begin(), output.begin() + size);
    std::vector<PointWithCovariance> result(host.begin(), host.end());
    std::stable_sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs)
    {
        return key(lhs.position) < key(rhs.position);
    });
    return result;
}

GICP::Result reference(const std::vector<PointWithCovariance>& source,
                       const std::vector<PointWithCovariance>& target,
                       const Eigen::Isometry3d& initial_guess, const GICP::Config& config)
{
    GICP::Result result;
    result.num_source_points = source.size();
    result.num_target_points = target.size();
    result.transform = initial_guess;
    Sophus::SE3d pose(initial_guess.rotation(), initial_guess.translation());
    if (source.empty() || target.empty()) return result;
    for (int iteration = 0; iteration < config.max_iterations; ++iteration)
    {
        Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
        std::size_t count = 0;
        double squared_error = 0.0;
        const Eigen::Matrix3d rotation = pose.rotationMatrix();
        for (const auto& point : source)
        {
            const Eigen::Vector3d transformed = pose * point.position.cast<double>();
            const Eigen::Vector3f query = transformed.cast<float>();
            const auto [qx, qy, qz] = key(query);
            float best = config.max_correspondence_distance * config.max_correspondence_distance;
            const PointWithCovariance* nearest = nullptr;
            for (const auto& candidate : target)
            {
                const auto [x, y, z] = key(candidate.position);
                if (std::abs(x - qx) > 1 || std::abs(y - qy) > 1 || std::abs(z - qz) > 1) continue;
                const float distance = (candidate.position - query).squaredNorm();
                if (distance < best)
                {
                    nearest = &candidate;
                    best = distance;
                }
            }
            if (!nearest) continue;
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
            if (point.covariance_valid)
            {
                if (nearest->covariance_valid) covariance = nearest->covariance.cast<double>();
                covariance.noalias() += rotation * point.covariance.cast<double>() * rotation.transpose();
            }
            const Eigen::LDLT<Eigen::Matrix3d> decomposition(covariance);
            const Eigen::Vector3d residual = transformed - nearest->position.cast<double>();
            Eigen::Matrix<double, 3, 6> jacobian;
            jacobian.leftCols<3>().setIdentity();
            jacobian.rightCols<3>() = -Sophus::SO3d::hat(transformed);
            hessian.noalias() += jacobian.transpose() * decomposition.solve(jacobian);
            gradient.noalias() += jacobian.transpose() * decomposition.solve(residual);
            squared_error += residual.squaredNorm();
            ++count;
        }
        if (count == 0) break;
        const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(hessian);
        if (decomposition.info() != Eigen::Success) break;
        const Sophus::SE3d::Tangent increment = decomposition.solve(-gradient);
        if (!increment.allFinite()) break;
        pose = Sophus::SE3d::exp(increment) * pose;
        result.num_correspondences = count;
        result.fitness_score = squared_error / static_cast<double>(count);
        ++result.iterations;
        if (increment.head<3>().norm() < config.convergence_translation &&
            increment.tail<3>().norm() < config.convergence_rotation)
            break;
    }
    result.transform = Eigen::Isometry3d(pose.matrix());
    result.converged = result.iterations > 0 && std::isfinite(result.fitness_score);
    return result;
}

void compare(const GICP::Result& actual, const GICP::Result& expected)
{
    require(actual.num_source_points == expected.num_source_points, "source point count differs");
    require(actual.num_target_points == expected.num_target_points, "target point count differs");
    require(actual.num_correspondences == expected.num_correspondences, "correspondence count differs");
    require(actual.iterations == expected.iterations, "iteration count differs");
    require(actual.converged == expected.converged, "convergence differs");
    require(actual.transform.matrix().isApprox(expected.transform.matrix(), 2e-6), "pose differs from CPU reference");
    require(actual.fitness_score == expected.fitness_score ||
            std::abs(actual.fitness_score - expected.fitness_score) < 1e-8, "fitness differs from CPU reference");
}

void run()
{
    pcl::PointCloud<pcl::PointXYZ> target;
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j)
        {
            target.emplace_back(i * 0.25f, j * 0.25f, 0.0f);
            target.emplace_back(i * 0.25f, 0.0f, j * 0.25f);
            target.emplace_back(0.0f, i * 0.25f, j * 0.25f);
        }
    Sophus::SE3d::Tangent tangent;
    tangent << 0.035, -0.025, 0.045, 0.012, -0.008, 0.015;
    const Sophus::SE3d truth = Sophus::SE3d::exp(tangent);
    pcl::PointCloud<pcl::PointXYZ> source;
    for (const auto& point : target)
    {
        const Eigen::Vector3f transformed = (truth.inverse() * point.getVector3fMap().cast<double>()).cast<float>();
        source.emplace_back(transformed.x(), transformed.y(), transformed.z());
    }
    const auto source_points = points(source), target_points = points(target);
    GICP gicp;
    gicp.insertTargetPoints(target);
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const auto expected = reference(source_points, target_points, identity, gicp.config());
    const auto actual = gicp.align(source, identity);
    compare(actual, expected);
    require(actual.converged && actual.transform.matrix().isApprox(truth.matrix(), 2e-5),
            "alignment did not recover the known rigid transform");
    compare(gicp.align(target, identity), reference(target_points, target_points, identity, gicp.config()));

    GICP::Config config = gicp.config();
    config.max_iterations = 0;
    gicp.setConfig(config);
    compare(gicp.align(source, actual.transform),
            reference(source_points, target_points, actual.transform, config));
    config.max_iterations = 1;
    config.min_correspondences = 100000;
    config.max_fitness_score = 0.0;
    gicp.setConfig(config);
    compare(gicp.align(source, identity), reference(source_points, target_points, identity, config));
    config.max_iterations = 5;
    config.convergence_translation = 0.0;
    config.convergence_rotation = 0.0;
    gicp.setConfig(config);
    compare(gicp.align(source, identity), reference(source_points, target_points, identity, config));
    Eigen::Isometry3d far = identity;
    far.translation().x() = 50.0;
    compare(gicp.align(source, far), reference(source_points, target_points, far, config));

    bool rejected = false;
    try { gicp.align({}, identity); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "empty source was not rejected");
    Eigen::Isometry3d invalid = identity;
    invalid.translation().x() = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try { gicp.align(source, invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "nonfinite initial pose was not rejected");

    GICP empty;
    const auto no_target = empty.align(source, identity);
    require(!no_target.converged && no_target.num_target_points == 0 &&
            no_target.num_source_points == source_points.size() && no_target.iterations == 0,
            "empty target behavior changed");
    GICP moved = std::move(gicp);
    compare(moved.align(source, identity), reference(source_points, target_points, identity, config));
    empty = std::move(moved);
    compare(empty.align(source, identity), reference(source_points, target_points, identity, config));

    pcl::PointCloud<pcl::PointXYZ> sparse;
    sparse.emplace_back(0.0f, 0.0f, 0.0f);
    sparse.emplace_back(1.0f, 0.0f, 0.0f);
    sparse.emplace_back(0.0f, 1.0f, 0.0f);
    sparse.emplace_back(0.0f, 0.0f, 1.0f);
    const auto sparse_points = points(sparse);
    GICP unweighted;
    unweighted.insertTargetPoints(sparse);
    compare(unweighted.align(sparse, identity), reference(sparse_points, sparse_points, identity, unweighted.config()));
    GICP singular;
    pcl::PointCloud<pcl::PointXYZ> single;
    single.emplace_back(1.0f, 0.0f, 0.0f);
    singular.insertTargetPoints(single);
    const auto single_point = points(single);
    compare(singular.align(single, identity), reference(single_point, single_point, identity, singular.config()));
}
}

int main()
{
    try
    {
        run();
        std::cout << "GICP CUDA tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
