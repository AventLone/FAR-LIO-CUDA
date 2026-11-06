#include "localization/GICP/GICP.h"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

namespace
{
void check(const cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

struct Pose
{
    double rotation[9];
    double translation[3];
};

struct Equation
{
    double hessian[21]{};
    double gradient[6]{};
    double squared_error{0.0};
    std::size_t count{0};
    std::size_t error{0};
};

struct Sum
{
    __host__ __device__ Equation operator()(Equation lhs, const Equation& rhs) const
    {
        for (int i = 0; i < 21; ++i) lhs.hessian[i] += rhs.hessian[i];
        for (int i = 0; i < 6; ++i) lhs.gradient[i] += rhs.gradient[i];
        lhs.squared_error += rhs.squared_error;
        lhs.count += rhs.count;
        lhs.error |= rhs.error;
        return lhs;
    }
};

__global__ void countPoints(const SparseVoxel* source, const SparseVoxel* target, std::size_t* counts)
{
    counts[0] = source->points().size();
    counts[1] = target->points().size();
}

__device__ Equation contribute(const PointWithCovariance& source, const SparseVoxel* target,
                               const Pose& pose, const float max_distance)
{
    Equation equation;
    const Eigen::Matrix3d rotation = Eigen::Map<const Eigen::Matrix3d>(pose.rotation);
    const Eigen::Vector3d position = rotation * source.position.cast<double>() +
                                    Eigen::Map<const Eigen::Vector3d>(pose.translation);
    const Eigen::Vector3f query = position.cast<float>();
    for (int axis = 0; axis < 3; ++axis)
        if (!(query[axis] >= -16384.0f && query[axis] < 16384.0f))
        {
            equation.error = 1;
            return equation;
        }
    const auto [point, squared_distance] = target->nearestNeighbor(query, max_distance);
    if (!point) return equation;

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
    if (source.covariance_valid)
    {
        if (point->covariance_valid) covariance = point->covariance.cast<double>();
        covariance.noalias() += rotation * source.covariance.cast<double>() * rotation.transpose();
    }
    const Eigen::Matrix3d precision = covariance.inverse();
    for (int i = 0; i < 9; ++i)
    {
        if (!isfinite(precision.data()[i]))
        {
            equation.error = 1;
            return equation;
        }
    }

    const Eigen::Vector3d residual = position - point->position.cast<double>();
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.leftCols<3>().setIdentity();
    jacobian.rightCols<3>() = -Sophus::SO3d::hat(position);
    const Eigen::Matrix<double, 6, 6> hessian = jacobian.transpose() * precision * jacobian;
    const Eigen::Matrix<double, 6, 1> gradient = jacobian.transpose() * precision * residual;
    int index = 0;
    for (int row = 0; row < 6; ++row)
    {
        equation.gradient[row] = gradient[row];
        for (int col = 0; col <= row; ++col) equation.hessian[index++] = hessian(row, col);
    }
    equation.squared_error = residual.squaredNorm();
    equation.count = 1;
    return equation;
}
}

void GICP::insertTargetPoints(const pcl::PointCloud<pcl::PointXYZ>& points) const
{
    mTarget->insert(points);
}

GICP::Result GICP::align(const pcl::PointCloud<pcl::PointXYZ>& source, const Eigen::Isometry3d& initial_guess) const
{
    if (source.empty())
    {
        throw std::invalid_argument("source cloud is empty!");
    }
    if (!initial_guess.matrix().allFinite() || !std::isfinite(mConfig.max_correspondence_distance) ||
        mConfig.max_correspondence_distance <= 0.0f)
    {
        throw std::invalid_argument("GICP requires a finite pose and a positive correspondence distance");
    }
    const auto source_index = std::make_unique<SparseVoxel>();
    source_index->insert(source);
    const SparseVoxel* source_pointer = source_index.get();
    const SparseVoxel* target_pointer = mTarget.get();
    thrust::device_vector<std::size_t> counts(2);
    countPoints<<<1, 1>>>(source_pointer, target_pointer, counts.data().get());
    std::size_t sizes[2];
    check(cudaMemcpy(sizes, counts.data().get(), sizeof(sizes), cudaMemcpyDeviceToHost), "read GICP point counts");

    Result result;
    result.transform = initial_guess;
    result.num_source_points = sizes[0];
    result.num_target_points = sizes[1];
    if (sizes[0] == 0 || sizes[1] == 0)
    {
        return result;
    }

    Sophus::SE3d pose(initial_guess.rotation(), initial_guess.translation());
    const auto first = thrust::make_counting_iterator<std::size_t>(0);
    const float max_distance = mConfig.max_correspondence_distance;
    for (int iteration = 0; iteration < mConfig.max_iterations; ++iteration)
    {
        Pose device_pose{};
        const Eigen::Matrix3d rotation = pose.rotationMatrix();
        std::copy_n(rotation.data(), 9, device_pose.rotation);
        std::copy_n(pose.translation().data(), 3, device_pose.translation);
        const Equation equation = thrust::transform_reduce(thrust::device, first, first + sizes[0],
            [=] __device__(const std::size_t i) -> Equation
            {
                return contribute(*source_pointer->points()[i], target_pointer, device_pose, max_distance);
            }, Equation{}, Sum{});
        if (equation.error || !std::isfinite(equation.squared_error))
        {
            throw std::runtime_error("GICP encountered invalid query coordinates or covariance");
        }
        if (equation.count == 0)
        {
            break;
        }

        Eigen::Matrix<double, 6, 6> hessian;
        Eigen::Matrix<double, 6, 1> rhs;
        int index = 0;
        for (int row = 0; row < 6; ++row)
        {
            rhs[row] = -equation.gradient[row];
            for (int col = 0; col <= row; ++col)
            {
                hessian(row, col) = hessian(col, row) = equation.hessian[index++];
            }
        }
        const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(hessian);
        if (decomposition.info() != Eigen::Success)
        {
            break;
        }
        const Sophus::SE3d::Tangent increment = decomposition.solve(rhs);
        if (!increment.allFinite())
        {
            break;
        }
        pose = Sophus::SE3d::exp(increment) * pose;
        result.num_correspondences = equation.count;
        result.fitness_score = equation.squared_error / static_cast<double>(equation.count);
        ++result.iterations;
        if (increment.head<3>().norm() < mConfig.convergence_translation && increment.tail<3>().norm() < mConfig.convergence_rotation)
        {
            break;
        }
    }
    result.transform = Eigen::Isometry3d(pose.matrix());
    result.converged = result.iterations > 0 && std::isfinite(result.fitness_score);
    return result;
}
