#include "localization/GICP/GICP.h"
#include "localization/types/SparseVoxelDevice.cuh"
#include <cub/block/block_reduce.cuh>
#include <cstring>

namespace
{
using namespace localization::cuda;

struct Pose
{
    double matrix[16];
};

struct FloatPose
{
    float matrix[16];
};

struct Preprocessing
{
    FloatPose sensor_to_output;
    Pose lidar_to_imu;
    Pose world_to_lidar_end;
    double begin_time{0.0};
    double end_time{0.0};
    Index trajectory_size{0};
    float inverse_voxel_size{1.0f};
    float intensity_threshold{-1.0f};
    float footprint_x{0.0f};
    float footprint_y{0.0f};
};

struct State
{
    double rotation[9];
    double translation[3];
    double initial_rotation[9];
    double initial_translation[3];
    float map_transform[16];
    float render_transform[16];
    Count processed;
    Count invalid;
    Count source_points;
    Count target_points;
    Count correspondences;
    double fitness;
    int iterations;
    int active;
    int converged;
    int initialize;
    int map_enabled;
};

struct Download
{
    State state;
    Metadata source;
    Metadata target;
    Count visual_points;
    Status status;
};

enum class Mode : int { align_only, initialize, frame };

struct NormalEquation
{
    double upper[21]{};
    double gradient[6]{};
    double squared_error{0.0};
    Count count{0};
};

struct SumEquation
{
    __host__ __device__ NormalEquation operator()(NormalEquation lhs, const NormalEquation& rhs) const
    {
        for (int i = 0; i < 21; ++i) lhs.upper[i] += rhs.upper[i];
        for (int i = 0; i < 6; ++i) lhs.gradient[i] += rhs.gradient[i];
        lhs.squared_error += rhs.squared_error;
        lhs.count += rhs.count;
        return lhs;
    }
};

template <int N>
struct FixedLDLT
{
    double lower[N][N]{};
    double diagonal[N]{};
    int permutation[N]{};

    __device__ bool compute(const Eigen::Matrix<double, N, N>& matrix)
    {
        double schur[N][N];
        for (int row = 0; row < N; ++row)
        {
            permutation[row] = row;
            for (int col = 0; col < N; ++col)
            {
                if (!isfinite(matrix(row, col))) return false;
                schur[row][col] = matrix(row, col);
                lower[row][col] = row == col ? 1.0 : 0.0;
            }
        }
        for (int k = 0; k < N; ++k)
        {
            int pivot = k;
            for (int row = k + 1; row < N; ++row)
                if (fabs(schur[row][row]) > fabs(schur[pivot][pivot])) pivot = row;
            if (pivot != k)
            {
                for (int col = 0; col < N; ++col)
                {
                    const double value = schur[k][col];
                    schur[k][col] = schur[pivot][col];
                    schur[pivot][col] = value;
                }
                for (int row = 0; row < N; ++row)
                {
                    const double value = schur[row][k];
                    schur[row][k] = schur[row][pivot];
                    schur[row][pivot] = value;
                }
                for (int col = 0; col < k; ++col)
                {
                    const double value = lower[k][col];
                    lower[k][col] = lower[pivot][col];
                    lower[pivot][col] = value;
                }
                const int index = permutation[k];
                permutation[k] = permutation[pivot];
                permutation[pivot] = index;
            }
            diagonal[k] = schur[k][k];
            if (!isfinite(diagonal[k])) return false;
            if (fabs(diagonal[k]) <= std::numeric_limits<double>::min())
            {
                diagonal[k] = 0.0;
                for (int row = k + 1; row < N; ++row)
                    if (schur[row][k] != 0.0) return false;
                continue;
            }
            for (int row = k + 1; row < N; ++row) lower[row][k] = schur[row][k] / diagonal[k];
            for (int row = k + 1; row < N; ++row)
                for (int col = row; col < N; ++col)
                {
                    schur[col][row] -= lower[col][k] * diagonal[k] * lower[row][k];
                    schur[row][col] = schur[col][row];
                }
        }
        return true;
    }

    __device__ bool solve(const Eigen::Matrix<double, N, 1>& rhs, Eigen::Matrix<double, N, 1>& result) const
    {
        double values[N];
        for (int row = 0; row < N; ++row)
        {
            values[row] = rhs[permutation[row]];
            for (int col = 0; col < row; ++col) values[row] -= lower[row][col] * values[col];
        }
        for (int row = 0; row < N; ++row)
            // Match Eigen's zero-pivot solution for rank-deficient normal equations.
            values[row] = diagonal[row] == 0.0 ? 0.0 : values[row] / diagonal[row];
        for (int row = N - 1; row >= 0; --row)
        {
            for (int col = row + 1; col < N; ++col) values[row] -= lower[col][row] * values[col];
            if (!isfinite(values[row])) return false;
            result[permutation[row]] = values[row];
        }
        return true;
    }
};

template <typename Scalar>
__device__ Eigen::Matrix<Scalar, 3, 1> transformPoint(const Scalar* matrix,
                                                   const Eigen::Matrix<Scalar, 3, 1>& point)
{
    return {matrix[0] * point.x() + matrix[4] * point.y() + matrix[8] * point.z() + matrix[12],
            matrix[1] * point.x() + matrix[5] * point.y() + matrix[9] * point.z() + matrix[13],
            matrix[2] * point.x() + matrix[6] * point.y() + matrix[10] * point.z() + matrix[14]};
}

Pose poseValue(const Eigen::Isometry3d& pose)
{
    if (!pose.matrix().allFinite() ||
        !(pose.rotation().transpose() * pose.rotation()).isApprox(Eigen::Matrix3d::Identity(), 1e-6) ||
        std::abs(pose.rotation().determinant() - 1.0) > 1e-6)
        throw std::invalid_argument("GICP pose must be a finite rigid transform");
    Pose result{};
    std::copy_n(pose.matrix().data(), 16, result.matrix);
    return result;
}

FloatPose floatPoseValue(const Eigen::Isometry3d& pose)
{
    const Pose value = poseValue(pose);
    FloatPose result{};
    for (int i = 0; i < 16; ++i)
    {
        result.matrix[i] = static_cast<float>(value.matrix[i]);
        if (!std::isfinite(result.matrix[i]))
            throw std::out_of_range("GICP pose cannot be represented by float point coordinates");
    }
    return result;
}

Eigen::Isometry3d trajectoryPose(std::span<const TimedPose> samples, double timestamp)
{
    const auto after = std::lower_bound(samples.begin(), samples.end(), timestamp,
        [](const TimedPose& pose, double time) { return pose.timestamp < time; });
    const TimedPose& right = after == samples.end() ? samples.back() : *after;
    const TimedPose& left = after == samples.begin() || after == samples.end() ? right : *(after - 1);
    const double alpha = right.timestamp == left.timestamp ? 0.0 :
        std::clamp((timestamp - left.timestamp) / (right.timestamp - left.timestamp), 0.0, 1.0);
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    result.linear() = left.rotation.slerp(alpha, right.rotation).toRotationMatrix();
    result.translation() = (1.0 - alpha) * left.translation + alpha * right.translation;
    return result;
}

void validateConfig(const GICP::Config& config)
{
    if (!std::isfinite(config.max_correspondence_distance) || config.max_correspondence_distance <= 0.0f ||
        config.max_iterations < 0 || !std::isfinite(config.convergence_translation) ||
        config.convergence_translation < 0.0 || !std::isfinite(config.convergence_rotation) ||
        config.convergence_rotation < 0.0 || std::isnan(config.max_fitness_score) || config.max_fitness_score < 0.0)
        throw std::invalid_argument("GICP configuration contains an invalid threshold or iteration count");
}

Preprocessing preprocessValue(const GICP::PreprocessConfig& config)
{
    if (!std::isfinite(config.voxel_size) || config.voxel_size <= 0.0f ||
        !std::isfinite(config.map_voxel_size) || config.map_voxel_size <= 0.0f ||
        !std::isfinite(config.intensity_threshold) || !std::isfinite(config.footprint_x) ||
        !std::isfinite(config.footprint_y) || config.footprint_x < 0.0f || config.footprint_y < 0.0f)
        throw std::invalid_argument("GICP preprocessing configuration is invalid");
    const Eigen::Isometry3d extrinsic = config.sensor_to_output.cast<double>();
    Preprocessing result{};
    result.sensor_to_output = floatPoseValue(extrinsic);
    result.inverse_voxel_size = 1.0f / config.voxel_size;
    if (!std::isfinite(result.inverse_voxel_size))
        throw std::invalid_argument("GICP preprocessing voxel size is too small");
    result.intensity_threshold = config.intensity_threshold;
    result.footprint_x = config.footprint_x;
    result.footprint_y = config.footprint_y;
    return result;
}

__device__ bool gridKey(const Eigen::Vector3f& point, float inverse_size, GridKey& key)
{
    int values[3];
    for (int axis = 0; axis < 3; ++axis)
    {
        const float value = floorf(point[axis] * inverse_size);
        if (!isfinite(value) || static_cast<double>(value) < std::numeric_limits<int>::min() ||
            static_cast<double>(value) > std::numeric_limits<int>::max()) return false;
        values[axis] = static_cast<int>(value);
    }
    key = {0, values[0], values[1], values[2]};
    return true;
}

struct Sample
{
    Eigen::Vector3f position;
    float intensity;
    double offset;
};

__device__ Sample sample(const pcl::PointXYZI& point)
{
    return {{point.x, point.y, point.z}, point.intensity, 0.0};
}

__device__ Sample sample(const PointXYZIT& point)
{
    return {point.position, point.intensity, point.time_offset};
}

__device__ Eigen::Vector3f deskewPoint(const Sample& point, const Preprocessing& config, const TimedPose* trajectory)
{
    const double timestamp = fmin(config.end_time, fmax(config.begin_time, config.begin_time + point.offset));
    Index lo = 0;
    Index hi = config.trajectory_size - 1;
    double alpha = 0.0;
    if (timestamp <= trajectory[lo].timestamp) hi = lo;
    else if (timestamp >= trajectory[hi].timestamp) lo = hi;
    else
    {
        while (lo + 1 < hi)
        {
            const Index mid = lo + (hi - lo) / 2;
            if (trajectory[mid].timestamp < timestamp) lo = mid;
            else hi = mid;
        }
        alpha = (timestamp - trajectory[lo].timestamp) / (trajectory[hi].timestamp - trajectory[lo].timestamp);
    }
    const Eigen::Quaterniond rotation = trajectory[lo].rotation.slerp(alpha, trajectory[hi].rotation);
    const Eigen::Vector3d translation =
        (1.0 - alpha) * trajectory[lo].translation + alpha * trajectory[hi].translation;
    const Eigen::Vector3d local = point.position.cast<double>();
    const Eigen::Vector3d imu = transformPoint(config.lidar_to_imu.matrix, local);
    const Eigen::Vector3d world = rotation * imu + translation;
    return transformPoint(config.world_to_lidar_end.matrix, world).cast<float>();
}

template <typename Input>
__global__ void preprocessKeysKernel(const Input* input, Index count, Preprocessing config,
                                     const TimedPose* trajectory, Eigen::Vector3f* positions,
                                     GridKey* keys, Index* indices, State* state, Status* status)
{
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<Index>(blockDim.x) * gridDim.x)
    {
        keys[i] = {1, 0, 0, 0};
        indices[i] = i;
        if (status->code != 0) continue;
        const Sample value = sample(input[i]);
        if (!isfinite(value.position.x()) || !isfinite(value.position.y()) || !isfinite(value.position.z()) ||
            !isfinite(value.intensity) || !isfinite(value.offset))
        {
            atomicAdd(&state->invalid, Count{1});
            continue;
        }
        const Eigen::Vector3f position = config.trajectory_size == 0 ? value.position : deskewPoint(value, config, trajectory);
        if (value.intensity < config.intensity_threshold) continue;
        if (config.footprint_x > 0.0f && config.footprint_y > 0.0f &&
            fabsf(position.x()) <= config.footprint_x && fabsf(position.y()) <= config.footprint_y) continue;
        GridKey key{};
        if (!gridKey(position, config.inverse_voxel_size, key))
        {
            setError(status, Error::coordinate_range, i);
            continue;
        }
        positions[i] = position;
        keys[i] = key;
    }
}

__global__ void gridHeadsKernel(const GridKey* keys, Index count, Index* flags)
{
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<Index>(blockDim.x) * gridDim.x)
        flags[i] = !keys[i].invalid && (i == 0 || !(keys[i] == keys[i - 1])) ? 1 : 0;
}

__global__ void gridCountKernel(const Index* flags, const Index* ranks, Index count, Count* output)
{
    *output = count == 0 ? 0 : ranks[count - 1] + flags[count - 1];
}

__global__ void centroidsKernel(const Eigen::Vector3f* input, const GridKey* keys, const Index* indices,
                               const Index* flags, const Index* ranks, Index count, FloatPose transform,
                               Eigen::Vector3f* output, const Status* status)
{
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<Index>(blockDim.x) * gridDim.x)
    {
        if (status->code != 0 || !flags[i]) continue;
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        Index j = i;
        for (; j < count && keys[j] == keys[i]; ++j) sum += input[indices[j]];
        const Eigen::Vector3f mean = sum / static_cast<float>(j - i);
        output[ranks[i]] = transformPoint(transform.matrix, mean);
    }
}

__global__ void xyzKernel(const pcl::PointXYZ* input, Eigen::Vector3f* output, Index count, State* state)
{
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<Index>(blockDim.x) * gridDim.x)
        output[i] = Eigen::Vector3f(input[i].x, input[i].y, input[i].z);
    if (blockIdx.x == 0 && threadIdx.x == 0) state->processed = count;
}

__global__ void startKernel(State* state, Pose initial, GICP::Config config, const Metadata* source,
                            const Metadata* target, Mode mode, Count minimum_points, const Status* status)
{
    for (int col = 0; col < 3; ++col)
        for (int row = 0; row < 3; ++row)
            state->rotation[col * 3 + row] = state->initial_rotation[col * 3 + row] = initial.matrix[col * 4 + row];
    for (int row = 0; row < 3; ++row)
        state->translation[row] = state->initial_translation[row] = initial.matrix[12 + row];
    state->source_points = source->points;
    state->target_points = target->points;
    state->correspondences = 0;
    state->fitness = std::numeric_limits<double>::infinity();
    state->iterations = 0;
    state->converged = 0;
    state->initialize = status->code == 0 && mode != Mode::align_only &&
                        (mode == Mode::initialize || target->points == 0) && state->processed != 0 &&
                        state->processed >= minimum_points;
    state->active = status->code == 0 && mode != Mode::initialize && target->points != 0 &&
                    source->points != 0 && config.max_iterations != 0;
    state->map_enabled = 0;
}

__device__ NormalEquation contribution(const PointWithCovariance& source, DeviceView target,
                                      const State& state, float squared_radius, Status* status, Index index)
{
    NormalEquation result{};
    const Eigen::Map<const Eigen::Matrix3d> rotation(state.rotation);
    const Eigen::Map<const Eigen::Vector3d> translation(state.translation);
    const Eigen::Vector3d transformed = rotation * source.position.cast<double>() + translation;
    const Eigen::Vector3f query = transformed.cast<float>();
    const Error error = positionError(query);
    if (error != Error::none)
    {
        setError(status, error, index);
        return result;
    }
    const auto found = nearest(target, query, squared_radius);
    if (found.index == INVALID_INDEX) return result;
    const PointWithCovariance& destination = target.point(found.index);
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
    if (source.covariance_valid)
    {
        if (destination.covariance_valid) covariance = destination.covariance.cast<double>();
        covariance.noalias() += rotation * source.covariance.cast<double>() * rotation.transpose();
    }
    FixedLDLT<3> factor;
    if (!factor.compute(covariance)) return result;
    const Eigen::Vector3d residual = transformed - destination.position.cast<double>();
    Eigen::Vector3d precision_residual;
    if (!factor.solve(residual, precision_residual)) return result;
    double jacobian[3][6] = {
        {1.0, 0.0, 0.0, 0.0, transformed.z(), -transformed.y()},
        {0.0, 1.0, 0.0, -transformed.z(), 0.0, transformed.x()},
        {0.0, 0.0, 1.0, transformed.y(), -transformed.x(), 0.0}};
    double precision_jacobian[3][6];
    for (int col = 0; col < 6; ++col)
    {
        const Eigen::Vector3d rhs(jacobian[0][col], jacobian[1][col], jacobian[2][col]);
        Eigen::Vector3d value;
        if (!factor.solve(rhs, value)) return NormalEquation{};
        for (int row = 0; row < 3; ++row) precision_jacobian[row][col] = value[row];
    }
    int next = 0;
    for (int row = 0; row < 6; ++row)
    {
        for (int col = row; col < 6; ++col)
        {
            for (int axis = 0; axis < 3; ++axis)
                result.upper[next] += jacobian[axis][row] * precision_jacobian[axis][col];
            ++next;
        }
        for (int axis = 0; axis < 3; ++axis) result.gradient[row] += jacobian[axis][row] * precision_residual[axis];
    }
    result.squared_error = residual.squaredNorm();
    result.count = 1;
    return result;
}

__global__ void linearizeKernel(DeviceView source, DeviceView target, const State* state,
                               NormalEquation* partials, float squared_radius, Status* status)
{
    if (!state->active) return;
    NormalEquation local{};
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x; i < source.metadata->points;
         i += static_cast<Index>(blockDim.x) * gridDim.x)
        local = SumEquation{}(local, contribution(source.point(source.point_indices[i]), target, *state,
                                                 squared_radius, status, i));
    using Reduction = cub::BlockReduce<NormalEquation, BLOCK_SIZE>;
    __shared__ typename Reduction::TempStorage storage;
    const NormalEquation sum = Reduction(storage).Reduce(local, SumEquation{});
    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}

__device__ bool leftUpdate(State* state, const Eigen::Matrix<double, 6, 1>& increment)
{
    const Eigen::Vector3d velocity = increment.head<3>();
    const Eigen::Vector3d omega = increment.tail<3>();
    const double squared_angle = omega.squaredNorm();
    double a, b, c;
    if (squared_angle < 1e-8)
    {
        const double fourth = squared_angle * squared_angle;
        a = 1.0 - squared_angle / 6.0 + fourth / 120.0;
        b = 0.5 - squared_angle / 24.0 + fourth / 720.0;
        c = 1.0 / 6.0 - squared_angle / 120.0 + fourth / 5040.0;
    }
    else
    {
        const double angle = sqrt(squared_angle);
        a = sin(angle) / angle;
        b = (1.0 - cos(angle)) / squared_angle;
        c = (1.0 - a) / squared_angle;
    }
    Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
    skew(0, 1) = -omega.z();
    skew(0, 2) = omega.y();
    skew(1, 0) = omega.z();
    skew(1, 2) = -omega.x();
    skew(2, 0) = -omega.y();
    skew(2, 1) = omega.x();
    const Eigen::Matrix3d square = skew * skew;
    const Eigen::Matrix3d delta_rotation = Eigen::Matrix3d::Identity() + a * skew + b * square;
    const Eigen::Vector3d delta_translation = (Eigen::Matrix3d::Identity() + b * skew + c * square) * velocity;
    const Eigen::Map<const Eigen::Matrix3d> rotation(state->rotation);
    const Eigen::Map<const Eigen::Vector3d> translation(state->translation);
    const Eigen::Matrix3d updated_rotation = delta_rotation * rotation;
    const Eigen::Vector3d updated_translation = delta_rotation * translation + delta_translation;
    for (int i = 0; i < 9; ++i) if (!isfinite(updated_rotation.data()[i])) return false;
    for (int i = 0; i < 3; ++i) if (!isfinite(updated_translation[i])) return false;
    for (int i = 0; i < 9; ++i) state->rotation[i] = updated_rotation.data()[i];
    for (int i = 0; i < 3; ++i) state->translation[i] = updated_translation[i];
    return true;
}

__global__ void solveKernel(const NormalEquation* partials, Index count, State* state,
                           GICP::Config config, Status* status)
{
    if (!state->active) return;
    NormalEquation local{};
    for (Index i = threadIdx.x; i < count; i += blockDim.x) local = SumEquation{}(local, partials[i]);
    using Reduction = cub::BlockReduce<NormalEquation, BLOCK_SIZE>;
    __shared__ typename Reduction::TempStorage storage;
    const NormalEquation sum = Reduction(storage).Reduce(local, SumEquation{});
    if (threadIdx.x != 0) return;
    if (status->code != 0 || sum.count == 0 || !isfinite(sum.squared_error))
    {
        state->active = 0;
        return;
    }
    Eigen::Matrix<double, 6, 6> hessian;
    Eigen::Matrix<double, 6, 1> rhs;
    int next = 0;
    for (int row = 0; row < 6; ++row)
    {
        rhs[row] = -sum.gradient[row];
        for (int col = row; col < 6; ++col) hessian(row, col) = hessian(col, row) = sum.upper[next++];
    }
    FixedLDLT<6> factor;
    Eigen::Matrix<double, 6, 1> increment;
    if (!factor.compute(hessian) || !factor.solve(rhs, increment))
    {
        state->active = 0;
        return;
    }
    if (!leftUpdate(state, increment))
    {
        setError(status, Error::solver, 0);
        state->active = 0;
        return;
    }
    state->correspondences = sum.count;
    state->fitness = sum.squared_error / static_cast<double>(sum.count);
    ++state->iterations;
    if (increment.head<3>().norm() < config.convergence_translation &&
        increment.tail<3>().norm() < config.convergence_rotation)
    {
        state->converged = 1;
        state->active = 0;
    }
}

__global__ void finalizeKernel(State* state, GICP::Config config, Mode mode, const Status* status)
{
    if (!state->converged && state->iterations > 0 && isfinite(state->fitness)) state->converged = 1;
    const bool pose_accepted = state->converged && state->correspondences >= config.min_correspondences;
    state->map_enabled = status->code == 0 &&
        (state->initialize || (mode == Mode::frame && pose_accepted && state->fitness <= config.max_fitness_score));
    for (int i = 0; i < 16; ++i)
    {
        state->map_transform[i] = 0.0f;
        state->render_transform[i] = 0.0f;
    }
    state->map_transform[15] = state->render_transform[15] = 1.0f;
    for (int col = 0; col < 3; ++col)
        for (int row = 0; row < 3; ++row)
        {
            state->map_transform[col * 4 + row] = static_cast<float>(state->rotation[col * 3 + row]);
            state->render_transform[col * 4 + row] = static_cast<float>(
                pose_accepted ? state->rotation[col * 3 + row] : state->initial_rotation[col * 3 + row]);
        }
    for (int row = 0; row < 3; ++row)
    {
        state->map_transform[12 + row] = static_cast<float>(state->translation[row]);
        state->render_transform[12 + row] = static_cast<float>(
            pose_accepted ? state->translation[row] : state->initial_translation[row]);
    }
}

__global__ void setPoseKernel(State* state, FloatPose pose, const Metadata* target, bool insert)
{
    for (int i = 0; i < 16; ++i)
    {
        if (insert) state->map_transform[i] = pose.matrix[i];
        else state->render_transform[i] = pose.matrix[i];
    }
    if (insert)
    {
        state->initialize = target->points == 0;
        state->map_enabled = 1;
    }
}

__global__ void visualKeysKernel(const Eigen::Vector3f* old_points, const Count* old_count,
                                const Eigen::Vector3f* scan, const State* state, Index bound, float inverse_size,
                                Eigen::Vector3f* merged, GridKey* keys, Index* indices, Status* status)
{
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x; i < bound;
         i += static_cast<Index>(blockDim.x) * gridDim.x)
    {
        indices[i] = i;
        keys[i] = {1, 0, 0, 0};
        if (status->code != 0 || !state->map_enabled || state->initialize ||
            i >= *old_count + state->processed) continue;
        const Eigen::Vector3f point = i < *old_count ? old_points[i]
            : transformPoint(state->map_transform, scan[i - *old_count]);
        GridKey key{};
        if (!gridKey(point, inverse_size, key))
        {
            setError(status, Error::coordinate_range, i);
            continue;
        }
        merged[i] = point;
        keys[i] = key;
    }
}

__global__ void commitVisualKernel(Eigen::Vector3f* destination, Count* count, const Eigen::Vector3f* centroids,
                                  const Count* centroid_count, const Eigen::Vector3f* scan,
                                  const State* state, Index bound, const Status* status)
{
    if (status->code != 0 || !state->map_enabled) return;
    const Count size = state->initialize ? state->processed : *centroid_count;
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x; i < bound;
         i += static_cast<Index>(blockDim.x) * gridDim.x)
        if (i < size) destination[i] = state->initialize ? transformPoint(state->map_transform, scan[i]) : centroids[i];
    if (blockIdx.x == 0 && threadIdx.x == 0) *count = size;
}

__global__ void renderKernel(const Eigen::Vector3f* scan, Eigen::Vector3f* output, Index bound, const State* state)
{
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x; i < bound;
         i += static_cast<Index>(blockDim.x) * gridDim.x)
    {
        if (i < state->processed) output[i] = transformPoint(state->render_transform, scan[i]);
        else output[i].setZero();
    }
}

__global__ void downloadKernel(const State* state, const Metadata* source, const Metadata* target,
                              const Count* visual_count, const Status* status, Download* output)
{
    output->state = *state;
    output->source = *source;
    output->target = *target;
    output->visual_points = *visual_count;
    output->status = *status;
}
}

struct GICP::Impl
{
    Stream stream;
    SparseVoxel source;
    SparseVoxel target;
    Buffer<State> state{stream.value};
    Buffer<Status> status{stream.value};
    Buffer<Download> output{stream.value};
    PinnedBuffer<Download> host_output;
    Buffer<pcl::PointXYZ> xyz{stream.value};
    PinnedBuffer<pcl::PointXYZ> xyz_host;
    Buffer<pcl::PointXYZI> xyzi{stream.value};
    PinnedBuffer<pcl::PointXYZI> xyzi_host;
    Buffer<PointXYZIT> timed{stream.value};
    PinnedBuffer<PointXYZIT> timed_host;
    Buffer<TimedPose> trajectory{stream.value};
    PinnedBuffer<TimedPose> trajectory_host;
    Buffer<Eigen::Vector3f> raw_positions{stream.value};
    Buffer<Eigen::Vector3f> positions{stream.value};
    RadixGroups<GridKey> preprocessing{stream.value};
    Buffer<NormalEquation> partials{stream.value};
    Buffer<Eigen::Vector3f> visual_points{stream.value};
    Buffer<Count> visual_count{stream.value};
    Buffer<Eigen::Vector3f> merged{stream.value};
    Buffer<Eigen::Vector3f> visual_centroids{stream.value};
    Buffer<Count> centroid_count{stream.value};
    RadixGroups<GridKey> visual_groups{stream.value};
    Buffer<Eigen::Vector3f> rendered{stream.value};
    PinnedBuffer<Eigen::Vector3f> scan_host;
    PinnedBuffer<Eigen::Vector3f> map_host;
    std::mutex mutex;
    Index bound{0};
    Index visual_upper_bound{0};
    Index processed_count{0};
    float map_voxel_size{0.2f};
    bool upload_pending{false};
    bool prepared{false};

    Impl()
    {
        source.mImpl = std::make_unique<SparseVoxel::Impl>(stream.value);
        target.mImpl = std::make_unique<SparseVoxel::Impl>(stream.value);
        state.reserve(1);
        status.reserve(1);
        output.reserve(1);
        host_output.reserve(1);
        visual_count.reserve(1);
        centroid_count.reserve(1);
    }

    ~Impl() { logError(cudaStreamSynchronize(stream.value), "finish registration teardown"); }

    void begin(Index size)
    {
        if (upload_pending)
            check(cudaStreamSynchronize(stream.value), "finish replaced prepared input");
        prepared = false;
        itemCount(size);
        positions.reserve(size);
        raw_positions.reserve(size);
        bound = size;
        check(cudaMemsetAsync(state.data(), 0, sizeof(State), stream.value), "reset registration state");
        source.mImpl->clearAsync();
        upload_pending = true;
    }

    void buildSource()
    {
        source.mImpl->insertAsync({positions.data(), &state.data()->processed, bound}, nullptr, nullptr, status.data());
    }

    void prepareXYZ(const pcl::PointCloud<pcl::PointXYZ>& input, bool build_source = true)
    {
        begin(input.size());
        xyz.reserve(bound);
        xyz_host.reserve(bound);
        if (bound != 0)
        {
            std::memcpy(xyz_host.data(), input.data(), bound * sizeof(pcl::PointXYZ));
            check(cudaMemcpyAsync(xyz.data(), xyz_host.data(), bound * sizeof(pcl::PointXYZ),
                                  cudaMemcpyHostToDevice, stream.value), "upload registration input");
            xyzKernel<<<blocks(bound), BLOCK_SIZE, 0, stream.value>>>(xyz.data(), positions.data(), bound, state.data());
            check(cudaGetLastError(), "launch registration input conversion");
        }
        if (build_source) buildSource();
        prepared = true;
    }

    template <typename Input>
    void preprocess(const Input* input, Preprocessing config)
    {
        preprocessing.reserve(bound);
        if (bound != 0)
        {
            preprocessKeysKernel<<<blocks(bound), BLOCK_SIZE, 0, stream.value>>>(
                input, bound, config, trajectory.data(), raw_positions.data(), preprocessing.input_keys.data(),
                preprocessing.input_indices.data(), state.data(), status.data());
            check(cudaGetLastError(), "launch scan preprocessing");
            preprocessing.sort(bound);
            gridHeadsKernel<<<blocks(bound), BLOCK_SIZE, 0, stream.value>>>(
                preprocessing.keys.data(), bound, preprocessing.flags.data());
            preprocessing.scan(bound);
            gridCountKernel<<<1, 1, 0, stream.value>>>(
                preprocessing.flags.data(), preprocessing.ranks.data(), bound, &state.data()->processed);
            centroidsKernel<<<blocks(bound), BLOCK_SIZE, 0, stream.value>>>(
                raw_positions.data(), preprocessing.keys.data(), preprocessing.indices.data(),
                preprocessing.flags.data(), preprocessing.ranks.data(), bound,
                config.sensor_to_output, positions.data(), status.data());
            check(cudaGetLastError(), "launch scan centroids");
        }
        buildSource();
        prepared = true;
    }

    void prepare(const pcl::PointCloud<pcl::PointXYZI>& input, const PreprocessConfig& config)
    {
        const Preprocessing parameters = preprocessValue(config);
        begin(input.size());
        map_voxel_size = config.map_voxel_size;
        xyzi.reserve(bound);
        xyzi_host.reserve(bound);
        if (bound != 0)
        {
            std::memcpy(xyzi_host.data(), input.data(), bound * sizeof(pcl::PointXYZI));
            check(cudaMemcpyAsync(xyzi.data(), xyzi_host.data(), bound * sizeof(pcl::PointXYZI),
                                  cudaMemcpyHostToDevice, stream.value), "upload raw scan");
        }
        preprocess(xyzi.data(), parameters);
    }

    void prepare(const StampedCloud& input, const PreprocessConfig& config,
                 std::span<const TimedPose> samples, const Eigen::Isometry3d& extrinsic)
    {
        Preprocessing parameters = preprocessValue(config);
        if (!std::isfinite(input.begin_time) || !std::isfinite(input.end_time) || input.end_time < input.begin_time)
            throw std::invalid_argument("Deskew scan timestamps are invalid");
        parameters.lidar_to_imu = poseValue(extrinsic);
        parameters.begin_time = input.begin_time;
        parameters.end_time = input.end_time;
        parameters.trajectory_size = samples.size();
        for (Index i = 0; i < samples.size(); ++i)
        {
            if (!std::isfinite(samples[i].timestamp) || !samples[i].translation.allFinite() ||
                !samples[i].rotation.coeffs().allFinite() ||
                std::abs(samples[i].rotation.norm() - 1.0) > 1e-6 ||
                (i != 0 && samples[i].timestamp <= samples[i - 1].timestamp))
                throw std::invalid_argument("Deskew trajectory must contain finite, ordered rigid poses");
        }
        if (!samples.empty())
        {
            const Eigen::Isometry3d end = trajectoryPose(samples, input.end_time);
            parameters.world_to_lidar_end = poseValue((end * extrinsic).inverse());
        }
        begin(input.points.size());
        map_voxel_size = config.map_voxel_size;
        timed.reserve(bound);
        timed_host.reserve(bound);
        trajectory.reserve(samples.size());
        trajectory_host.reserve(samples.size());
        if (bound != 0)
        {
            std::memcpy(timed_host.data(), input.points.data(), bound * sizeof(PointXYZIT));
            check(cudaMemcpyAsync(timed.data(), timed_host.data(), bound * sizeof(PointXYZIT),
                                  cudaMemcpyHostToDevice, stream.value), "upload timed scan");
        }
        if (!samples.empty())
        {
            std::copy(samples.begin(), samples.end(), trajectory_host.data());
            check(cudaMemcpyAsync(trajectory.data(), trajectory_host.data(), samples.size_bytes(),
                                  cudaMemcpyHostToDevice, stream.value), "upload IMU trajectory");
        }
        preprocess(timed.data(), parameters);
    }

    void optimize(const Eigen::Isometry3d& initial, const Config& config, Mode mode, Index minimum_points)
    {
        if (!prepared) throw std::logic_error("No GPU scan has been prepared");
        validateConfig(config);
        const Pose pose = poseValue(initial);
        const Index partial_count = std::max<Index>(1, blocks(bound));
        partials.reserve(partial_count);
        startKernel<<<1, 1, 0, stream.value>>>(
            state.data(), pose, config, source.mImpl->metadata.data(), target.mImpl->metadata.data(),
            mode, minimum_points, status.data());
        check(cudaGetLastError(), "launch registration initialization");
        const float radius = config.max_correspondence_distance * config.max_correspondence_distance;
        if (mode != Mode::initialize && bound != 0)
        {
            // Queue the batch without host readbacks; device flags stop work after convergence.
            for (int iteration = 0; iteration < config.max_iterations; ++iteration)
            {
                linearizeKernel<<<partial_count, BLOCK_SIZE, 0, stream.value>>>(
                    source.mImpl->view(), target.mImpl->view(), state.data(), partials.data(), radius, status.data());
                solveKernel<<<1, BLOCK_SIZE, 0, stream.value>>>(partials.data(), partial_count, state.data(), config, status.data());
                check(cudaGetLastError(), "launch registration iteration");
            }
        }
        finalizeKernel<<<1, 1, 0, stream.value>>>(state.data(), config, mode, status.data());
        check(cudaGetLastError(), "launch registration acceptance");
    }

    void updateVisual()
    {
        if (bound > std::numeric_limits<Index>::max() - visual_upper_bound)
            throw std::length_error("Visualization point count exceeds capacity");
        const Index count = visual_upper_bound + bound;
        visual_upper_bound = count;
        visual_points.reserve(count);
        merged.reserve(count);
        visual_centroids.reserve(count);
        visual_groups.reserve(count);
        if (count == 0) return;
        visualKeysKernel<<<blocks(count), BLOCK_SIZE, 0, stream.value>>>(
            visual_points.data(), visual_count.data(), positions.data(), state.data(), count, 1.0f / map_voxel_size,
            merged.data(), visual_groups.input_keys.data(), visual_groups.input_indices.data(), status.data());
        visual_groups.sort(count);
        gridHeadsKernel<<<blocks(count), BLOCK_SIZE, 0, stream.value>>>(
            visual_groups.keys.data(), count, visual_groups.flags.data());
        visual_groups.scan(count);
        gridCountKernel<<<1, 1, 0, stream.value>>>(
            visual_groups.flags.data(), visual_groups.ranks.data(), count, centroid_count.data());
        FloatPose identity{};
        identity.matrix[0] = identity.matrix[5] = identity.matrix[10] = identity.matrix[15] = 1.0f;
        centroidsKernel<<<blocks(count), BLOCK_SIZE, 0, stream.value>>>(
            merged.data(), visual_groups.keys.data(), visual_groups.indices.data(), visual_groups.flags.data(),
            visual_groups.ranks.data(), count, identity, visual_centroids.data(), status.data());
        commitVisualKernel<<<blocks(count), BLOCK_SIZE, 0, stream.value>>>(
            visual_points.data(), visual_count.data(), visual_centroids.data(), centroid_count.data(),
            positions.data(), state.data(), count, status.data());
        check(cudaGetLastError(), "launch visualization map update");
    }

    void insert()
    {
        prepared = false;
        target.mImpl->insertAsync({positions.data(), &state.data()->processed, bound},
            state.data()->map_transform, &state.data()->map_enabled, status.data());
        updateVisual();
        prepared = true;
    }

    FrameResult finish(bool download_scan, bool download_map)
    {
        if (download_scan && bound != 0)
        {
            rendered.reserve(bound);
            scan_host.reserve(bound);
            renderKernel<<<blocks(bound), BLOCK_SIZE, 0, stream.value>>>(
                positions.data(), rendered.data(), bound, state.data());
            check(cudaGetLastError(), "launch visualization scan transform");
            check(cudaMemcpyAsync(scan_host.data(), rendered.data(), bound * sizeof(Eigen::Vector3f),
                                  cudaMemcpyDeviceToHost, stream.value), "download subscribed scan");
        }
        if (download_map && visual_upper_bound != 0)
        {
            map_host.reserve(visual_upper_bound);
            check(cudaMemcpyAsync(map_host.data(), visual_points.data(), visual_upper_bound * sizeof(Eigen::Vector3f),
                                  cudaMemcpyDeviceToHost, stream.value), "download subscribed map");
        }
        downloadKernel<<<1, 1, 0, stream.value>>>(
            state.data(), source.mImpl->metadata.data(), target.mImpl->metadata.data(),
            visual_count.data(), status.data(), output.data());
        check(cudaGetLastError(), "launch registration output");
        check(cudaMemcpyAsync(host_output.data(), output.data(), sizeof(Download), cudaMemcpyDeviceToHost, stream.value),
              "download registration result");
        check(cudaStreamSynchronize(stream.value), "finish registration result");
        upload_pending = false;
        const Download& value = *host_output.data();
        if (value.state.processed > bound || value.visual_points > visual_upper_bound ||
            value.source.voxels > source.mImpl->voxel_upper_bound ||
            value.source.points > source.mImpl->point_upper_bound ||
            value.target.voxels > target.mImpl->voxel_upper_bound ||
            value.target.points > target.mImpl->point_upper_bound)
        {
            prepared = false;
            throw std::runtime_error("CUDA registration returned invalid point counts");
        }
        source.mImpl->acceptMetadata(value.source);
        target.mImpl->acceptMetadata(value.target);
        visual_upper_bound = static_cast<Index>(value.visual_points);
        processed_count = static_cast<Index>(value.state.processed);
        if (value.status.code == static_cast<int>(Error::nonfinite_position) ||
            value.status.code == static_cast<int>(Error::coordinate_range) ||
            value.status.code == static_cast<int>(Error::trajectory))
            check(cudaMemsetAsync(status.data(), 0, sizeof(Status), stream.value), "reset rejected registration input");
        if (value.status.code != 0) prepared = false;
        throwStatus(value.status);
        FrameResult result;
        const State& current = value.state;
        result.registration.converged = current.converged != 0;
        result.registration.iterations = current.iterations;
        result.registration.num_source_points = static_cast<Index>(current.source_points);
        result.registration.num_target_points = static_cast<Index>(current.target_points);
        result.registration.num_invalid_points = static_cast<Index>(current.invalid);
        result.registration.num_correspondences = static_cast<Index>(current.correspondences);
        result.registration.fitness_score = current.fitness;
        for (int col = 0; col < 3; ++col)
            for (int row = 0; row < 3; ++row) result.registration.transform.linear()(row, col) = current.rotation[col * 3 + row];
        for (int row = 0; row < 3; ++row) result.registration.transform.translation()[row] = current.translation[row];
        result.num_processed_points = processed_count;
        result.num_invalid_points = static_cast<Index>(current.invalid);
        result.initialized = value.target.points != 0;
        result.map_updated = current.map_enabled != 0;
        if (download_scan)
        {
            result.scan_in_map.reserve(processed_count);
            for (Index i = 0; i < processed_count; ++i)
                result.scan_in_map.emplace_back(scan_host.data()[i].x(), scan_host.data()[i].y(), scan_host.data()[i].z());
        }
        if (download_map)
        {
            result.map.reserve(visual_upper_bound);
            for (Index i = 0; i < visual_upper_bound; ++i)
                result.map.emplace_back(map_host.data()[i].x(), map_host.data()[i].y(), map_host.data()[i].z());
        }
        return result;
    }
};

GICP::GICP() : GICP(Config{}) {}
GICP::GICP(const Config& config) : mConfig(config), mImpl(std::make_unique<Impl>()) {}
GICP::~GICP() = default;
GICP::GICP(GICP&&) noexcept = default;
GICP& GICP::operator=(GICP&&) noexcept = default;

GICP::Impl& GICP::deviceState() const
{
    if (!mImpl) mImpl = std::make_unique<Impl>();
    return *mImpl;
}

void GICP::insertTargetPoints(const pcl::PointCloud<pcl::PointXYZ>& points)
{
    if (points.empty()) return;
    Impl& device = deviceState();
    const std::lock_guard lock(device.mutex);
    device.prepareXYZ(points, false);
    setPoseKernel<<<1, 1, 0, device.stream.value>>>(
        device.state.data(), floatPoseValue(Eigen::Isometry3d::Identity()), device.target.mImpl->metadata.data(), true);
    check(cudaGetLastError(), "launch target insertion pose");
    device.insert();
    device.finish(false, false);
    device.prepared = false;
}

GICP::Result GICP::align(const pcl::PointCloud<pcl::PointXYZ>& source, const Eigen::Isometry3d& initial_guess) const
{
    if (source.empty()) throw std::invalid_argument("source cloud is empty!");
    validateConfig(mConfig);
    poseValue(initial_guess);
    Impl& device = deviceState();
    const std::lock_guard lock(device.mutex);
    device.prepareXYZ(source);
    device.optimize(initial_guess, mConfig, Mode::align_only, 0);
    return device.finish(false, false).registration;
}

void GICP::prepareSource(const pcl::PointCloud<pcl::PointXYZI>& source, const PreprocessConfig& config)
{
    Impl& device = deviceState();
    const std::lock_guard lock(device.mutex);
    device.prepare(source, config);
}

void GICP::prepareSource(const StampedCloud& source, const PreprocessConfig& config,
                         std::span<const TimedPose> trajectory, const Eigen::Isometry3d& lidar_to_imu)
{
    Impl& device = deviceState();
    const std::lock_guard lock(device.mutex);
    device.prepare(source, config, trajectory, lidar_to_imu);
}

GICP::Result GICP::alignPrepared(const Eigen::Isometry3d& initial_guess) const
{
    Impl& device = deviceState();
    const std::lock_guard lock(device.mutex);
    device.optimize(initial_guess, mConfig, Mode::align_only, 0);
    return device.finish(false, false).registration;
}

bool GICP::initializePreparedTarget(const Eigen::Isometry3d& pose, std::size_t minimum_points)
{
    Impl& device = deviceState();
    const std::lock_guard lock(device.mutex);
    device.optimize(pose, mConfig, Mode::initialize, minimum_points);
    device.insert();
    return device.finish(false, false).map_updated;
}

void GICP::insertPreparedTarget(const Eigen::Isometry3d& pose)
{
    Impl& device = deviceState();
    const std::lock_guard lock(device.mutex);
    if (!device.prepared) throw std::logic_error("No GPU scan has been prepared");
    setPoseKernel<<<1, 1, 0, device.stream.value>>>(
        device.state.data(), floatPoseValue(pose), device.target.mImpl->metadata.data(), true);
    check(cudaGetLastError(), "launch prepared map pose");
    device.insert();
}

std::size_t GICP::preparedPointCount() const noexcept
{
    return mImpl ? mImpl->processed_count : 0;
}

GICP::FrameResult GICP::downloadPrepared(const Eigen::Isometry3d& pose, bool download_scan, bool download_map)
{
    Impl& device = deviceState();
    const std::lock_guard lock(device.mutex);
    if (!device.prepared) throw std::logic_error("No GPU scan has been prepared");
    setPoseKernel<<<1, 1, 0, device.stream.value>>>(
        device.state.data(), floatPoseValue(pose), device.target.mImpl->metadata.data(), false);
    check(cudaGetLastError(), "launch published scan pose");
    return device.finish(download_scan, download_map);
}

GICP::FrameResult GICP::processFrame(const pcl::PointCloud<pcl::PointXYZI>& source,
                                   const Eigen::Isometry3d& initial_guess, const PreprocessConfig& config,
                                   bool download_scan, bool download_map, std::size_t minimum_initial_points)
{
    validateConfig(mConfig);
    poseValue(initial_guess);
    Impl& device = deviceState();
    const std::lock_guard lock(device.mutex);
    device.prepare(source, config);
    device.optimize(initial_guess, mConfig, Mode::frame, minimum_initial_points);
    device.insert();
    return device.finish(download_scan, download_map);
}
