#include "localization/GICP/GICP.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <pcl/registration/gicp.h>
#include <tbb/global_control.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using Clock = std::chrono::steady_clock;
constexpr int MAX_ITERATIONS = 100;
constexpr int COVARIANCE_NEIGHBORS = SparseVoxel::MAX_COVARIANCE_NEIGHBORS;
constexpr double CORRESPONDENCE_DISTANCE = 0.5;
constexpr double CONVERGENCE_EPSILON = 1e-4;
constexpr double RAD_TO_DEG = 180.0 / std::numbers::pi;

struct Scenario
{
    std::string name;
    std::size_t points;
    double noise;
    double overlap;
};

struct Dataset
{
    Cloud::Ptr source{new Cloud};
    Cloud::Ptr target{new Cloud};
    std::vector<Eigen::Vector3d> reference;
    std::size_t overlapping_points{0};
    Eigen::Isometry3d truth{Eigen::Isometry3d::Identity()};
};

struct Measurement
{
    Eigen::Isometry3d transform{Eigen::Isometry3d::Identity()};
    double target_ms{0.0};
    double align_ms{0.0};
    std::size_t source_points{0};
    std::size_t target_points{0};
    int iterations{0};
    bool reported_converged{false};
};

struct Accuracy
{
    double translation_mm;
    double rotation_deg;
    double reference_rmse_mm;
    double nn_rmse_mm;
    double oracle_rmse_mm;
    bool success;
};

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

double milliseconds(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

Eigen::Vector3d surfacePoint(std::mt19937& random, double minimum_x, double maximum_x)
{
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const double x = minimum_x + (maximum_x - minimum_x) * unit(random);
    const double y = 2.6 * (unit(random) - 0.5);
    const double z = 2.8 * unit(random);
    switch (std::uniform_int_distribution<int>(0, 5)(random))
    {
    case 0:
        return {x, y, 0.0};
    case 1:
        return {x, y, 2.8};
    case 2:
        return {x, -1.3, z};
    case 3:
        return {x, 1.3, z};
    case 4:
        return {x, 0.65 * y, 0.8 + 0.25 * std::sin(1.7 * x) + 0.15 * y};
    default:
        const double angle = 2.0 * std::numbers::pi * unit(random);
        const double radius = 0.45 + 0.12 * std::sin(2.3 * x);
        return {x, radius * std::cos(angle), 1.65 + radius * std::sin(angle)};
    }
}

Dataset makeDataset(const Scenario& scenario, std::uint32_t seed)
{
    require(scenario.points >= COVARIANCE_NEIGHBORS && scenario.noise >= 0.0 &&
            scenario.overlap > 0.0 && scenario.overlap <= 1.0, "Invalid synthetic scenario");
    Dataset data;
    std::mt19937 random(seed);
    std::normal_distribution<double> noise(0.0, 1.0);
    const auto perturb = [&](Eigen::Vector3d position)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            position[axis] += scenario.noise * noise(random);
        }
        return position;
    };
    data.truth.linear() = (Eigen::AngleAxisd(3.0 / RAD_TO_DEG, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(-1.5 / RAD_TO_DEG, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(1.0 / RAD_TO_DEG, Eigen::Vector3d::UnitX())).toRotationMatrix();
    data.truth.translation() = Eigen::Vector3d(0.18, -0.08, 0.04);
    data.overlapping_points = static_cast<std::size_t>(std::llround(
        scenario.overlap * static_cast<double>(scenario.points)));
    data.source->reserve(scenario.points);
    data.target->reserve(scenario.points);
    data.reference.reserve(scenario.points);
    for (std::size_t index = 0; index < scenario.points; ++index)
    {
        const Eigen::Vector3d point = perturb(surfacePoint(random, 0.0, 8.0));
        data.target->emplace_back(static_cast<float>(point.x()), static_cast<float>(point.y()),
                                 static_cast<float>(point.z()));
    }
    const double crop_start = 8.0 * (1.0 - scenario.overlap);
    const Eigen::Isometry3d inverse = data.truth.inverse();
    for (std::size_t index = 0; index < scenario.points; ++index)
    {
        const Eigen::Vector3d reference = index < data.overlapping_points
            ? surfacePoint(random, crop_start, 8.0)
            : surfacePoint(random, 8.0, 8.0 + crop_start);
        const Eigen::Vector3d point = perturb(inverse * reference);
        data.source->emplace_back(static_cast<float>(point.x()), static_cast<float>(point.y()),
                                 static_cast<float>(point.z()));
        data.reference.push_back(reference);
    }
    // Both algorithms see the same order, without grouping the non-overlapping points.
    std::vector<std::size_t> order(scenario.points);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), random);
    Cloud shuffled;
    shuffled.reserve(scenario.points);
    std::vector<Eigen::Vector3d> reference;
    reference.reserve(scenario.points);
    for (const std::size_t index : order)
    {
        shuffled.push_back((*data.source)[index]);
        reference.push_back(data.reference[index]);
    }
    *data.source = std::move(shuffled);
    data.reference = std::move(reference);
    return data;
}

Cloud::Ptr sparseCloud(const Cloud& input)
{
    SparseVoxel index;
    require(index.insert(input), "Cannot sparsify empty input");
    Cloud::Ptr result(new Cloud);
    result->reserve(index.pointCount());
    for (const PointWithCovariance* point : index.points())
    {
        result->emplace_back(point->position.x(), point->position.y(), point->position.z());
    }
    return result;
}

void configurePcl(pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>& registration)
{
    registration.setMaximumIterations(MAX_ITERATIONS);
    registration.setMaxCorrespondenceDistance(CORRESPONDENCE_DISTANCE);
    registration.setTransformationEpsilon(CONVERGENCE_EPSILON);
    registration.setRotationEpsilon(CONVERGENCE_EPSILON);
    registration.setCorrespondenceRandomness(COVARIANCE_NEIGHBORS);
}

class PreparedPclGicp : public pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>
{
public:
    void prepareTarget(const Cloud::ConstPtr& target)
    {
        setInputTarget(target);
        auto tree = pcl::make_shared<InputKdTree>();
        tree->setInputCloud(target);
        setSearchMethodTarget(tree, true);
        auto covariances = pcl::make_shared<MatricesVector>();
        computeCovariances<pcl::PointXYZ>(target, tree, *covariances);
        setTargetCovariances(covariances);
    }

    int iterations() const
    {
        return nr_iterations_;
    }

    const MatricesVector& targetCovariances() const
    {
        require(target_covariances_ != nullptr, "PCL target covariances are not prepared");
        return *target_covariances_;
    }
};

void checkCovarianceSpectrum(const PointWithCovariance& point)
{
    require(point.covariance_valid, "Supported neighborhood must have a valid covariance");
    const auto& covariance = point.covariance;
    require(covariance.allFinite() && covariance.isApprox(covariance.transpose(), 1e-6f),
            "GICP covariance must be finite and symmetric");
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> decomposition(covariance);
    require(decomposition.info() == Eigen::Success &&
            decomposition.eigenvalues().isApprox(Eigen::Vector3f(0.001f, 1.0f, 1.0f), 1e-5f),
            "GICP covariance spectrum must match PCL's (0.001, 1, 1)");
    require(Eigen::LLT<Eigen::Matrix3f>(covariance).info() == Eigen::Success,
            "Normalized GICP covariance must be positive definite");
}

void checkCovarianceModel()
{
    const Eigen::Matrix3f rotation = (Eigen::AngleAxisf(0.4f, Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(-0.2f, Eigen::Vector3f::UnitY())).toRotationMatrix();
    const Eigen::Vector3f normal = rotation.col(2);
    for (const float normal_noise : {0.0f, 0.02f})
    {
        Cloud::Ptr cloud(new Cloud);
        const int width = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(COVARIANCE_NEIGHBORS))));
        const float center = 0.5f * static_cast<float>(width - 1);
        for (int x = 0; x < width; ++x)
        {
            for (int y = 0; y < width && cloud->size() < static_cast<std::size_t>(COVARIANCE_NEIGHBORS); ++y)
            {
                const Eigen::Vector3f point = rotation * Eigen::Vector3f(
                    0.12f * (static_cast<float>(x) - center),
                    0.12f * (static_cast<float>(y) - center),
                    normal_noise * std::sin(1.1f * static_cast<float>(x) + 0.8f * static_cast<float>(y)));
                cloud->emplace_back(point.x(), point.y(), point.z());
            }
        }
        PreparedPclGicp reference;
        configurePcl(reference);
        reference.prepareTarget(cloud);
        const Eigen::Matrix3f expected = reference.targetCovariances().front().cast<float>();
        SparseVoxel index;
        require(index.insert(*cloud) && index.pointCount() == cloud->size(),
                "Covariance fixture must retain all configured neighbors");
        for (const PointWithCovariance* point : index.points())
        {
            require(index.nearestNeighbors(point->position, COVARIANCE_NEIGHBORS).size() == cloud->size(),
                    "PCL and voxel covariance fixtures must use the same neighborhood");
            checkCovarianceSpectrum(*point);
            require(point->covariance.isApprox(expected, 2e-5f),
                    "Voxel covariance must match PCL for the same planar or noisy neighborhood");
            if (normal_noise == 0.0f)
            {
                require((point->covariance * normal - 0.001f * normal).norm() < 1e-5f,
                        "Normalization must preserve the plane normal");
            }
        }

        Cloud first;
        Cloud remaining;
        for (std::size_t i = 0; i < cloud->size(); ++i)
        {
            (i < static_cast<std::size_t>(SparseVoxel::MIN_COVARIANCE_NEIGHBORS - 1)
                ? first : remaining).push_back((*cloud)[i]);
        }
        SparseVoxel incremental;
        require(incremental.insert(first) && incremental.pointCount() == first.size(),
                "Sparse covariance fixture insertion failed");
        for (const PointWithCovariance* point : incremental.points())
        {
            require(!point->covariance_valid && point->covariance.isIdentity(),
                    "Insufficient neighbors must preserve the invalid identity fallback");
        }
        require(incremental.insert(remaining) && incremental.pointCount() == cloud->size(),
                "Incremental insertion must preserve every fixture point");
        for (const PointWithCovariance* point : incremental.points())
        {
            checkCovarianceSpectrum(*point);
            require(point->covariance.isApprox(expected, 2e-5f),
                    "Incremental insertion must refresh old and new covariance weights");
        }
        incremental.clear();
        require(incremental.empty() && incremental.insert(first), "Cleared voxel index must be reusable");
        for (const PointWithCovariance* point : incremental.points())
        {
            require(!point->covariance_valid && point->covariance.isIdentity(),
                    "Clearing must not retain old normalized covariances");
        }
    }
}

Measurement runCustom(const Cloud& source, const Cloud& target)
{
    GICP::Config config;
    config.max_iterations = MAX_ITERATIONS;
    config.max_correspondence_distance = static_cast<float>(CORRESPONDENCE_DISTANCE);
    config.convergence_translation = CONVERGENCE_EPSILON;
    config.convergence_rotation = CONVERGENCE_EPSILON;
    GICP registration(config);
    const auto start = Clock::now();
    registration.insertTargetPoints(target);
    const auto prepared = Clock::now();
    const GICP::Result result = registration.align(source, Eigen::Isometry3d::Identity());
    const auto aligned = Clock::now();
    return {result.transform, milliseconds(start, prepared), milliseconds(prepared, aligned),
            result.num_source_points, result.num_target_points, result.iterations, result.converged};
}

Measurement runPcl(const Cloud::ConstPtr& source, const Cloud::ConstPtr& target)
{
    PreparedPclGicp registration;
    configurePcl(registration);
    const auto start = Clock::now();
    registration.prepareTarget(target);
    const auto prepared = Clock::now();
    registration.setInputSource(source);
    Cloud output;
    registration.align(output, Eigen::Matrix4f::Identity());
    const auto aligned = Clock::now();
    Eigen::Isometry3d transform;
    transform.matrix() = registration.getFinalTransformation().cast<double>();
    return {transform, milliseconds(start, prepared), milliseconds(prepared, aligned),
            source->size(), target->size(), registration.iterations(), registration.hasConverged()};
}

Accuracy evaluate(const Dataset& data, pcl::search::KdTree<pcl::PointXYZ>& target_tree,
                  const Eigen::Isometry3d& transform)
{
    require(transform.matrix().allFinite(), "Registration produced a non-finite transform");
    const double translation_mm = 1000.0 * (transform.translation() - data.truth.translation()).norm();
    Eigen::Quaterniond rotation(transform.rotation() * data.truth.rotation().transpose());
    rotation.normalize();
    const double rotation_deg = 2.0 * std::atan2(rotation.vec().norm(), std::abs(rotation.w())) * RAD_TO_DEG;
    double reference_error = 0.0;
    double nn_error = 0.0;
    double oracle_error = 0.0;
    std::size_t inliers = 0;
    pcl::Indices neighbors(1);
    std::vector<float> distances(1);
    for (std::size_t index = 0; index < data.source->size(); ++index)
    {
        // The overlap mask comes from noiseless geometry, not an algorithm's distance gate.
        if (data.reference[index].x() > 8.0)
        {
            continue;
        }
        const Eigen::Vector3d point = (*data.source)[index].getVector3fMap().cast<double>();
        const Eigen::Vector3d aligned = transform * point;
        reference_error += (aligned - data.reference[index]).squaredNorm();
        oracle_error += (data.truth * point - data.reference[index]).squaredNorm();
        const pcl::PointXYZ query(static_cast<float>(aligned.x()), static_cast<float>(aligned.y()),
                                  static_cast<float>(aligned.z()));
        require(target_tree.nearestKSearch(query, 1, neighbors, distances) == 1, "Evaluation NN search failed");
        nn_error += static_cast<double>(distances.front());
        ++inliers;
    }
    require(inliers == data.overlapping_points && inliers > 0, "Synthetic overlap mask is inconsistent");
    const double count = static_cast<double>(inliers);
    return {translation_mm, rotation_deg, 1000.0 * std::sqrt(reference_error / count),
            1000.0 * std::sqrt(nn_error / count), 1000.0 * std::sqrt(oracle_error / count),
            translation_mm < 20.0 && rotation_deg < 0.2};
}

void checkBenchmark(const Dataset& data, pcl::search::KdTree<pcl::PointXYZ>& tree)
{
    checkCovarianceModel();
    const Accuracy oracle = evaluate(data, tree, data.truth);
    require(oracle.translation_mm < 1e-9 && oracle.rotation_deg < 1e-9 &&
            std::abs(oracle.reference_rmse_mm - oracle.oracle_rmse_mm) < 1e-9,
            "Ground-truth evaluation is incorrect");
    Eigen::Isometry3d offset = data.truth;
    offset.translation().x() += 0.03;
    offset.linear() = Eigen::AngleAxisd(1.0 / RAD_TO_DEG, Eigen::Vector3d::UnitZ()) * offset.rotation();
    const Accuracy displaced = evaluate(data, tree, offset);
    require(std::abs(displaced.translation_mm - 30.0) < 1e-9 &&
            std::abs(displaced.rotation_deg - 1.0) < 1e-9 && !displaced.success,
            "Pose-error units or success thresholds are incorrect");
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> ordinary;
    configurePcl(ordinary);
    ordinary.setInputTarget(data.target);
    ordinary.setInputSource(data.source);
    Cloud output;
    ordinary.align(output);
    const Measurement prepared = runPcl(data.source, data.target);
    require(prepared.transform.matrix().isApprox(ordinary.getFinalTransformation().cast<double>(), 1e-5),
            "Moving PCL target preparation outside align changed the result");
}

double median(std::vector<double> values)
{
    require(!values.empty(), "Cannot summarize empty measurements");
    std::ranges::sort(values);
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

struct Summary
{
    std::vector<double> target_ms;
    std::vector<double> align_ms;
    std::vector<double> translation_mm;
    std::vector<double> rotation_deg;
    int successes{0};

    void add(const Measurement& measurement, const Accuracy& accuracy)
    {
        target_ms.push_back(measurement.target_ms);
        align_ms.push_back(measurement.align_ms);
        translation_mm.push_back(accuracy.translation_mm);
        rotation_deg.push_back(accuracy.rotation_deg);
        successes += accuracy.success ? 1 : 0;
    }
};

int positiveInteger(const std::string& text)
{
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    require(consumed == text.size() && value > 0, "Expected a positive integer: " + text);
    return value;
}
}

int main(int argc, char** argv)
{
    try
    {
        int repeats = 5;
        int threads = 8;
        bool quick = false;
        std::string csv_path;
        for (int index = 1; index < argc; ++index)
        {
            const std::string option = argv[index];
            if (option == "--quick")
            {
                quick = true;
                continue;
            }
            require(index + 1 < argc, "Missing value for " + option);
            const std::string value = argv[++index];
            if (option == "--repeats")
            {
                repeats = positiveInteger(value);
            }
            else if (option == "--threads")
            {
                threads = positiveInteger(value);
            }
            else if (option == "--csv")
            {
                csv_path = value;
            }
            else
            {
                throw std::invalid_argument("Unknown option: " + option);
            }
        }
        require(threads > 1, "--threads must exceed 1; the single-thread control is always included");
        std::ofstream csv;
        if (!csv_path.empty())
        {
            csv.open(csv_path);
            require(csv.is_open(), "Cannot open CSV output: " + csv_path);
            csv << "scenario,points,noise_m,overlap,seed,input_mode,algorithm,threads,"
                   "input_source_points,input_target_points,used_source_points,used_target_points,"
                   "target_ms,align_ms,total_ms,iterations,reported_converged,reached_iteration_limit,"
                   "translation_mm,rotation_deg,reference_rmse_mm,nn_rmse_mm,oracle_rmse_mm,success\n";
            csv << std::setprecision(12);
        }
        const std::vector<Scenario> scenarios = quick
            ? std::vector<Scenario>{{"smoke", 1200, 0.005, 1.0}}
            : std::vector<Scenario>{{"small", 3000, 0.005, 1.0},
                                    {"medium", 12000, 0.005, 1.0},
                                    {"large", 40000, 0.005, 1.0},
                                    {"noiseless", 12000, 0.0, 1.0},
                                    {"noise_20mm", 12000, 0.020, 1.0},
                                    {"overlap_70", 12000, 0.005, 0.7},
                                    {"overlap_40", 12000, 0.005, 0.4}};
        std::cout << "PCL " << PCL_VERSION_PRETTY << "; max_iterations=" << MAX_ITERATIONS
                  << "; max_distance_m=" << CORRESPONDENCE_DISTANCE << "; PCL covariance k="
                  << COVARIANCE_NEIGHBORS << "; initial_guess=identity; success: <20mm and <0.2deg\n"
                  << "Target preparation includes index/covariances; align includes source preparation.\n"
                  << "One unrecorded warmup per group. Timings exclude data generation/evaluation/shared sampling.\n"
                  << "scenario,mode,algorithm,threads,target_p50_ms,align_p50_ms,t_p50_mm,r_p50_deg,success\n";
        std::cout << std::fixed << std::setprecision(4);
        for (const Scenario& scenario : scenarios)
        {
            std::array<std::array<Summary, 3>, 2> summaries;
            for (int trial = -1; trial < repeats; ++trial)
            {
                const std::uint32_t seed = trial < 0 ? 999U : 1000U + static_cast<std::uint32_t>(trial);
                const Dataset data = makeDataset(scenario, seed);
                pcl::search::KdTree<pcl::PointXYZ> evaluation_tree;
                evaluation_tree.setInputCloud(data.target);
                if (quick && trial == -1)
                {
                    const Dataset repeated = makeDataset(scenario, seed);
                    require(std::equal(data.source->begin(), data.source->end(), repeated.source->begin(),
                        [](const pcl::PointXYZ& lhs, const pcl::PointXYZ& rhs)
                        {
                            return (lhs.getVector3fMap().array() == rhs.getVector3fMap().array()).all();
                        }), "Seeded data generation is not deterministic");
                    checkBenchmark(data, evaluation_tree);
                }
                for (std::size_t mode = 0; mode < 2; ++mode)
                {
                    Cloud::Ptr source = data.source;
                    Cloud::Ptr target = data.target;
                    if (mode == 1)
                    {
                        tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                                    static_cast<std::size_t>(threads));
                        source = sparseCloud(*data.source);
                        target = sparseCloud(*data.target);
                    }
                    // Rotate execution order so one implementation is not always measured first.
                    for (int position = 0; position < 3; ++position)
                    {
                        const std::size_t algorithm = static_cast<std::size_t>((position + trial + 3) % 3);
                        const bool pcl = algorithm == 0;
                        const int budget = algorithm == 2 ? threads : 1;
                        tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                                    static_cast<std::size_t>(budget));
                        const Measurement measurement = pcl ? runPcl(source, target) : runCustom(*source, *target);
                        if (mode == 1)
                        {
                            require(measurement.source_points == source->size() &&
                                    measurement.target_points == target->size(),
                                    "Matched input was downsampled again; point counts are not comparable");
                        }
                        const Accuracy accuracy = evaluate(data, evaluation_tree, measurement.transform);
                        if (trial < 0)
                        {
                            continue;
                        }
                        summaries[mode][algorithm].add(measurement, accuracy);
                        if (csv.is_open())
                        {
                            csv << scenario.name << ',' << scenario.points << ',' << scenario.noise << ','
                                << scenario.overlap << ',' << seed << ',' << (mode == 0 ? "raw" : "matched_sparse")
                                << ',' << (pcl ? "pcl" : "custom") << ',' << budget << ','
                                << source->size() << ',' << target->size() << ',' << measurement.source_points << ','
                                << measurement.target_points << ',' << measurement.target_ms << ','
                                << measurement.align_ms << ',' << measurement.target_ms + measurement.align_ms << ','
                                << measurement.iterations << ',' << measurement.reported_converged << ','
                                << (measurement.iterations >= MAX_ITERATIONS) << ',' << accuracy.translation_mm << ','
                                << accuracy.rotation_deg << ',' << accuracy.reference_rmse_mm << ','
                                << accuracy.nn_rmse_mm << ',' << accuracy.oracle_rmse_mm << ',' << accuracy.success << '\n';
                            require(csv.good(), "Cannot write CSV output");
                        }
                    }
                }
            }
            for (std::size_t mode = 0; mode < 2; ++mode)
            {
                for (std::size_t algorithm = 0; algorithm < 3; ++algorithm)
                {
                    const Summary& summary = summaries[mode][algorithm];
                    std::cout << scenario.name << ',' << (mode == 0 ? "raw" : "matched_sparse") << ','
                              << (algorithm == 0 ? "pcl" : "custom") << ',' << (algorithm == 2 ? threads : 1)
                              << ',' << median(summary.target_ms) << ',' << median(summary.align_ms) << ','
                              << median(summary.translation_mm) << ',' << median(summary.rotation_deg) << ','
                              << summary.successes << '/' << repeats << std::endl;
                }
            }
        }
        if (csv.is_open())
        {
            csv.flush();
            require(csv.good(), "Cannot flush CSV output");
        }
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "GICP benchmark failed: " << exception.what() << '\n';
        return 1;
    }
}
