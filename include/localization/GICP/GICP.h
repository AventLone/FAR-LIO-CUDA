#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "localization/types/SparseVoxel.cuh"
#include "localization/types/deskew.hpp"
#include <sophus/se3.hpp>

class GICP
{
public:
    struct Config
    {
        // 最近邻欧氏距离上限，单位 m，要求 > 0；只在 adjacent_voxels 覆盖的体素内查找。
        // 增大放宽匹配但增加误匹配风险；减小更严格、可能无对应点。单独增大不会扩大体素查询范围。
        float max_correspondence_distance{0.5f};

        std::size_t min_correspondences{300};
        double max_fitness_score{0.01};

        int max_iterations{100};
        double convergence_translation{1.0e-4};   // SE(3) 增量中平移分量的范数阈值，单位 m
        double convergence_rotation{1.0e-4};   // SE(3) 增量中旋转向量的范数阈值，单位 rad
    };

    struct Result
    {
        bool converged{false};
        int iterations{0};
        std::size_t num_source_points{0};
        std::size_t num_target_points{0};
        std::size_t num_invalid_points{0};
        std::size_t num_correspondences{0};
        double fitness_score{std::numeric_limits<double>::infinity()};
        Eigen::Isometry3d transform{Eigen::Isometry3d::Identity()};
    };

    struct PreprocessConfig
    {
        float voxel_size{0.1f};
        float intensity_threshold{-1.0f};
        float footprint_x{0.0f};
        float footprint_y{0.0f};
        float map_voxel_size{0.2f};
        Eigen::Isometry3f sensor_to_output{Eigen::Isometry3f::Identity()};
    };

    struct FrameResult
    {
        Result registration;
        std::size_t num_processed_points{0};
        std::size_t num_invalid_points{0};
        bool initialized{false};
        bool map_updated{false};
        pcl::PointCloud<pcl::PointXYZ> scan_in_map;
        pcl::PointCloud<pcl::PointXYZ> map;
    };

    GICP();
    explicit GICP(const Config& config);
    GICP(GICP&&) noexcept;
    GICP& operator=(GICP&&) noexcept;

    GICP(const GICP&) = delete;
    GICP& operator=(const GICP&) = delete;

    ~GICP();

    [[nodiscard]] const Config& config() const noexcept
    {
        return mConfig;
    }

    void setConfig(const Config& config) noexcept
    {
        mConfig = config;
    }

    void insertTargetPoints(const pcl::PointCloud<pcl::PointXYZ>& points);

    Result align(const pcl::PointCloud<pcl::PointXYZ>& source, const Eigen::Isometry3d& initial_guess) const;

    // Preparation and map insertion enqueue work on the same stream without downloading point data.
    void prepareSource(const pcl::PointCloud<pcl::PointXYZI>& source, const PreprocessConfig& config);
    void prepareSource(const StampedCloud& source, const PreprocessConfig& config,
                       std::span<const TimedPose> trajectory = {},
                       const Eigen::Isometry3d& lidar_to_imu = Eigen::Isometry3d::Identity());
    Result alignPrepared(const Eigen::Isometry3d& initial_guess) const;
    bool initializePreparedTarget(const Eigen::Isometry3d& pose, std::size_t minimum_points);
    void insertPreparedTarget(const Eigen::Isometry3d& pose);
    std::size_t preparedPointCount() const noexcept;

    FrameResult downloadPrepared(const Eigen::Isometry3d& pose, bool download_scan, bool download_map);
    FrameResult processFrame(const pcl::PointCloud<pcl::PointXYZI>& source, const Eigen::Isometry3d& initial_guess,
                             const PreprocessConfig& config, bool download_scan = false, bool download_map = false,
                             std::size_t minimum_initial_points = 101);

private:
    struct Impl;
    Config mConfig;
    mutable std::unique_ptr<Impl> mImpl;

    Impl& deviceState() const;
};