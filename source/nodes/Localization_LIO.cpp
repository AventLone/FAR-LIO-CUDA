#include "localization/nodes/Localization_LIO.h"
#include <pcl_conversions/pcl_conversions.h>
#include <algorithm>
#include <cmath>
#include <future>
#include <optional>

#include "localization/types/stamped_cloud.hpp"

namespace
{
struct IntensityAnalysis
{
    float minimum{};
    float p25{};
    float p40{};
    float median{};
    float p75{};
    float p90{};
    float p95{};
    float p99{};
    float maximum{};
    float suggested_threshold{};
    std::size_t valid_count{};
    std::size_t retained_count{};
};

std::optional<IntensityAnalysis> analyzeIntensity(const pcl::PointCloud<pcl::PointXYZI>& cloud, const float keep_ratio)
{
    std::vector<float> intensities;
    intensities.reserve(cloud.size());
    for (const auto& point : cloud)
    {
        if (std::isfinite(point.intensity))
        {
            intensities.push_back(point.intensity);
        }
    }

    if (intensities.empty())
    {
        return std::nullopt;
    }

    std::ranges::sort(intensities);
    const auto percentile = [&intensities](const double fraction)
        {
            const auto index = static_cast<std::size_t>(
                std::round(fraction * static_cast<double>(intensities.size() - 1)));
            return intensities[index];
        };

    const float minimum = intensities.front();
    const float maximum = intensities.back();
    const std::size_t target_retained_count = std::min(intensities.size(),
        static_cast<std::size_t>(std::ceil(static_cast<double>(keep_ratio) * intensities.size())));
    const std::size_t threshold_index = intensities.size() - target_retained_count;
    const float suggested_threshold = intensities[threshold_index];
    const auto first_retained = std::ranges::lower_bound(intensities, suggested_threshold);
    const std::size_t retained_count = static_cast<std::size_t>(intensities.end() - first_retained);
    return IntensityAnalysis{minimum, percentile(0.25), percentile(0.40), percentile(0.50), percentile(0.75),
                             percentile(0.90), percentile(0.95), percentile(0.99), maximum,
                             suggested_threshold, intensities.size(), retained_count};
}
}

void Localization_LIO::updateVoxelMap()
{
    mGicp.insertPreparedTarget(mESKF.pose());
}

GICP::PreprocessConfig Localization_LIO::preprocessing() const
{
    GICP::PreprocessConfig config;
    config.voxel_size = MAP_RESOLUTION;
    config.map_voxel_size = MAP_RESOLUTION * 2.0f;
    config.intensity_threshold = mIntensityThreshold;
    config.footprint_x = config.footprint_y = 0.8f;
    config.sensor_to_output = mT_il;
    return config;
}

bool Localization_LIO::alignScanToMap()
{
    const GICP::Result result = mGicp.alignPrepared(mESKF.pose());
    if (result.num_invalid_points != 0)
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "GPU filtering rejected %zu nonfinite points.",
                            result.num_invalid_points);

    if (result.num_source_points == 0)
    {
        RCLCPP_WARN(get_logger(), "GPU preprocessing retained no scan points.");
        return false;
    }
    // mESKF.observe(Sophus::SE3d(result.transform.rotation(), result.transform.translation()));
    // RCLCPP_INFO_STREAM_THROTTLE(get_logger(), *get_clock(), 1000, "Measure Cov" << result.measure_covariance);
    // mESKF.observe(Sophus::SE3d(result.transform.rotation(), result.transform.translation()), result.measure_covariance);
    mESKF.observe(Sophus::SE3d(result.transform.rotation(), result.transform.translation()));

    const auto& gicp_config = mGicp.config();
    if (result.num_correspondences < gicp_config.min_correspondences)
    {
        RCLCPP_WARN(get_logger(), "GICP converged with too few correspondences (%zu < %zu), skipping map update.",
                    result.num_correspondences, gicp_config.min_correspondences);
        return false;
    }
    if (result.fitness_score > gicp_config.max_fitness_score)
    {
        RCLCPP_WARN(get_logger(), "GICP converged but fitness score (%.4f) > threshold (%.4f), skipping map update.",
                    result.fitness_score, gicp_config.max_fitness_score);
        return false;
    }
    return true;
}

void Localization_LIO::workerLoop()
{
    StampedCloud stamped_cloud;
    double last_timestamp = 0.0;
    while (rclcpp::ok())
    {
        /* Wait untill received scan data */
        sensor_msgs::msg::PointCloud2::ConstSharedPtr scan_msg;
        {
            std::unique_lock lock(mScanBufferMutex);
            mTrigger.wait(lock, [this]() -> bool { return !mScanBuffer.empty() || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            scan_msg = std::move(mScanBuffer.front());
            mScanBuffer.pop_front();
        }

        const auto frame_start_time = std::chrono::high_resolution_clock::now();

        pcl::PointCloud<pcl::PointXYZI> lidar_points;
        if (!parseCloudMsg(*scan_msg, stamped_cloud))
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to parse point cloud message!");
            continue;
        }

        /* Initialize the whole system */
        if (!mInitialized)
        {
            mInitialized = true;
            last_timestamp = stamped_cloud.end_time;

            const auto cloud = getCloudXYZI(stamped_cloud.points);

            if (const auto analysis = analyzeIntensity(cloud, mIntensityKeepRatio))
            {
                const double retained_percentage = 100.0 * static_cast<double>(analysis->retained_count) /
                                                   static_cast<double>(analysis->valid_count);
                RCLCPP_INFO(get_logger(),
                            "First-frame intensity distribution (%zu valid points): min %.2f, P25 %.2f, "
                            "P40 %.2f, P50 %.2f, P75 %.2f, P90 %.2f, P95 %.2f, P99 %.2f, max %.2f.",
                            analysis->valid_count, analysis->minimum, analysis->p25, analysis->p40, analysis->median,
                            analysis->p75, analysis->p90, analysis->p95, analysis->p99, analysis->maximum);
                RCLCPP_INFO(get_logger(),
                            "Suggested intensity threshold: %.2f (target keep ratio %.1f%%, actual %.1f%%).",
                            analysis->suggested_threshold, 100.0 * mIntensityKeepRatio, retained_percentage);
                if (mIntensityThreshold < 0.0f)
                {
                    mIntensityThreshold = analysis->suggested_threshold;
                    RCLCPP_INFO(get_logger(), "Using automatically selected intensity threshold %.2f.",
                                mIntensityThreshold);
                }
                else
                {
                    RCLCPP_INFO(get_logger(), "Using configured intensity threshold %.2f.", mIntensityThreshold);
                }
            }
            else
            {
                RCLCPP_WARN(get_logger(), "First frame contains no finite intensity values; no threshold was selected.");
            }

            std::vector<lio::ImuData> imu_datas;
            {
                std::lock_guard<std::mutex> lock(mImuBufferMutex);
                imu_datas = std::vector(mImuBuffer.begin(), mImuBuffer.end());
            }
            if (!mESKF.initialize(imu_datas))
            {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "ESKF failed to initialize!");
                mInitialized = false;
                continue;
            }
            // mESKF.setTimestamp(stamped_cloud.end_time);

            try
            {
                mGicp.prepareSource(stamped_cloud, preprocessing());
                if (!mGicp.initializePreparedTarget(mESKF.pose(), 200))
                {
                    RCLCPP_WARN(get_logger(), "GPU preprocessing retained too few points to initialize the map.");
                    mInitialized = false;
                }
            }
            catch (const std::exception& exception)
            {
                RCLCPP_ERROR(get_logger(), "CUDA map initialization failed: %s", exception.what());
                mInitialized = false;
            }
            continue;
        }

        /* ESKF prediction */
        const auto last_state = mESKF.state();
        auto eskf_predict_future = std::async(std::launch::async,
            [this, last_timestamp, current_timestamp = stamped_cloud.end_time]()
            {
                std::vector<lio::ImuData> imu_datas;
                {
                    std::lock_guard<std::mutex> lock(mImuBufferMutex);
                    imu_datas = std::vector(mImuBuffer.begin(), mImuBuffer.end());
                }
                const auto sequence = lio::ImuProcessor::buildImuSequence(imu_datas, last_timestamp, current_timestamp);
                mESKF.predict(sequence);
            });
        last_timestamp = stamped_cloud.end_time;

        /* Deskew the point cloud and transform it into IMU frame */
        while (rclcpp::ok() && !mIsShutdown)
        {
            std::vector<lio::ImuData> imu_datas;
            {
                std::lock_guard<std::mutex> lock(mImuBufferMutex);
                imu_datas = std::vector(mImuBuffer.begin(), mImuBuffer.end());
            }
            try
            {
                mImuProcessor->prepareTrajectory(imu_datas, stamped_cloud, last_state);
            }
            catch (const std::exception& e)
            {
                // RCLCPP_WARN(get_logger(), e.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            break;
        }

        const auto trajectory = mImuProcessor->trajectory();
        try
        {
            mGicp.prepareSource(stamped_cloud, preprocessing(), trajectory,
                               Eigen::Isometry3d(mImuProcessor->extrinsic().matrix()));
        }
        catch (const std::exception& exception)
        {
            RCLCPP_ERROR(get_logger(), "CUDA scan preparation failed: %s", exception.what());
            eskf_predict_future.get();
            continue;
        }

        /* GICP and observe */
        eskf_predict_future.get();
        try
        {
            if (alignScanToMap()) updateVoxelMap();
        }
        catch (const std::exception& exception)
        {
            RCLCPP_ERROR(get_logger(), "Registration/map update failed: %s", exception.what());
            continue;
        }
        const bool publish_scan = mProcessedScanVisPub->get_subscription_count() != 0 ||
                                  mProcessedScanVisPub->get_intra_process_subscription_count() != 0;
        const bool publish_map = mVoxelMapPub->get_subscription_count() != 0 ||
                                 mVoxelMapPub->get_intra_process_subscription_count() != 0;
        if (publish_scan || publish_map)
        {
            GICP::FrameResult visualization;
            try
            {
                visualization = mGicp.downloadPrepared(mESKF.pose(), publish_scan, publish_map);
            }
            catch (const std::exception& exception)
            {
                RCLCPP_ERROR(get_logger(), "CUDA visualization download failed: %s", exception.what());
                continue;
            }
            if (publish_scan)
            {
                sensor_msgs::msg::PointCloud2 scan_vis_msg;
                pcl::toROSMsg(visualization.scan_in_map, scan_vis_msg);
                scan_vis_msg.header.stamp = this->now();
                scan_vis_msg.header.frame_id = "map";
                mProcessedScanVisPub->publish(scan_vis_msg);
            }
            if (publish_map)
            {
                sensor_msgs::msg::PointCloud2 map_msg;
                pcl::toROSMsg(visualization.map, map_msg);
                map_msg.header.stamp = this->now();
                map_msg.header.frame_id = "map";
                mVoxelMapPub->publish(map_msg);
            }
        }

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = rclcpp::Time(static_cast<int64_t>(stamped_cloud.end_time * 1e9));
        pose_msg.header.frame_id = "map";
        pose_msg.pose = tf2::toMsg(mESKF.pose());
        mBasePosePub->publish(pose_msg);

        mBasePosePath.header = pose_msg.header;
        mBasePosePath.poses.push_back(pose_msg);
        mBasePosePathPub->publish(mBasePosePath);

        const auto frame_end_time = std::chrono::high_resolution_clock::now();
        const double frame_duration_ms = std::chrono::duration<double, std::milli>(frame_end_time - frame_start_time).count();
        RCLCPP_INFO(get_logger(), "Whole frame processing time: %.2f ms", frame_duration_ms);
    }
}