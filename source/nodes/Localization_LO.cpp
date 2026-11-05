#include "localization/nodes/Localization_LO.h"
#include <pcl_conversions/pcl_conversions.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

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

void Localization_LO::recordGicpDuration(const double duration_ms)
{
    if (constexpr std::size_t statistics_window_size = 200;
        mGicpDurationsMs.size() == statistics_window_size)
    {
        mGicpDurationsMs.erase(mGicpDurationsMs.begin());
    }
    mGicpDurationsMs.push_back(duration_ms);
    ++mGicpDurationCount;

    if (constexpr std::size_t report_interval = 50;
        mGicpDurationsMs.size() < report_interval || mGicpDurationCount % report_interval != 0)
    {
        return;
    }

    std::vector<double> sorted_durations = mGicpDurationsMs;
    std::ranges::sort(sorted_durations);
    const auto percentile = [&sorted_durations](const double fraction)
        {
            const std::size_t index = static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(sorted_durations.size()))) - 1;
            return sorted_durations[std::min(index, sorted_durations.size() - 1)];
        };
    const double mean = std::accumulate(mGicpDurationsMs.begin(), mGicpDurationsMs.end(), 0.0) /
                        static_cast<double>(mGicpDurationsMs.size());

    RCLCPP_INFO(get_logger(),
                "CUDA scan pipeline latency (%zu-frame window): mean %.2f ms, P50 %.2f ms, P95 %.2f ms, P99 %.2f ms, max %.2f ms",
                mGicpDurationsMs.size(), mean, percentile(0.50), percentile(0.95), percentile(0.99), sorted_durations.back());
}

void Localization_LO::workerLoop()
{
    while (rclcpp::ok())
    {
        /* Wait and receive scan data */
        sensor_msgs::msg::PointCloud2::ConstSharedPtr scan_msg;
        {
            std::unique_lock lock(mScanBufferMutex);
            mTrigger.wait(lock, [this]() -> bool { return !mScanBuffer.empty() || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            scan_msg = std::move(mScanBuffer.front());
            mScanBuffer.pop();
        }

        const auto frame_start_time = std::chrono::high_resolution_clock::now();

        pcl::PointCloud<pcl::PointXYZI> lidar_points;
        pcl::fromROSMsg(*scan_msg, lidar_points);

        // pcl::transformPointCloud(lidar_points, lidar_points, T_truck2lidar);

        if (!mIntensityAnalyzed)
        {
            mIntensityAnalyzed = true;
            if (const auto analysis = analyzeIntensity(lidar_points, mIntensityKeepRatio))
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
                    RCLCPP_INFO(get_logger(), "Using automatically selected intensity threshold %.2f.", mIntensityThreshold);
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
        }

        const bool initializing = !mMapInitialized;
        const bool publish_scan = !initializing && (mProcessedScanVisPub->get_subscription_count() != 0 ||
                                                   mProcessedScanVisPub->get_intra_process_subscription_count() != 0);
        const bool publish_map = !initializing && (mVoxelMapPub->get_subscription_count() != 0 ||
                                                  mVoxelMapPub->get_intra_process_subscription_count() != 0);
        GICP::PreprocessConfig preprocessing;
        preprocessing.voxel_size = MAP_RESOLUTION;
        preprocessing.map_voxel_size = MAP_RESOLUTION * 2.0f;
        preprocessing.intensity_threshold = mIntensityThreshold;
        preprocessing.footprint_x = 1.8f;
        preprocessing.footprint_y = 0.8f;
        const auto pipeline_start = std::chrono::steady_clock::now();
        GICP::FrameResult frame;
        try
        {
            frame = mGicp.processFrame(lidar_points, mBasePose, preprocessing, publish_scan, publish_map);
        }
        catch (const std::exception& exception)
        {
            RCLCPP_ERROR(get_logger(), "CUDA scan pipeline failed: %s", exception.what());
            continue;
        }
        if (frame.num_invalid_points != 0)
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "GPU filtering rejected %zu nonfinite points.",
                                frame.num_invalid_points);
        mMapInitialized = frame.initialized;
        if (initializing)
        {
            if (!mMapInitialized) RCLCPP_WARN(get_logger(), "GPU preprocessing retained too few points to initialize the map.");
            continue;
        }
        if (frame.num_processed_points == 0)
        {
            RCLCPP_WARN(get_logger(), "Skipping GICP: GPU filtering retained no scan points.");
            continue;
        }
        const GICP::Result& result = frame.registration;
        const auto& config = mGicp.config();
        if (result.converged && result.num_correspondences >= config.min_correspondences)
        {
            mBasePose = result.transform;
            if (result.fitness_score > config.max_fitness_score)
                RCLCPP_WARN(get_logger(), "GICP fitness %.4f > %.4f; GPU map update skipped.",
                            result.fitness_score, config.max_fitness_score);
        }
        else
            RCLCPP_WARN(get_logger(), "GICP rejected: iterations %d, correspondences %zu/%zu, fitness %.6f.",
                        result.iterations, result.num_correspondences, result.num_source_points, result.fitness_score);
        if constexpr (PRINT_BENCHMARK)
        {
            const double duration_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - pipeline_start).count();
            RCLCPP_INFO(get_logger(), "CUDA scan pipeline: %.2f ms, iterations %d, correspondences %zu/%zu, fitness %.6f.",
                        duration_ms, result.iterations, result.num_correspondences, result.num_source_points,
                        result.fitness_score);
            recordGicpDuration(duration_ms);
        }
        if (publish_scan)
        {
            sensor_msgs::msg::PointCloud2 scan_vis_msg;
            pcl::toROSMsg(frame.scan_in_map, scan_vis_msg);
            scan_vis_msg.header.stamp = this->now();
            scan_vis_msg.header.frame_id = "map";
            mProcessedScanVisPub->publish(scan_vis_msg);
        }
        if (publish_map)
        {
            sensor_msgs::msg::PointCloud2 map_msg;
            pcl::toROSMsg(frame.map, map_msg);
            map_msg.header.stamp = this->now();
            map_msg.header.frame_id = "map";
            mVoxelMapPub->publish(map_msg);
        }

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = scan_msg->header;
        pose_msg.header.frame_id = "map";
        pose_msg.pose = tf2::toMsg(mBasePose.cast<double>());
        mBasePosePub->publish(pose_msg);

        geometry_msgs::msg::TransformStamped tf;
        // tf.header.stamp = scan_msg->header.stamp;
        tf.header.stamp = this->get_clock()->now();
        tf.header.frame_id = "map";
        tf.child_frame_id = "LOLA";
        tf.transform.translation.x = pose_msg.pose.position.x;
        tf.transform.translation.y = pose_msg.pose.position.y;
        tf.transform.translation.z = pose_msg.pose.position.z;
        tf.transform.rotation.x = pose_msg.pose.orientation.x;
        tf.transform.rotation.y = pose_msg.pose.orientation.y;
        tf.transform.rotation.z = pose_msg.pose.orientation.z;
        tf.transform.rotation.w = pose_msg.pose.orientation.w;
        mTfBroadcaster->sendTransform(tf);

        mBasePosePath.header = pose_msg.header;
        mBasePosePath.poses.push_back(pose_msg);
        mBasePosePathPub->publish(mBasePosePath);

        const auto frame_end_time = std::chrono::high_resolution_clock::now();
        const double frame_duration_ms = std::chrono::duration<double, std::milli>(frame_end_time - frame_start_time).count();

        if constexpr (PRINT_BENCHMARK)
        {
            RCLCPP_INFO(get_logger(), "Whole frame processing time: %.2f ms", frame_duration_ms);
        }
    }
}