#pragma once
#include <Eigen/Geometry>
#include <vector>

struct PointXYZIT
{
    Eigen::Vector3f position;
    float intensity;
    double time_offset;   // Seconds relative to scan begin.
};

struct StampedCloud
{
    double begin_time, end_time;   // Scan timestamps in seconds.
    std::vector<PointXYZIT> points;
};

struct TimedPose
{
    double timestamp{0.0};   // Same clock and seconds as StampedCloud.
    Eigen::Quaterniond rotation{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
};
