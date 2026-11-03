#pragma once
#include <rclcpp/qos.hpp>

inline const rclcpp::QoS SENSOR_DATA = rclcpp::SensorDataQoS();
inline const rclcpp::QoS SENSOR_LATEST = rclcpp::SensorDataQoS(rclcpp::KeepLast(1));
inline const rclcpp::QoS RELIABLE = rclcpp::QoS(10).reliable().durability_volatile();
inline const rclcpp::QoS RELIABLE_LATEST = rclcpp::QoS(1).reliable().durability_volatile();
inline const rclcpp::QoS LATCHED_STATE = rclcpp::QoS(1).reliable().transient_local();

