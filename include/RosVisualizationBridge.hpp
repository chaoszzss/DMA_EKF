//
// Created by ubuntu on 25-3-1.
//
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include "SystemPrimitives.hpp"

#ifndef AVP_WS_ROS_VIEWER_H
#define AVP_WS_ROS_VIEWER_H
class RosViewer {
public:
    RosViewer(ros::NodeHandle &n);
    ~RosViewer();
    void displayAvpMap(const Map &avp_map, double time);

    void publishPoints(const std::vector<Eigen::Vector3f> &points, double time);

private:
    void publishSlots(const std::vector<Slot> &slots, double time, ros::Publisher &publisher, bool avp_slots);

    void publishPoints(const std::vector<Eigen::Vector3i> &grid, double time, ros::Publisher &publisher);

    void publishPoints2(const std::vector<Eigen::Vector3i> &grid, double time, ros::Publisher &publisher);



    ros::NodeHandle &nh_;
    nav_msgs::Path path_, gt_path_, planned_path_, planned_A_path_, planned_hybridA_path_;
    ros::Publisher pub_path_, pub_odometry_, pub_mesh_, pub_gt_path_, pub_planned_path_, pub_planned_A_path_, pub_planned_hybridA_path_;
    ros::Publisher pub_global_dash_pts_, pub_global_arrow_pts_, pub_global_slot_pts_;
    ros::Publisher pub_current_pts_;
    ros::Publisher pub_slot_marker_, pub_slot_marker_ipm_;
};
#endif //AVP_WS_ROS_VIEWER_H
