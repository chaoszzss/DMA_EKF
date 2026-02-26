//
// Created by ubuntu on 25-3-1.
//

#include "../include/RosVisualizationBridge.hpp"

geometry_msgs::Point to_ros(const Eigen::Vector3d &pt) {
    geometry_msgs::Point ret;
    ret.x = pt.x();
    ret.y = pt.y();
    ret.z = pt.z();
    return ret;
}

RosViewer::RosViewer(ros::NodeHandle &n) : nh_(n) {
    pub_global_dash_pts_ = n.advertise<sensor_msgs::PointCloud>("global_dash_pts", 100, true);
    pub_global_arrow_pts_ = n.advertise<sensor_msgs::PointCloud>("global_arrow_pts", 100, true);
    pub_global_slot_pts_ = n.advertise<sensor_msgs::PointCloud>("global_slot_pts", 100, true);
    pub_slot_marker_ = n.advertise<visualization_msgs::Marker>("slot_vector", 100, true);
    pub_slot_marker_ipm_ = n.advertise<visualization_msgs::Marker>("slot_vector_ipm", 100, true);
    pub_current_pts_ = n.advertise<sensor_msgs::PointCloud>("current_pts", 100, true);
}

RosViewer::~RosViewer() {
    ros::shutdown();
}

void RosViewer::displayAvpMap(const Map &avp_map, double time) {
    // publish slot
    publishSlots(avp_map.getAllSlots(), time, pub_slot_marker_, true);
    // publish semantic points
    auto grid_slot = avp_map.getSemanticElement(SemanticLabel::kSlot);
    publishPoints({grid_slot.begin(), grid_slot.end()}, time, pub_global_slot_pts_);
    auto grid_arrow = avp_map.getSemanticElement(SemanticLabel::kArrowLine);
    publishPoints({grid_arrow.begin(), grid_arrow.end()}, time, pub_global_arrow_pts_);
    auto grid_dash = avp_map.getSemanticElement(SemanticLabel::kDashLine);
    publishPoints({grid_dash.begin(), grid_dash.end()}, time, pub_global_dash_pts_);

    ros::spinOnce();
}


void RosViewer::publishSlots(const std::vector<Slot> &slots, double time, ros::Publisher &publisher, bool avp_slots) {
    if (slots.empty()) {
        return;
    }
    visualization_msgs::Marker line_list;
    line_list.header.frame_id = "world";
    line_list.header.stamp = ros::Time(time);
    line_list.ns = "lines";
    line_list.action = visualization_msgs::Marker::ADD;
    line_list.pose.orientation.w = 1.0;
    line_list.type = visualization_msgs::Marker::LINE_LIST;
    line_list.scale.x = 0.1;
    line_list.id = 0;

    if (avp_slots) {
        line_list.color.g = 1.0;
    } else {
        line_list.color.r = 1.0;
    }
    line_list.color.a = 1.0;

    for (const auto &slot: slots) {
        auto p0 = to_ros(slot.corners_[0]);
        auto p1 = to_ros(slot.corners_[1]);
        auto p2 = to_ros(slot.corners_[2]);
        auto p3 = to_ros(slot.corners_[3]);
        line_list.points.push_back(p0);
        line_list.points.push_back(p1);
        line_list.points.push_back(p1);
        line_list.points.push_back(p2);
        line_list.points.push_back(p2);
        line_list.points.push_back(p3);
        line_list.points.push_back(p3);
        line_list.points.push_back(p0);
    }
    publisher.publish(line_list);
}


void RosViewer::publishPoints(const std::vector<Eigen::Vector3i> &grid, double time, ros::Publisher &publisher) {
    sensor_msgs::PointCloud global_cloud;
    global_cloud.header.frame_id = "world";
    global_cloud.header.stamp = ros::Time(time);
    // 设置点云的颜色
    
    for (Eigen::Vector3i pt: grid) {
        geometry_msgs::Point32 p;
        p.x = pt.x() * kPixelScale;
        p.y = pt.y() * kPixelScale;
        p.z = pt.z() * kPixelScale;
        global_cloud.points.push_back(p);
    }
    printf("publish points %d \n", global_cloud.points.size());
    publisher.publish(global_cloud);
}
