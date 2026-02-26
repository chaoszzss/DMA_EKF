//
// Created by ubuntu on 2024-12-31.
// Tong Qin: qintong@sjtu.edu.cn
//

#include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ros/ros.h"
#include "visualization_msgs/Marker.h"
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/transform_broadcaster.h>

#include "../include/LocalizationEngine.hpp"
#include "RosVisualizationBridge.hpp"

#include <lgsvl_msgs/CanBusData.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std;

ros::Publisher pub_estimation_path, pub_gt_path, pub_measurement_path,
    pub_odometry;
nav_msgs::Path path_estimation, path_gt, path_measurment;
ros::Publisher meshPub;

struct Node {
  Node() {
    ros::NodeHandle pnh("~");
    pnh.param("use_compressed_input", use_compressed_input_, false);
    pnh.param("show_img", show_img_, false);
    pnh.param<std::string>("filter_mode", filter_mode_name_, "ekf");
    pnh.param<std::string>(
        "ipm_topic", ipm_topic_,
        use_compressed_input_ ? std::string("/ipm_label/compressed")
                              : std::string("/ipm_label"));
    pnh.param<std::string>("gps_tum_path", gps_tum_path_, std::string(""));
    pnh.param<std::string>("estimation_tum_path", estimation_tum_path_,
                           std::string(""));
    pnh.param<std::string>("diagnostics_csv_path", diagnostics_csv_path_,
                           std::string(""));

    filter_mode_ = parseFilterMode(filter_mode_name_);
    readExperimentConfig(pnh);

    // Avoid std::make_shared because AvpLocalization uses aligned Eigen types.
    avp_localization_.reset(
        new AvpLocalization(DATASET_PATH "avp_map_sim_compressed.bin", show_img_,
                            filter_mode_, config_));

    ROS_INFO_STREAM("Using filter_mode: " << filterModeName(filter_mode_));
    ROS_INFO_STREAM("Experiment config | shared_frontend: scarcity="
                    << (config_.enable_scarcity_weighting ? "on" : "off")
                    << ", weight_mode=" << config_.primitive_weight_mode
                    << ", multi_primitive="
                    << (config_.enable_multi_primitive_residual ? "on" : "off")
                    << ", icp_gate="
                    << (config_.enable_icp_quality_gate ? "on" : "off")
                    << ", distance_field="
                    << (config_.enable_distance_field_registration ? "on"
                                                                   : "off")
                    << ", obs_consistency="
                    << (config_.enable_observation_consistency_gate ? "on"
                                                                    : "off")
                    << " | dma_only: strong_tracking="
                    << (config_.enable_dma_strong_tracking ? "on" : "off")
                    << ", obs_weighting="
                    << (config_.enable_dma_observation_weighting ? "on"
                                                                 : "off")
                    << ", iterative_refine="
                    << (config_.enable_iterative_refinement ? "on" : "off"));

    map_viewer_ = std::make_shared<RosViewer>(nh_);

    const Map &map = avp_localization_->getMap();
    map_viewer_->displayAvpMap(map, ros::Time::now().toSec());

    pub_odometry = nh_.advertise<nav_msgs::Odometry>("/odometry_est", 2000);
    pub_estimation_path =
        nh_.advertise<nav_msgs::Path>("/estimation_path", 2000);
    pub_gt_path = nh_.advertise<nav_msgs::Path>("/gps_path", 5000);
    meshPub =
        nh_.advertise<visualization_msgs::Marker>("/vehicle_mesh", 100, true);

    path_gt.header.frame_id = "world";
    path_estimation.header.frame_id = "world";

    initializeDataFiles();
  }

  bool useCompressedInput() const { return use_compressed_input_; }
  const std::string &ipmTopic() const { return ipm_topic_; }

  ~Node() {
    if (gps_file_.is_open()) {
      gps_file_.close();
    }
    if (estimation_file_.is_open()) {
      estimation_file_.close();
    }
    if (diagnostics_file_.is_open()) {
      diagnostics_file_.close();
    }
    std::cout << "Data files saved successfully." << std::endl;
  }

  void publishPath(const TimedPose &pose) {
    Eigen::Vector3d position = pose.t_;
    Eigen::Quaterniond q = pose.R_;

    nav_msgs::Odometry odometry;
    odometry.header.frame_id = "world";
    odometry.header.stamp = ros::Time(pose.time_);
    odometry.pose.pose.position.x = position(0);
    odometry.pose.pose.position.y = position(1);
    odometry.pose.pose.position.z = position(2);
    odometry.pose.pose.orientation.x = q.x();
    odometry.pose.pose.orientation.y = q.y();
    odometry.pose.pose.orientation.z = q.z();
    odometry.pose.pose.orientation.w = q.w();
    pub_odometry.publish(odometry);

    geometry_msgs::PoseStamped poseStamped;
    poseStamped.header.frame_id = "world";
    poseStamped.header.stamp = ros::Time(pose.time_);
    poseStamped.pose = odometry.pose.pose;
    path_estimation.poses.push_back(poseStamped);
    if (path_estimation.poses.size() > 50000) {
      path_estimation.poses.erase(path_estimation.poses.begin());
    }
    pub_estimation_path.publish(path_estimation);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q_tf;
    transform.setOrigin(tf::Vector3(position(0), position(1), position(2)));
    q_tf.setX(q.x());
    q_tf.setY(q.y());
    q_tf.setZ(q.z());
    q_tf.setW(q.w());
    transform.setRotation(q_tf);
    br.sendTransform(
        tf::StampedTransform(transform, ros::Time(pose.time_), "world", "map"));

    visualization_msgs::Marker meshROS;
    meshROS.header.frame_id = std::string("world");
    meshROS.header.stamp = ros::Time(pose.time_);
    meshROS.ns = "mesh";
    meshROS.id = 0;
    meshROS.type = visualization_msgs::Marker::MESH_RESOURCE;
    meshROS.action = visualization_msgs::Marker::ADD;
    Eigen::Matrix3d frontleftup2rightfrontup;
    frontleftup2rightfrontup << 0, 1, 0, -1, 0, 0, 0, 0, 1;
    Eigen::Matrix3d rot_mesh;
    rot_mesh << -1, 0, 0, 0, 0, 1, 0, 1, 0;
    Eigen::Quaterniond q_mesh;
    q_mesh = q * frontleftup2rightfrontup.transpose() * rot_mesh;
    Eigen::Vector3d t_mesh = q * Eigen::Vector3d(0, 1.5, 0) + position;
    meshROS.pose.orientation.w = q_mesh.w();
    meshROS.pose.orientation.x = q_mesh.x();
    meshROS.pose.orientation.y = q_mesh.y();
    meshROS.pose.orientation.z = q_mesh.z();
    meshROS.pose.position.x = t_mesh(0);
    meshROS.pose.position.y = t_mesh(1);
    meshROS.pose.position.z = t_mesh(2);
    meshROS.scale.x = 1.0;
    meshROS.scale.y = 1.0;
    meshROS.scale.z = 1.0;
    meshROS.color.a = 1.0;
    meshROS.color.r = 1.0;
    meshROS.color.g = 0.0;
    meshROS.color.b = 0.0;
    meshROS.mesh_resource = std::string("package://avp_mapping/launch/car.dae");
    meshPub.publish(meshROS);

    if (estimation_file_.is_open()) {
      // TUM format: timestamp tx ty tz qx qy qz qw
      estimation_file_ << pose.time_ << " " << position(0) << " " << position(1)
                       << " " << position(2) << " " << q.x() << " " << q.y()
                       << " " << q.z() << " " << q.w() << std::endl;
    }
  }

  void addOdomGps(const nav_msgs::OdometryConstPtr &msg) {
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
    t.x() = msg->pose.pose.position.x;
    t.y() = msg->pose.pose.position.y;
    t.z() = msg->pose.pose.position.z;
    q.x() = msg->pose.pose.orientation.x;
    q.y() = msg->pose.pose.orientation.y;
    q.z() = msg->pose.pose.orientation.z;
    q.w() = msg->pose.pose.orientation.w;

    if (!ekf_inited_) {
      ekf_inited_ = true;
      avp_localization_->initState(msg->header.stamp.toSec(), t.x(), t.y(),
                                   GetYaw(q));
    }

    geometry_msgs::PoseStamped poseStamped_gt;
    poseStamped_gt.header = msg->header;
    poseStamped_gt.header.frame_id = "world";
    poseStamped_gt.pose = msg->pose.pose;
    path_gt.poses.push_back(poseStamped_gt);
    if (path_gt.poses.size() > 50000) {
      path_gt.poses.erase(path_gt.poses.begin());
    }
    pub_gt_path.publish(path_gt);

    if (gps_file_.is_open()) {
      // TUM format: timestamp tx ty tz qx qy qz qw
      gps_file_ << msg->header.stamp.toSec() << " " << t.x() << " " << t.y()
                << " " << t.z() << " " << q.x() << " " << q.y() << " "
                << q.z() << " " << q.w() << std::endl;
    }
  }

  void addImuChassis(const lgsvl_msgs::CanBusDataConstPtr &chassis_speed,
                     const sensor_msgs::ImuConstPtr &imu) {
    if (ekf_inited_) {
      WheelMeasurement wheel_measurement;
      wheel_measurement.time_ = chassis_speed->header.stamp.toSec();
      double vx = chassis_speed->linear_velocities.x;
      double vz = chassis_speed->linear_velocities.z;
      int gear = chassis_speed->selected_gear;

      if (gear == 1) {
        wheel_measurement.velocity_ = sqrt(vx * vx + vz * vz);
      } else if (gear == 2) {
        wheel_measurement.velocity_ = -sqrt(vx * vx + vz * vz);
      }
      wheel_measurement.yaw_rate_ = imu->angular_velocity.z;
      avp_localization_->processWheelMeasurement(wheel_measurement);
    }
  }

  void processDecodedImage(const cv::Mat &image, double stamp) {
    avp_localization_->processImage(stamp, image);
    writeLatestDiagnostics();
    TimedPose current_pose = avp_localization_->getCurrentPose();
    publishPath(current_pose);
  }

  void addIPmImageRaw(const sensor_msgs::ImageConstPtr &msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception &e) {
      ROS_ERROR_STREAM_THROTTLE(1.0,
                                "cv_bridge raw image convert failed: "
                                    << e.what());
      return;
    }

    if (!cv_ptr->image.empty()) {
      processDecodedImage(cv_ptr->image, msg->header.stamp.toSec());
    }
  }

  void addIPmImageCompressed(const sensor_msgs::CompressedImageConstPtr &msg) {
    cv::Mat image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (!image.empty()) {
      processDecodedImage(image, msg->header.stamp.toSec());
    }
  }

  bool ekf_inited_{false};
  std::shared_ptr<AvpLocalization> avp_localization_;
  std::shared_ptr<RosViewer> map_viewer_;

  ros::NodeHandle nh_;
  ros::Publisher pub_estimation_path, pub_gt_path, pub_measurement_path,
      pub_odometry;
  nav_msgs::Path path_estimation, path_gt, path_measurment;
  ros::Publisher meshPub;

  ros::Publisher pub_map_;

 private:
  void writeLatestDiagnostics() {
    if (!diagnostics_file_.is_open()) {
      return;
    }

    LocalizationDiagnostics diag;
    if (!avp_localization_->getLatestDiagnostics(diag)) {
      return;
    }
    if (diag.seq <= last_diag_seq_written_) {
      return;
    }
    last_diag_seq_written_ = diag.seq;

    diagnostics_file_ << std::setprecision(9) << diag.time << ","
                      << diag.seq << "," << (diag.keyframe ? 1 : 0) << ","
                      << (diag.registration_success ? 1 : 0) << ","
                      << (diag.update_applied ? 1 : 0) << ","
                      << (diag.iterative_used ? 1 : 0) << ","
                      << diag.yaw_rate << "," << diag.filter_mode << ","
                      << diag.weight_mode << "," << diag.quality.matched << ","
                      << diag.quality.candidates << ","
                      << diag.quality.rmse_meter << ","
                      << diag.quality.inlier_ratio << ","
                      << diag.quality.quality_score << ","
                      << diag.quality.hessian_kappa << ","
                      << diag.filter_diag.nis << "," << diag.filter_diag.phi
                      << "," << diag.filter_diag.gamma << ","
                      << (diag.filter_diag.nis_outlier ? 1 : 0) << ","
                      << diag.feature_ms << "," << diag.icp_ms << ","
                      << diag.undistort_ms << "," << diag.filter_ms << ","
                      << diag.total_ms << "," << diag.reject_reason
                      << std::endl;
  }

  void readExperimentConfig(ros::NodeHandle &pnh) {
    pnh.param("keyframe_distance_thresh", config_.keyframe_distance_thresh,
              0.6);
    pnh.param("keyframe_yaw_thresh_deg", config_.keyframe_yaw_thresh_deg, 3.0);
    pnh.param("keyframe_time_thresh", config_.keyframe_time_thresh, 0.25);

    // Shared front-end for all baselines and ours.
    pnh.param("enable_scarcity_weighting", config_.enable_scarcity_weighting,
              true);
    pnh.param<std::string>("primitive_weight_mode",
                           config_.primitive_weight_mode, "scarcity");
    pnh.param("enable_multi_primitive_residual",
              config_.enable_multi_primitive_residual, true);
    pnh.param("enable_icp_quality_gate", config_.enable_icp_quality_gate,
              true);
    pnh.param("enable_distance_field_registration",
              config_.enable_distance_field_registration, false);
    pnh.param("enable_observation_consistency_gate",
              config_.enable_observation_consistency_gate, true);

    pnh.param("observation_max_translation_residual",
              config_.observation_max_translation_residual, 0.30);
    pnh.param("observation_max_yaw_residual_deg",
              config_.observation_max_yaw_residual_deg, 8.0);
    pnh.param("observation_hard_translation_reject",
              config_.observation_hard_translation_reject, 0.60);
    pnh.param("observation_hard_yaw_reject_deg",
              config_.observation_hard_yaw_reject_deg, 18.0);

    // DMA-only flags.
    const bool default_dma_on = (filter_mode_ == FilterMode::kDMAEKF);
    pnh.param("enable_dma_strong_tracking", config_.enable_dma_strong_tracking,
              default_dma_on);
    pnh.param("enable_dma_observation_weighting",
              config_.enable_dma_observation_weighting, default_dma_on);
    pnh.param("enable_iterative_refinement",
              config_.enable_iterative_refinement, default_dma_on);

    pnh.param("iterative_yaw_rate_thresh", config_.iterative_yaw_rate_thresh,
              0.45);
    pnh.param("dma_ewma_rho", config_.dma_ewma_rho, 0.95);
    pnh.param("dma_phi_max", config_.dma_phi_max, 12.0);
    pnh.param("dma_gamma_max", config_.dma_gamma_max, 6.0);
    pnh.param("dma_quality_alpha", config_.dma_quality_alpha, 0.2);
    pnh.param("dma_innovation_alpha", config_.dma_innovation_alpha, 0.8);
    pnh.param("dma_quality_trigger", config_.dma_quality_trigger, 0.35);
    pnh.param("dma_innovation_trigger", config_.dma_innovation_trigger, 1.10);

    if (filter_mode_ != FilterMode::kDMAEKF) {
      config_.enable_dma_strong_tracking = false;
      config_.enable_dma_observation_weighting = false;
      config_.enable_iterative_refinement = false;
    }
  }

  void initializeDataFiles() {
    const auto ensure_parent = [](const std::string &file_path) {
      if (file_path.empty()) {
        return;
      }
      std::filesystem::path p(file_path);
      if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
      }
    };

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();

    std::string gps_filename = gps_tum_path_.empty()
                                   ? DATASET_PATH +
                                         std::string("../output/gps_data_") +
                                         timestamp + ".tum"
                                   : gps_tum_path_;
    ensure_parent(gps_filename);
    gps_file_.open(gps_filename);
    if (gps_file_.is_open()) {
      gps_file_ << std::fixed << std::setprecision(6);
      std::cout << "GPS data file opened: " << gps_filename << std::endl;
    } else {
      std::cerr << "Failed to open GPS data file: " << gps_filename
                << std::endl;
    }

    std::string estimation_filename = estimation_tum_path_.empty()
                                          ? DATASET_PATH +
                                                std::string("../output/estimation_data_") +
                                                timestamp + ".tum"
                                          : estimation_tum_path_;
    ensure_parent(estimation_filename);
    estimation_file_.open(estimation_filename);
    if (estimation_file_.is_open()) {
      estimation_file_ << std::fixed << std::setprecision(6);
      std::cout << "Estimation data file opened: " << estimation_filename
                << std::endl;
    } else {
      std::cerr << "Failed to open estimation data file: " << estimation_filename
                << std::endl;
    }

    std::string diagnostics_filename = diagnostics_csv_path_.empty()
                                           ? DATASET_PATH +
                                                 std::string("../output/diagnostics_data_") +
                                                 timestamp + ".csv"
                                           : diagnostics_csv_path_;
    ensure_parent(diagnostics_filename);
    diagnostics_file_.open(diagnostics_filename);
    if (diagnostics_file_.is_open()) {
      diagnostics_file_ << "time,seq,keyframe,registration_success,update_applied,"
                        << "iterative_used,yaw_rate,filter_mode,weight_mode,"
                        << "matched,candidates,rmse_meter,inlier_ratio,quality_score,"
                        << "kappa_h,nis,phi,gamma,nis_outlier,feature_ms,icp_ms,"
                        << "undistort_ms,filter_ms,total_ms,reject_reason"
                        << std::endl;
      diagnostics_file_ << std::fixed << std::setprecision(9);
      std::cout << "Diagnostics file opened: " << diagnostics_filename
                << std::endl;
    } else {
      std::cerr << "Failed to open diagnostics file: " << diagnostics_filename
                << std::endl;
    }
  }

  std::ofstream gps_file_;
  std::ofstream estimation_file_;
  std::ofstream diagnostics_file_;
  std::size_t last_diag_seq_written_{0};

  bool use_compressed_input_{false};
  bool show_img_{false};
  std::string filter_mode_name_{"ekf"};
  FilterMode filter_mode_{FilterMode::kEKF};
  std::string ipm_topic_{"/ipm_label"};
  std::string gps_tum_path_;
  std::string estimation_tum_path_;
  std::string diagnostics_csv_path_;
  LocalizationConfig config_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "avp_localization_node");
  ros::NodeHandle nh("~");

  Node node;

  message_filters::Subscriber<sensor_msgs::Imu> sub0(nh, "/imu", 100);
  message_filters::Subscriber<lgsvl_msgs::CanBusData> sub1(nh, "/chassis", 100);

  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Imu,
                                                     lgsvl_msgs::CanBusData>
      sync_pol;
  message_filters::Synchronizer<sync_pol> sync(sync_pol(100), sub0, sub1);
  sync.registerCallback(boost::bind(&Node::addImuChassis, &node, _2, _1));

  ros::Subscriber gps_sub = nh.subscribe<nav_msgs::Odometry>(
      "/gps_odom", 100, boost::bind(&Node::addOdomGps, &node, _1));

  ros::Subscriber ipm_sub_raw;
  ros::Subscriber ipm_sub_compressed;
  if (node.useCompressedInput()) {
    ipm_sub_compressed = nh.subscribe<sensor_msgs::CompressedImage>(
        node.ipmTopic(), 10, boost::bind(&Node::addIPmImageCompressed, &node, _1));
    ROS_WARN_STREAM("Using compressed IPM input: " << node.ipmTopic());
  } else {
    ipm_sub_raw = nh.subscribe<sensor_msgs::Image>(
        node.ipmTopic(), 10, boost::bind(&Node::addIPmImageRaw, &node, _1));
    ROS_INFO_STREAM("Using raw IPM input: " << node.ipmTopic());
  }

  ros::spin();
}
