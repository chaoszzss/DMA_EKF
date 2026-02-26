#pragma once

#include <cstddef>
#include <deque>
#include <limits>
#include <memory>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "SystemPrimitives.hpp"

#include <nav_msgs/OccupancyGrid.h>

struct TimedPose {
  double time_;
  Eigen::Vector3d t_;
  Eigen::Quaterniond R_;
};

struct WheelMeasurement {
  double time_;
  double velocity_;
  double yaw_rate_;
};

enum class PrimitiveWeightMode {
  kLength = 0,
  kUniform,
  kScarcity
};

inline const char *primitiveWeightModeName(PrimitiveWeightMode mode) {
  switch (mode) {
    case PrimitiveWeightMode::kUniform:
      return "uniform";
    case PrimitiveWeightMode::kScarcity:
      return "scarcity";
    case PrimitiveWeightMode::kLength:
    default:
      return "length";
  }
}

struct LocalizationDiagnostics {
  std::size_t seq{0};
  double time{0.0};
  bool keyframe{false};
  bool registration_success{false};
  bool update_applied{false};
  bool iterative_used{false};
  double yaw_rate{0.0};
  std::string filter_mode;
  std::string weight_mode;
  std::string reject_reason;
  FilterObservationQuality quality;
  FilterStepDiagnostics filter_diag;
  double feature_ms{0.0};
  double icp_ms{0.0};
  double undistort_ms{0.0};
  double filter_ms{0.0};
  double total_ms{0.0};
};

// Experiment-facing switches used to separate shared front-end and
// DMA-specific back-end mechanisms.
struct LocalizationConfig {
  // Keyframe scheduling (shared for all filters).
  double keyframe_distance_thresh{0.6};          // meter
  double keyframe_yaw_thresh_deg{3.0};           // degree
  double keyframe_time_thresh{0.25};             // second

  // Shared front-end switches (can be enabled for EKF/AEKF/STEKF and DMAEKF).
  bool enable_scarcity_weighting{true};
  std::string primitive_weight_mode{"scarcity"};  // uniform/length/scarcity
  bool enable_multi_primitive_residual{true};
  bool enable_icp_quality_gate{true};
  bool enable_distance_field_registration{false};
  bool enable_observation_consistency_gate{true};

  // Observation consistency gate (shared by all filters).
  double observation_max_translation_residual{0.30};   // meter (soft clamp)
  double observation_max_yaw_residual_deg{8.0};        // degree (soft clamp)
  double observation_hard_translation_reject{0.60};    // meter (hard reject)
  double observation_hard_yaw_reject_deg{18.0};        // degree (hard reject)

  // DMA-specific switches (forced off for non-DMA filters in node).
  bool enable_dma_strong_tracking{true};
  bool enable_dma_observation_weighting{true};
  bool enable_iterative_refinement{true};

  // Thresholds / tuning.
  double iterative_yaw_rate_thresh{0.45};  // rad/s
  double dma_ewma_rho{0.95};
  double dma_phi_max{12.0};
  double dma_gamma_max{6.0};
  double dma_quality_alpha{0.2};
  double dma_innovation_alpha{0.8};
  double dma_quality_trigger{0.35};
  double dma_innovation_trigger{1.10};
};

class AvpLocalization {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  AvpLocalization(std::string map_file, bool enable_map_viewer = true,
                  FilterMode filter_mode = FilterMode::kEKF,
                  const LocalizationConfig &config = LocalizationConfig{});

  ~AvpLocalization() = default;

  void processWheelMeasurement(const WheelMeasurement &wheel_measurement);
  void processImage(double time, const cv::Mat &ipm_seg_img);
  void initState(double time, double x, double y, double yaw);

  // Registration with shared quality metrics.
  bool imageRegistration(Eigen::Affine3d &T_world_vehicle,
                         FilterObservationQuality &quality);

  const Map &getMap() const { return avp_map_; }
  TimedPose getCurrentPose() const;
  bool getLatestDiagnostics(LocalizationDiagnostics &diag) const;

 private:
  void buildSlotDistanceField();
  bool imageRegistrationLegacy(Eigen::Affine3d &T_world_vehicle,
                               FilterObservationQuality &quality);
  bool refineLegacyTranslationWithDistanceField(
      const Eigen::Affine3d &legacy_pose, Eigen::Affine3d &refined_pose,
      FilterObservationQuality &quality);
  bool imageRegistrationDistanceField(Eigen::Affine3d &T_world_vehicle,
                                      FilterObservationQuality &quality);
  bool sampleSlotDistanceField(const Eigen::Vector2f &pt_grid, float &dist,
                               Eigen::Vector2f &grad) const;

  bool isKeyFrame(const TimedPose &pose);
  Eigen::Vector3d ipmPlane2Global(const TimedPose &T_world_vehicle,
                                  const cv::Point2f &ipm_point);
  void extractFeature(const cv::Mat &img_gray,
                      const TimedPose &T_world_vehicle);

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_dash_line_;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_arrow_line_;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_slot_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr dash_line_cloud_in_{
      new pcl::PointCloud<pcl::PointXYZ>};
  pcl::PointCloud<pcl::PointXYZ>::Ptr arrow_line_cloud_in_{
      new pcl::PointCloud<pcl::PointXYZ>};
  pcl::PointCloud<pcl::PointXYZ>::Ptr slot_cloud_in_{
      new pcl::PointCloud<pcl::PointXYZ>};

  cv::Mat slot_dist_map_;
  cv::Mat slot_grad_x_;
  cv::Mat slot_grad_y_;
  int slot_map_min_x_{0};
  int slot_map_min_y_{0};
  int slot_map_padding_{80};
  bool slot_distance_ready_{false};

  TimedPose pre_key_pose_;
  std::deque<WheelMeasurement> wheel_measurements_;

  Eigen::Affine3d T_vehicle_ipm_;
  Frame curr_frame_;
  Map avp_map_;
  std::unique_ptr<FilterBase> filter_;
  FilterMode filter_mode_{FilterMode::kEKF};
  PrimitiveWeightMode primitive_weight_mode_{PrimitiveWeightMode::kScarcity};
  LocalizationConfig config_;
  double last_predict_yaw_rate_{0.0};
  LocalizationDiagnostics last_diag_{};
  bool has_last_diag_{false};
  std::size_t diag_seq_counter_{0};

  std::shared_ptr<MapViewer> map_viewer_;
};
