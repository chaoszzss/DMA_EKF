#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../include/LocalizationEngine.hpp"
#include <pcl/common/transforms.h>
#include <ros/ros.h>

namespace {
constexpr size_t kMinIcpCorrespondenceForSolve = 20;
constexpr size_t kMinIcpCorrespondenceForUpdate = 30;
constexpr double kMaxIcpRmseForUpdate = 1.5;      // meter
// KD-tree distance is computed in grid coordinates (not meters).
// 20 grids ~= 0.4 m at kPixelScale=0.02.
constexpr float kIcpDistanceThreshold = 20.0f * 20.0f;  // grid^2
constexpr float kDistanceFieldInlierDistGrid = 20.0f;
constexpr float kDistanceFieldMaxSampleDistGrid = 40.0f;
constexpr float kDistanceFieldHuberDeltaMeter = 0.12f;
constexpr float kDistanceFieldRefineMaxCorrectionGrid = 15.0f;   // 0.3 m
constexpr float kDistanceFieldRefineMaxDeltaStepGrid = 2.0f;     // 0.04 m
constexpr float kDistanceFieldRefinePriorSigmaGrid = 10.0f;      // 0.2 m
constexpr int kDistanceFieldMaxSize = 12000;
constexpr double kDistanceFieldPriorSigmaTGrid = 15.0;        // 0.3 m
constexpr double kDistanceFieldPriorSigmaYawRad = 8.0 * kToRad;
constexpr double kDistanceFieldMaxCorrectionGrid = 45.0;      // 0.9 m
constexpr double kDistanceFieldMaxCorrectionYawRad = 12.0 * kToRad;
constexpr double kIterativeQualityMargin = 0.0;

inline bool grayNear(const uchar value, const uchar target) {
  return std::abs(static_cast<int>(value) - static_cast<int>(target)) <=
         kSemanticGrayTolerance;
}

double wrapAngle(double angle) {
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

bool passQualityGate(const FilterObservationQuality &quality) {
  return quality.matched >= kMinIcpCorrespondenceForUpdate &&
         quality.rmse_meter <= kMaxIcpRmseForUpdate;
}

float scarcityWeight(size_t count) {
  if (count == 0) {
    return 0.0f;
  }
  return 1.0f / std::sqrt(static_cast<float>(count));
}

PrimitiveWeightMode parsePrimitiveWeightMode(const std::string &name,
                                             bool fallback_scarcity) {
  if (name == "uniform" || name == "w0" || name == "W0") {
    return PrimitiveWeightMode::kUniform;
  }
  if (name == "length" || name == "w1" || name == "W1") {
    return PrimitiveWeightMode::kLength;
  }
  if (name == "scarcity" || name == "w2" || name == "W2") {
    return PrimitiveWeightMode::kScarcity;
  }
  return fallback_scarcity ? PrimitiveWeightMode::kScarcity
                           : PrimitiveWeightMode::kLength;
}

void computeLabelWeights(PrimitiveWeightMode mode, size_t slot_count,
                         size_t dash_count, size_t arrow_count, size_t total_num,
                         float &w_slot, float &w_dash, float &w_arrow) {
  w_slot = 1.0f;
  w_dash = 1.0f;
  w_arrow = 1.0f;

  switch (mode) {
    case PrimitiveWeightMode::kUniform:
      w_slot = (slot_count > 0) ? 1.0f / static_cast<float>(slot_count) : 0.0f;
      w_dash = (dash_count > 0) ? 1.0f / static_cast<float>(dash_count) : 0.0f;
      w_arrow =
          (arrow_count > 0) ? 1.0f / static_cast<float>(arrow_count) : 0.0f;
      break;
    case PrimitiveWeightMode::kScarcity:
      w_slot = scarcityWeight(slot_count);
      w_dash = scarcityWeight(dash_count);
      w_arrow = scarcityWeight(arrow_count);
      break;
    case PrimitiveWeightMode::kLength:
    default:
      w_slot = 1.0f;
      w_dash = 1.0f;
      w_arrow = 1.0f;
      break;
  }

  float sum_w = 0.0f;
  sum_w += static_cast<float>(slot_count) * w_slot;
  sum_w += static_cast<float>(dash_count) * w_dash;
  sum_w += static_cast<float>(arrow_count) * w_arrow;
  const float norm =
      (sum_w > 1e-6f) ? static_cast<float>(total_num) / sum_w : 1.0f;
  w_slot *= norm;
  w_dash *= norm;
  w_arrow *= norm;
}

float huberWeight(float residual_meter) {
  const float abs_r = std::abs(residual_meter);
  if (abs_r <= kDistanceFieldHuberDeltaMeter) {
    return 1.0f;
  }
  return kDistanceFieldHuberDeltaMeter / std::max(abs_r, 1e-6f);
}

bool bilinearSampleFloat(const cv::Mat &img, float x, float y, float &value) {
  if (img.empty() || img.type() != CV_32FC1) {
    return false;
  }
  if (x < 0.0f || y < 0.0f || x >= static_cast<float>(img.cols - 1) ||
      y >= static_cast<float>(img.rows - 1)) {
    return false;
  }

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float wx = x - static_cast<float>(x0);
  const float wy = y - static_cast<float>(y0);

  const float v00 = img.at<float>(y0, x0);
  const float v10 = img.at<float>(y0, x1);
  const float v01 = img.at<float>(y1, x0);
  const float v11 = img.at<float>(y1, x1);
  value = (1.0f - wx) * (1.0f - wy) * v00 + wx * (1.0f - wy) * v10 +
          (1.0f - wx) * wy * v01 + wx * wy * v11;
  return std::isfinite(value);
}

}  // namespace

AvpLocalization::AvpLocalization(std::string map_file, bool enable_map_viewer,
                                 FilterMode filter_mode,
                                 const LocalizationConfig &config)
    : filter_mode_(filter_mode), config_(config) {
  primitive_weight_mode_ =
      parsePrimitiveWeightMode(config_.primitive_weight_mode,
                               config_.enable_scarcity_weighting);
  pre_key_pose_.time_ = -1;

  avp_map_.load(map_file);
  if (enable_map_viewer) {
    map_viewer_.reset(new MapViewer(avp_map_));
    ROS_INFO("MapViewer enabled (OpenCV window).");
  } else {
    map_viewer_.reset();
    ROS_WARN("MapViewer disabled to avoid GTK/OpenCV GUI conflicts.");
  }

  ROS_INFO_STREAM("Localization filter mode: " << filterModeName(filter_mode_));
  ROS_INFO_STREAM("Keyframe config: dist=" << config_.keyframe_distance_thresh
                                            << " m, yaw="
                                            << config_.keyframe_yaw_thresh_deg
                                            << " deg, dt="
                                            << config_.keyframe_time_thresh
                                            << " s");
  ROS_INFO_STREAM("Shared frontend: scarcity_weighting="
                  << (config_.enable_scarcity_weighting ? "on" : "off")
                  << ", weight_mode="
                  << primitiveWeightModeName(primitive_weight_mode_)
                  << ", multi_primitive="
                  << (config_.enable_multi_primitive_residual ? "on" : "off")
                  << ", icp_gate="
                  << (config_.enable_icp_quality_gate ? "on" : "off")
                  << ", distance_field="
                  << (config_.enable_distance_field_registration ? "on"
                                                                 : "off")
                  << ", obs_consistency="
                  << (config_.enable_observation_consistency_gate ? "on"
                                                                  : "off"));
  ROS_INFO_STREAM("DMA extras: strong_tracking="
                  << (config_.enable_dma_strong_tracking ? "on" : "off")
                  << ", obs_weighting="
                  << (config_.enable_dma_observation_weighting ? "on" : "off")
                  << ", iterative_refine="
                  << (config_.enable_iterative_refinement ? "on" : "off"));

  T_vehicle_ipm_.linear().setIdentity();
  T_vehicle_ipm_.translation() = Eigen::Vector3d(0.0, 1.32, 0.0);

  for (const auto &element :
       avp_map_.getSemanticElement(SemanticLabel::kDashLine)) {
    dash_line_cloud_in_->points.emplace_back(element.x(), element.y(), 0.0f);
  }
  for (const auto &element :
       avp_map_.getSemanticElement(SemanticLabel::kArrowLine)) {
    arrow_line_cloud_in_->points.emplace_back(element.x(), element.y(), 0.0f);
  }
  for (const auto &element :
       avp_map_.getSemanticElement(SemanticLabel::kSlot)) {
    slot_cloud_in_->points.emplace_back(element.x(), element.y(), 0.0f);
  }

  kdtree_dash_line_.setInputCloud(dash_line_cloud_in_);
  kdtree_arrow_line_.setInputCloud(arrow_line_cloud_in_);
  kdtree_slot_.setInputCloud(slot_cloud_in_);

  buildSlotDistanceField();
}

TimedPose AvpLocalization::getCurrentPose() const {
  TimedPose pose;
  if (!filter_ || !filter_->isInit()) {
    pose.time_ = 0.0;
    pose.t_ = {0.0, 0.0, 0.0};
    pose.R_ = Eigen::Quaterniond::Identity();
    return pose;
  }

  const State state = filter_->getState();
  pose.time_ = state.time;
  pose.t_ = {state.x, state.y, 0.0};
  pose.R_ =
      Eigen::Quaterniond(Eigen::AngleAxisd(state.yaw, Eigen::Vector3d::UnitZ()));
  return pose;
}

bool AvpLocalization::getLatestDiagnostics(LocalizationDiagnostics &diag) const {
  if (!has_last_diag_) {
    return false;
  }
  diag = last_diag_;
  return true;
}

bool AvpLocalization::isKeyFrame(const TimedPose &pose) {
  const double keyframe_yaw_thresh = config_.keyframe_yaw_thresh_deg * kToRad;
  if ((pose.t_ - pre_key_pose_.t_).norm() > config_.keyframe_distance_thresh ||
      std::fabs(GetYaw(pre_key_pose_.R_.inverse() * pose.R_)) >
          keyframe_yaw_thresh ||
      pose.time_ - pre_key_pose_.time_ > config_.keyframe_time_thresh ||
      pre_key_pose_.time_ < -1.0) {
    pre_key_pose_ = pose;
    return true;
  }
  return false;
}

void AvpLocalization::initState(double time, double x, double y, double yaw) {
  DmaFilterConfig dma_config;
  dma_config.enable_strong_tracking = config_.enable_dma_strong_tracking;
  dma_config.enable_observation_weighting =
      config_.enable_dma_observation_weighting;
  dma_config.ewma_rho = config_.dma_ewma_rho;
  dma_config.phi_max = config_.dma_phi_max;
  dma_config.gamma_max = config_.dma_gamma_max;
  dma_config.quality_alpha = config_.dma_quality_alpha;
  dma_config.innovation_alpha = config_.dma_innovation_alpha;
  dma_config.quality_trigger = config_.dma_quality_trigger;
  dma_config.innovation_trigger = config_.dma_innovation_trigger;

  filter_ = createFilter(filter_mode_, time, x, y, yaw, dma_config);

  if (map_viewer_) {
    curr_frame_.t_update = {x, y, 0.0};
    map_viewer_->showFrame(curr_frame_);
  }
}

void AvpLocalization::processWheelMeasurement(
    const WheelMeasurement &wheel_measurement) {
  wheel_measurements_.push_back(wheel_measurement);
}

void AvpLocalization::processImage(double time, const cv::Mat &ipm_seg_img) {
  while (!wheel_measurements_.empty() &&
         wheel_measurements_.front().time_ < time) {
    if (filter_) {
      last_predict_yaw_rate_ = wheel_measurements_.front().yaw_rate_;
      filter_->predict(wheel_measurements_.front().time_,
                       wheel_measurements_.front().velocity_,
                       wheel_measurements_.front().yaw_rate_);
    }
    wheel_measurements_.pop_front();
  }

  if (!wheel_measurements_.empty() && filter_) {
    last_predict_yaw_rate_ = wheel_measurements_.front().yaw_rate_;
    filter_->predict(time, wheel_measurements_.front().velocity_,
                     wheel_measurements_.front().yaw_rate_);
  }

  if (!(filter_ && filter_->isInit())) {
    std::cout << "Unable to get state at " << std::to_string(time) << std::endl;
    return;
  }

  State state = filter_->getState();
  TimedPose pose;
  pose.time_ = state.time;
  pose.t_ = {state.x, state.y, 0.0};
  pose.R_ = Eigen::Quaterniond(
      Eigen::AngleAxisd{state.yaw, Eigen::Vector3d::UnitZ()});

  if (!isKeyFrame(pose)) {
    return;
  }

  LocalizationDiagnostics diag;
  diag.seq = ++diag_seq_counter_;
  diag.time = time;
  diag.keyframe = true;
  diag.filter_mode = filterModeName(filter_mode_);
  diag.weight_mode = primitiveWeightModeName(primitive_weight_mode_);
  diag.yaw_rate = last_predict_yaw_rate_;
  diag.iterative_used = false;

  const auto total_start = std::chrono::steady_clock::now();
  auto flushDiag = [&](const char *reason) {
    diag.reject_reason = reason ? reason : "";
    const auto total_end = std::chrono::steady_clock::now();
    diag.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    last_diag_ = diag;
    has_last_diag_ = true;
  };

  std::cout << std::endl;
  printf("Keyframe: time = %.4f  (%.3f, %.3f, %.2f) \n", time, state.x,
         state.y, state.yaw * kToDeg);

  cv::Mat img_gray;
  cv::cvtColor(ipm_seg_img, img_gray, cv::COLOR_BGR2GRAY);
  const auto feature_start = std::chrono::steady_clock::now();
  extractFeature(img_gray, pose);
  const auto feature_end = std::chrono::steady_clock::now();
  diag.feature_ms =
      std::chrono::duration<double, std::milli>(feature_end - feature_start)
          .count();

  Eigen::Affine3d T_world_vehicle = Eigen::Affine3d::Identity();
  FilterObservationQuality quality;
  const auto icp_start = std::chrono::steady_clock::now();
  const bool registration_ok = imageRegistration(T_world_vehicle, quality);
  const auto icp_end = std::chrono::steady_clock::now();
  diag.icp_ms +=
      std::chrono::duration<double, std::milli>(icp_end - icp_start).count();
  if (!registration_ok) {
    diag.registration_success = false;
    diag.quality = quality;
    ROS_WARN_STREAM_THROTTLE(
        1.0, "Skip EKF update: ICP solve failed. matched=" << quality.matched
                                                             << ", rmse="
                                                             << quality.rmse_meter);
    flushDiag("icp_solve_failed");
    return;
  }
  diag.registration_success = true;
  diag.quality = quality;

  const Eigen::Vector3d t_est = T_world_vehicle.translation();
  const Eigen::Matrix3d R_est = T_world_vehicle.rotation();
  const double yaw_est = GetYaw(Eigen::Quaterniond(R_est));
  printf("ICP: time = %.4f  (%.3f, %.3f, %.2f) \n", time, t_est.x(), t_est.y(),
         yaw_est * kToDeg);

  if (config_.enable_icp_quality_gate && !passQualityGate(quality)) {
    diag.quality = quality;
    ROS_WARN_STREAM_THROTTLE(1.0,
                             "Skip EKF update due to match quality gate. "
                             "matched="
                                 << quality.matched << ", rmse="
                                 << quality.rmse_meter
                                 << ", inlier_ratio=" << quality.inlier_ratio);
    flushDiag("quality_gate_reject");
    return;
  }

  auto sanitizeObservation = [&](Eigen::Vector3d &obs_t, double &obs_yaw,
                                 FilterObservationQuality &obs_quality) {
    if (!config_.enable_observation_consistency_gate) {
      return true;
    }

    const State pred_state = filter_->getState();
    const Eigen::Vector2d pred_xy(pred_state.x, pred_state.y);
    Eigen::Vector2d obs_xy(obs_t.x(), obs_t.y());
    const Eigen::Vector2d innovation_xy = obs_xy - pred_xy;
    const double innovation_trans = innovation_xy.norm();
    const double innovation_yaw = std::abs(wrapAngle(obs_yaw - pred_state.yaw));

    const double soft_trans =
        std::max(0.05, config_.observation_max_translation_residual);
    const double soft_yaw =
        std::max(1.0 * kToRad, config_.observation_max_yaw_residual_deg * kToRad);
    const double hard_trans =
        std::max(soft_trans, config_.observation_hard_translation_reject);
    const double hard_yaw =
        std::max(soft_yaw, config_.observation_hard_yaw_reject_deg * kToRad);

    if (innovation_trans > hard_trans || innovation_yaw > hard_yaw) {
      ROS_WARN_STREAM_THROTTLE(
          1.0, "Skip update: innovation hard reject. dxy="
                   << innovation_trans << " m, dyaw="
                   << innovation_yaw * kToDeg << " deg");
      return false;
    }

    const double trans_scale =
        (innovation_trans > 1e-9) ? std::min(1.0, soft_trans / innovation_trans)
                                  : 1.0;
    const double yaw_scale =
        (innovation_yaw > 1e-9) ? std::min(1.0, soft_yaw / innovation_yaw)
                                : 1.0;
    const double scale = std::min(trans_scale, yaw_scale);

    if (scale < 0.999) {
      obs_xy = pred_xy + scale * innovation_xy;
      const double yaw_delta = wrapAngle(obs_yaw - pred_state.yaw);
      obs_yaw = wrapAngle(pred_state.yaw + scale * yaw_delta);
      obs_t.x() = obs_xy.x();
      obs_t.y() = obs_xy.y();

      if (std::isfinite(obs_quality.quality_score)) {
        obs_quality.quality_score =
            std::max(obs_quality.quality_score, 0.25 + 0.7 * (1.0 - scale));
      }
      obs_quality.inlier_ratio = std::min(obs_quality.inlier_ratio, 0.2 + 0.8 * scale);

      ROS_WARN_STREAM_THROTTLE(
          1.0, "Clamp observation innovation. dxy="
                   << innovation_trans << " m, dyaw="
                   << innovation_yaw * kToDeg << " deg, scale=" << scale);
    }
    return true;
  };

  const bool use_iterative_refine =
      filter_mode_ == FilterMode::kDMAEKF &&
      config_.enable_iterative_refinement &&
      std::abs(last_predict_yaw_rate_) >= config_.iterative_yaw_rate_thresh;

  if (!use_iterative_refine) {
    Eigen::Vector3d selected_t = t_est;
    double selected_yaw = yaw_est;
    FilterObservationQuality selected_quality = quality;
    if (!sanitizeObservation(selected_t, selected_yaw, selected_quality)) {
      diag.quality = selected_quality;
      flushDiag("observation_hard_reject");
      return;
    }
    const auto filter_start = std::chrono::steady_clock::now();
    filter_->setObservationQuality(selected_quality);
    filter_->update(selected_t.x(), selected_t.y(), selected_yaw);
    const auto filter_end = std::chrono::steady_clock::now();
    diag.filter_ms += std::chrono::duration<double, std::milli>(
                          filter_end - filter_start)
                          .count();
    diag.update_applied = true;
    diag.quality = selected_quality;
    diag.filter_diag = filter_->getLastStepDiagnostics();
  } else {
    diag.iterative_used = true;
    const auto iterative_start = std::chrono::steady_clock::now();
    // Iterative mode uses best-of-two observation and applies only one final
    // update to avoid double-counting the same frame.
    const State state_before = filter_->getState();

    Eigen::Vector3d first_pass_t = t_est;
    double first_pass_yaw = yaw_est;
    FilterObservationQuality first_pass_quality = quality;
    if (!sanitizeObservation(first_pass_t, first_pass_yaw, first_pass_quality)) {
      diag.quality = first_pass_quality;
      flushDiag("iterative_first_pass_reject");
      return;
    }

    filter_->setObservationQuality(first_pass_quality);
    filter_->update(first_pass_t.x(), first_pass_t.y(), first_pass_yaw);

    State temp_state = filter_->getState();
    TimedPose temp_pose;
    temp_pose.time_ = temp_state.time;
    temp_pose.t_ = {temp_state.x, temp_state.y, 0.0};
    temp_pose.R_ = Eigen::Quaterniond(
        Eigen::AngleAxisd{temp_state.yaw, Eigen::Vector3d::UnitZ()});
    extractFeature(img_gray, temp_pose);

    Eigen::Affine3d T_world_vehicle_refined = Eigen::Affine3d::Identity();
    FilterObservationQuality refined_quality;
    bool refined_valid =
        imageRegistration(T_world_vehicle_refined, refined_quality);
    if (refined_valid && config_.enable_icp_quality_gate) {
      refined_valid = passQualityGate(refined_quality);
    }

    Eigen::Vector3d selected_t = first_pass_t;
    double selected_yaw = first_pass_yaw;
    FilterObservationQuality selected_quality = first_pass_quality;

    if (refined_valid) {
      const bool refined_better =
          refined_quality.quality_score + kIterativeQualityMargin <
          quality.quality_score;
      if (refined_better) {
        selected_t = T_world_vehicle_refined.translation();
        selected_yaw =
            GetYaw(Eigen::Quaterniond(T_world_vehicle_refined.rotation()));
        selected_quality = refined_quality;
      }
      ROS_INFO_STREAM_THROTTLE(
          1.0, "DMA iterative candidate. quality: " << quality.quality_score
                                                     << " -> "
                                                     << refined_quality.quality_score
                                                     << ", rmse: "
                                                     << quality.rmse_meter << " -> "
                                                     << refined_quality.rmse_meter);
    }

    filter_->setState(state_before);
    if (!sanitizeObservation(selected_t, selected_yaw, selected_quality)) {
      diag.quality = selected_quality;
      flushDiag("iterative_selected_reject");
      return;
    }
    const auto filter_start = std::chrono::steady_clock::now();
    filter_->setObservationQuality(selected_quality);
    filter_->update(selected_t.x(), selected_t.y(), selected_yaw);
    const auto filter_end = std::chrono::steady_clock::now();
    diag.filter_ms += std::chrono::duration<double, std::milli>(
                          filter_end - filter_start)
                          .count();
    const auto iterative_end = std::chrono::steady_clock::now();
    diag.undistort_ms += std::chrono::duration<double, std::milli>(
                             iterative_end - iterative_start)
                             .count();
    diag.update_applied = true;
    diag.quality = selected_quality;
    diag.filter_diag = filter_->getLastStepDiagnostics();
  }

  std::cout << "**********************************************" << std::endl;
  if (map_viewer_) {
    state = filter_->getState();
    pose.time_ = state.time;
    pose.t_ = {state.x, state.y, 0.0};
    pose.R_ = Eigen::Quaterniond(
        Eigen::AngleAxisd{state.yaw, Eigen::Vector3d::UnitZ()});

    extractFeature(img_gray, pose);
    curr_frame_.t_update = pose.t_;
    map_viewer_->showFrame(curr_frame_);
  }

  flushDiag("ok");
}

Eigen::Vector3d AvpLocalization::ipmPlane2Global(
    const TimedPose &T_world_vehicle, const cv::Point2f &ipm_point) {
  const Eigen::Affine3d T_world_ipm = Eigen::Translation3d(T_world_vehicle.t_) *
                                      T_world_vehicle.R_ * T_vehicle_ipm_;
  Eigen::Vector3d pt_ipm{-(0.5 * kIPMImgHeight - ipm_point.x) * kPixelScale,
                         (0.5 * kIPMImgWidth - ipm_point.y) * kPixelScale,
                         0.0};
  return T_world_ipm * pt_ipm;
}

void AvpLocalization::extractFeature(const cv::Mat &img_gray,
                                     const TimedPose &T_world_vehicle) {
  curr_frame_.clearGridMap();
  curr_frame_.T_world_ipm_ = Eigen::Translation3d(T_world_vehicle.t_) *
                             T_world_vehicle.R_ * T_vehicle_ipm_;

  cv::Mat slot_img = cv::Mat::zeros(img_gray.size(), img_gray.type());
  cv::Mat seg_img = cv::Mat::zeros(img_gray.size(), img_gray.type());

  for (int i = 0; i < img_gray.rows; ++i) {
    for (int j = 0; j < img_gray.cols; ++j) {
      const uchar pixel = img_gray.at<uchar>(i, j);
      if (grayNear(pixel, kSlotGray) || grayNear(pixel, kSlotGray1)) {
        slot_img.at<uchar>(i, j) = 254;
      } else if (grayNear(pixel, kArrowGray) || grayNear(pixel, kDashGray)) {
        seg_img.at<uchar>(i, j) = 254;
      }
    }
  }

  const int kernel_size = 15;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                             cv::Size(kernel_size, kernel_size));
  cv::Mat closed;
  cv::morphologyEx(slot_img, closed, cv::MORPH_CLOSE, kernel);

  cv::Mat skel = skeletonize(closed);
  removeIsolatedPixels(skel, 2);

  for (int i = 0; i < skel.rows; ++i) {
    for (int j = 0; j < skel.cols; ++j) {
      if (skel.at<uchar>(i, j)) {
        curr_frame_.addSemanticElement(
            SemanticLabel::kSlot,
            {ipmPlane2Global(T_world_vehicle, cv::Point2f(j, i))});
      }
    }
  }

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(seg_img, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  for (const auto &contour : contours) {
    for (const auto &point : contour) {
      const uchar pixel = img_gray.at<uchar>(point);
      if (grayNear(pixel, kArrowGray)) {
        curr_frame_.addSemanticElement(
            SemanticLabel::kArrowLine,
            {ipmPlane2Global(T_world_vehicle, point)});
      } else if (grayNear(pixel, kDashGray)) {
        curr_frame_.addSemanticElement(
            SemanticLabel::kDashLine,
            {ipmPlane2Global(T_world_vehicle, point)});
      }
    }
  }
}

void AvpLocalization::buildSlotDistanceField() {
  slot_distance_ready_ = false;

  const auto &slot_points = avp_map_.getSemanticElement(SemanticLabel::kSlot);
  if (slot_points.empty()) {
    ROS_WARN("Slot map is empty, distance-field registration disabled.");
    return;
  }

  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::lowest();
  int max_y = std::numeric_limits<int>::lowest();
  for (const auto &pt : slot_points) {
    min_x = std::min(min_x, pt.x());
    min_y = std::min(min_y, pt.y());
    max_x = std::max(max_x, pt.x());
    max_y = std::max(max_y, pt.y());
  }

  const int width = (max_x - min_x + 1) + 2 * slot_map_padding_;
  const int height = (max_y - min_y + 1) + 2 * slot_map_padding_;
  if (width <= 2 || height <= 2 || width > kDistanceFieldMaxSize ||
      height > kDistanceFieldMaxSize) {
    ROS_WARN_STREAM("Skip distance field due to invalid map size: " << width
                                                                     << "x"
                                                                     << height);
    return;
  }

  slot_map_min_x_ = min_x;
  slot_map_min_y_ = min_y;

  cv::Mat occ(height, width, CV_8UC1, cv::Scalar(255));
  for (const auto &pt : slot_points) {
    const int col = pt.x() - slot_map_min_x_ + slot_map_padding_;
    const int row = pt.y() - slot_map_min_y_ + slot_map_padding_;
    if (col >= 0 && col < occ.cols && row >= 0 && row < occ.rows) {
      occ.at<uchar>(row, col) = 0;
    }
  }

  cv::distanceTransform(occ, slot_dist_map_, cv::DIST_L2, 3);
  cv::Sobel(slot_dist_map_, slot_grad_x_, CV_32F, 1, 0, 3, 1.0, 0.0,
            cv::BORDER_REPLICATE);
  cv::Sobel(slot_dist_map_, slot_grad_y_, CV_32F, 0, 1, 3, 1.0, 0.0,
            cv::BORDER_REPLICATE);
  slot_distance_ready_ = true;

  ROS_INFO_STREAM("Slot distance field ready. size=" << width << "x" << height
                                                      << ", points="
                                                      << slot_points.size());
}

bool AvpLocalization::sampleSlotDistanceField(const Eigen::Vector2f &pt_grid,
                                              float &dist,
                                              Eigen::Vector2f &grad) const {
  if (!slot_distance_ready_) {
    return false;
  }

  const float col =
      pt_grid.x() - static_cast<float>(slot_map_min_x_ - slot_map_padding_);
  const float row =
      pt_grid.y() - static_cast<float>(slot_map_min_y_ - slot_map_padding_);

  float gx = 0.0f;
  float gy = 0.0f;
  if (!bilinearSampleFloat(slot_dist_map_, col, row, dist) ||
      !bilinearSampleFloat(slot_grad_x_, col, row, gx) ||
      !bilinearSampleFloat(slot_grad_y_, col, row, gy)) {
    return false;
  }
  grad = Eigen::Vector2f(gx, gy);
  return std::isfinite(dist) && grad.allFinite();
}

bool AvpLocalization::imageRegistrationDistanceField(
    Eigen::Affine3d &T_world_vehicle, FilterObservationQuality &quality) {
  const auto &slot_elems = curr_frame_.getSemanticElement(SemanticLabel::kSlot);
  const size_t total_num = slot_elems.size();
  quality = FilterObservationQuality{};
  quality.candidates = total_num;

  if (!slot_distance_ready_ || total_num < kMinIcpCorrespondenceForSolve) {
    return false;
  }

  std::vector<Eigen::Vector2f> points;
  points.reserve(total_num);
  for (const auto &grid : slot_elems) {
    points.emplace_back(static_cast<float>(grid.x()),
                        static_cast<float>(grid.y()));
  }

  double tx = 0.0;
  double ty = 0.0;
  double theta = 0.0;
  constexpr int max_iter = 12;
  constexpr double kRegLambda = 1e-3;

  for (int iter = 0; iter < max_iter; ++iter) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    size_t valid_num = 0;

    const double c = std::cos(theta);
    const double s = std::sin(theta);
    for (const auto &p : points) {
      const double qx = c * p.x() - s * p.y() + tx;
      const double qy = s * p.x() + c * p.y() + ty;

      float dist_grid = 0.0f;
      Eigen::Vector2f grad;
      if (!sampleSlotDistanceField(Eigen::Vector2f(static_cast<float>(qx),
                                                   static_cast<float>(qy)),
                                   dist_grid, grad)) {
        continue;
      }
      if (dist_grid > kDistanceFieldMaxSampleDistGrid) {
        continue;
      }

      const float residual_meter = dist_grid * static_cast<float>(kPixelScale);
      const float w = huberWeight(residual_meter);

      const double dqx_dtheta = -s * p.x() - c * p.y();
      const double dqy_dtheta = c * p.x() - s * p.y();
      Eigen::Vector3d J;
      J << grad.x(), grad.y(),
          grad.x() * dqx_dtheta + grad.y() * dqy_dtheta;

      H.noalias() += static_cast<double>(w) * (J * J.transpose());
      b.noalias() += static_cast<double>(w * dist_grid) * J;
      ++valid_num;
    }

    if (valid_num < kMinIcpCorrespondenceForSolve) {
      return false;
    }

    // Keep the optimizer near prediction to avoid falling into repetitive
    // parking-line local minima.
    const double prior_w_t =
        1.0 / (kDistanceFieldPriorSigmaTGrid * kDistanceFieldPriorSigmaTGrid);
    const double prior_w_yaw = 1.0 /
                               (kDistanceFieldPriorSigmaYawRad *
                                kDistanceFieldPriorSigmaYawRad);
    H(0, 0) += prior_w_t;
    H(1, 1) += prior_w_t;
    H(2, 2) += prior_w_yaw;
    b(0) += prior_w_t * tx;
    b(1) += prior_w_t * ty;
    b(2) += prior_w_yaw * theta;

    H += kRegLambda * Eigen::Matrix3d::Identity();
    const Eigen::Vector3d delta = -H.ldlt().solve(b);
    if (!delta.allFinite()) {
      return false;
    }

    tx += std::max(-3.0, std::min(3.0, delta(0)));
    ty += std::max(-3.0, std::min(3.0, delta(1)));
    const double d_theta = std::max(-0.1, std::min(0.1, delta(2)));
    theta = std::atan2(std::sin(theta + d_theta), std::cos(theta + d_theta));
    if (delta.norm() < 1e-3) {
      break;
    }
  }

  const double correction_norm = std::hypot(tx, ty);
  if (correction_norm > kDistanceFieldMaxCorrectionGrid ||
      std::abs(theta) > kDistanceFieldMaxCorrectionYawRad) {
    return false;
  }

  size_t valid_num = 0;
  size_t inlier_num = 0;
  double sum_sq_meter = 0.0;
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  for (const auto &p : points) {
    const double qx = c * p.x() - s * p.y() + tx;
    const double qy = s * p.x() + c * p.y() + ty;
    float dist_grid = 0.0f;
    Eigen::Vector2f grad = Eigen::Vector2f::Zero();
    if (!sampleSlotDistanceField(
            Eigen::Vector2f(static_cast<float>(qx), static_cast<float>(qy)),
            dist_grid, grad)) {
      continue;
    }
    if (dist_grid > kDistanceFieldMaxSampleDistGrid) {
      continue;
    }
    ++valid_num;
    if (dist_grid <= kDistanceFieldInlierDistGrid) {
      ++inlier_num;
      const double residual_meter = dist_grid * kPixelScale;
      sum_sq_meter += residual_meter * residual_meter;
    }
  }

  if (valid_num < kMinIcpCorrespondenceForSolve ||
      inlier_num < kMinIcpCorrespondenceForSolve) {
    return false;
  }

  quality.matched = inlier_num;
  quality.inlier_ratio =
      static_cast<double>(inlier_num) / static_cast<double>(total_num);
  quality.rmse_meter =
      std::sqrt(sum_sq_meter / static_cast<double>(quality.matched));
  if (!std::isfinite(quality.rmse_meter)) {
    return false;
  }

  const double rmse_norm = std::min(2.0, quality.rmse_meter / 1.0);
  quality.quality_score =
      0.5 * rmse_norm + 0.5 * (1.0 - quality.inlier_ratio);

  const Eigen::Affine3d dT = Eigen::Translation3d(
                                 Eigen::Vector3d(tx * kPixelScale,
                                                 ty * kPixelScale, 0.0)) *
                             Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
  T_world_vehicle = dT * curr_frame_.T_world_ipm_ * T_vehicle_ipm_.inverse();
  return true;
}

bool AvpLocalization::imageRegistrationLegacy(Eigen::Affine3d &T_world_vehicle,
                                              FilterObservationQuality &quality) {
  T_world_vehicle = curr_frame_.T_world_ipm_ * T_vehicle_ipm_.inverse();
  quality = FilterObservationQuality{};

  const size_t slot_count =
      curr_frame_.getSemanticElement(SemanticLabel::kSlot).size();
  const size_t dash_count_raw =
      curr_frame_.getSemanticElement(SemanticLabel::kDashLine).size();
  const size_t arrow_count_raw =
      curr_frame_.getSemanticElement(SemanticLabel::kArrowLine).size();

  const bool use_multi = config_.enable_multi_primitive_residual;
  const size_t dash_count = use_multi ? dash_count_raw : 0;
  const size_t arrow_count = use_multi ? arrow_count_raw : 0;

  const size_t slot_end = slot_count;
  const size_t dash_end = slot_end + dash_count;
  const size_t total_num = dash_end + arrow_count;
  quality.candidates = total_num;

  if (total_num < kMinIcpCorrespondenceForSolve) {
    return false;
  }

  std::vector<char> correspondences;
  std::vector<float> point_weights(total_num, 1.0f);
  pcl::PointCloud<pcl::PointXYZ> cloud_in;
  pcl::PointCloud<pcl::PointXYZ> cloud_out;
  pcl::PointCloud<pcl::PointXYZ> cloud_est;

  cloud_out.reserve(total_num);
  for (const auto &grid : curr_frame_.getSemanticElement(SemanticLabel::kSlot)) {
    cloud_out.emplace_back(grid.x(), grid.y(), 0.0f);
  }
  if (use_multi) {
    for (const auto &grid :
         curr_frame_.getSemanticElement(SemanticLabel::kDashLine)) {
      cloud_out.emplace_back(grid.x(), grid.y(), 0.0f);
    }
    for (const auto &grid :
         curr_frame_.getSemanticElement(SemanticLabel::kArrowLine)) {
      cloud_out.emplace_back(grid.x(), grid.y(), 0.0f);
    }
  }
  cloud_in.resize(cloud_out.size());

  float w_slot = 1.0f;
  float w_dash = 1.0f;
  float w_arrow = 1.0f;
  computeLabelWeights(primitive_weight_mode_, slot_count, dash_count, arrow_count,
                      total_num, w_slot, w_dash, w_arrow);
  for (size_t i = 0; i < slot_end; ++i) {
    point_weights[i] = w_slot;
  }
  for (size_t i = slot_end; i < dash_end; ++i) {
    point_weights[i] = w_dash;
  }
  for (size_t i = dash_end; i < total_num; ++i) {
    point_weights[i] = w_arrow;
  }

  Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  Eigen::Vector3f t = Eigen::Vector3f::Zero();
  const int max_iter = 10;

  double kappa_proxy = std::numeric_limits<double>::quiet_NaN();
  for (int n = 0; n < max_iter; ++n) {
    const Eigen::Affine3f T_est = Eigen::Translation3f(t) * R;
    pcl::transformPointCloud(cloud_out, cloud_est, T_est);

    correspondences.assign(total_num, 0);
    std::vector<int> indices(1);
    std::vector<float> distances(1);
    double sum_squared_distance = 0.0;

    for (size_t i = 0; i < total_num; ++i) {
      const pcl::PointXYZ search_point = cloud_est[i];
      int found = 0;

      if (i < slot_end) {
        found = kdtree_slot_.nearestKSearch(search_point, 1, indices, distances);
      } else if (i < dash_end) {
        found =
            kdtree_dash_line_.nearestKSearch(search_point, 1, indices, distances);
      } else {
        found =
            kdtree_arrow_line_.nearestKSearch(search_point, 1, indices, distances);
      }

      if (found && distances[0] < kIcpDistanceThreshold) {
        correspondences[i] = 1;
        sum_squared_distance += distances[0];
        if (i < slot_end) {
          cloud_in[i] = slot_cloud_in_->points[indices[0]];
        } else if (i < dash_end) {
          cloud_in[i] = dash_line_cloud_in_->points[indices[0]];
        } else {
          cloud_in[i] = arrow_line_cloud_in_->points[indices[0]];
        }
      }
    }

    size_t num_pts = 0;
    float sum_w = 0.0f;
    Eigen::Vector3f sum_point_in = Eigen::Vector3f::Zero();
    Eigen::Vector3f sum_point_est = Eigen::Vector3f::Zero();
    for (size_t i = 0; i < total_num; ++i) {
      if (!correspondences[i]) {
        continue;
      }
      ++num_pts;
      const float wi = point_weights[i];
      sum_w += wi;
      sum_point_in += wi * Eigen::Vector3f(cloud_in[i].x, cloud_in[i].y, cloud_in[i].z);
      sum_point_est += wi * Eigen::Vector3f(cloud_est[i].x, cloud_est[i].y, cloud_est[i].z);
    }

    if (num_pts < kMinIcpCorrespondenceForSolve || sum_w <= 1e-6f) {
      return false;
    }

    quality.matched = num_pts;
    quality.inlier_ratio =
        static_cast<double>(num_pts) / static_cast<double>(total_num);
    quality.rmse_meter =
        std::sqrt(sum_squared_distance / static_cast<double>(num_pts)) * kPixelScale;

    const Eigen::Vector3f u_point_in = sum_point_in / sum_w;
    const Eigen::Vector3f u_point_est = sum_point_est / sum_w;

    Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
    for (size_t i = 0; i < total_num; ++i) {
      if (!correspondences[i]) {
        continue;
      }
      const float wi = point_weights[i];
      const Eigen::Vector3f in_rc = cloud_in[i].getVector3fMap() - u_point_in;
      const Eigen::Vector3f est_rc = cloud_est[i].getVector3fMap() - u_point_est;
      W += wi * est_rc * in_rc.transpose();
    }

    const Eigen::JacobiSVD<Eigen::Matrix3f> svd(
        W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector3f singular_values = svd.singularValues();
    const double s0 = static_cast<double>(singular_values(0));
    const double s1 = static_cast<double>(singular_values(1));
    kappa_proxy = s0 / std::max(1e-6, s1);
    const Eigen::Matrix3f U = svd.matrixU();
    const Eigen::Matrix3f V = svd.matrixV();
    Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
    if ((V * U.transpose()).determinant() < 0.0f) {
      S(2, 2) = -1.0f;
    }
    const Eigen::Matrix3f R_ = V * S * U.transpose();
    const Eigen::Vector3f t_ = u_point_in - R_ * u_point_est;

    const Eigen::Affine3f T_iter = Eigen::Translation3f(t_) * R_;
    const Eigen::Affine3f T_total = T_iter * (Eigen::Translation3f(t) * R);
    R = T_total.rotation();
    t = T_total.translation();
  }

  if (quality.matched < kMinIcpCorrespondenceForSolve ||
      !std::isfinite(quality.rmse_meter)) {
    return false;
  }

  const double rmse_norm = std::min(2.0, quality.rmse_meter / 1.0);
  quality.quality_score =
      0.5 * rmse_norm + 0.5 * (1.0 - quality.inlier_ratio);
  quality.hessian_kappa = kappa_proxy;

  t *= static_cast<float>(kPixelScale);
  const Eigen::Affine3d dT =
      Eigen::Translation3d(t.cast<double>()) * R.cast<double>();
  T_world_vehicle = dT * curr_frame_.T_world_ipm_ * T_vehicle_ipm_.inverse();
  return true;
}

bool AvpLocalization::refineLegacyTranslationWithDistanceField(
    const Eigen::Affine3d &legacy_pose, Eigen::Affine3d &refined_pose,
    FilterObservationQuality &quality) {
  refined_pose = legacy_pose;
  quality = FilterObservationQuality{};
  if (!slot_distance_ready_) {
    return false;
  }

  const auto &slot_elems = curr_frame_.getSemanticElement(SemanticLabel::kSlot);
  const auto &dash_elems =
      curr_frame_.getSemanticElement(SemanticLabel::kDashLine);
  const auto &arrow_elems =
      curr_frame_.getSemanticElement(SemanticLabel::kArrowLine);
  const size_t slot_count = slot_elems.size();
  const size_t dash_count = dash_elems.size();
  const size_t arrow_count = arrow_elems.size();
  const size_t total_num = slot_count + dash_count + arrow_count;
  quality.candidates = total_num;
  if (total_num < kMinIcpCorrespondenceForSolve) {
    return false;
  }

  const Eigen::Affine3d T_predict =
      curr_frame_.T_world_ipm_ * T_vehicle_ipm_.inverse();
  const Eigen::Affine3d dT_legacy = legacy_pose * T_predict.inverse();
  const Eigen::Matrix2d R_legacy = dT_legacy.rotation().topLeftCorner<2, 2>();
  const Eigen::Vector2d t_legacy_grid =
      dT_legacy.translation().head<2>() * kPixelScaleInv;

  std::vector<Eigen::Vector2f> transformed_slot_points;
  std::vector<Eigen::Vector2f> transformed_dash_points;
  std::vector<Eigen::Vector2f> transformed_arrow_points;
  transformed_slot_points.reserve(slot_count);
  transformed_dash_points.reserve(dash_count);
  transformed_arrow_points.reserve(arrow_count);

  auto transformToLegacy = [&](const auto &src_points,
                               std::vector<Eigen::Vector2f> &dst_points) {
    for (const auto &pt : src_points) {
      const Eigen::Vector2d p_grid(static_cast<double>(pt.x()),
                                   static_cast<double>(pt.y()));
      const Eigen::Vector2d q_grid = R_legacy * p_grid + t_legacy_grid;
      dst_points.emplace_back(q_grid.cast<float>());
    }
  };
  transformToLegacy(slot_elems, transformed_slot_points);
  transformToLegacy(dash_elems, transformed_dash_points);
  transformToLegacy(arrow_elems, transformed_arrow_points);

  double w_slot = 1.0;
  double w_dash = 1.0;
  double w_arrow = 1.0;
  {
    float ws = 1.0f;
    float wd = 1.0f;
    float wa = 1.0f;
    computeLabelWeights(primitive_weight_mode_, slot_count, dash_count,
                        arrow_count, total_num, ws, wd, wa);
    w_slot = static_cast<double>(ws);
    w_dash = static_cast<double>(wd);
    w_arrow = static_cast<double>(wa);
  }

  auto pointResidualFromKdtree = [&](const Eigen::Vector2f &query,
                                     pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree,
                                     const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                     float &dist_grid, Eigen::Vector2f &grad) {
    pcl::PointXYZ search_point(query.x(), query.y(), 0.0f);
    std::vector<int> indices(1);
    std::vector<float> distances(1);
    const int found = kdtree.nearestKSearch(search_point, 1, indices, distances);
    if (!found || distances[0] >= kIcpDistanceThreshold) {
      return false;
    }
    dist_grid = std::sqrt(std::max(0.0f, distances[0]));
    const auto &map_pt = cloud->points[indices[0]];
    grad = Eigen::Vector2f(query.x() - map_pt.x, query.y() - map_pt.y);
    const float grad_norm = grad.norm();
    if (grad_norm < 1e-4f) {
      return false;
    }
    grad /= grad_norm;
    return true;
  };

  auto accumulateResidual = [&](double label_weight, float dist_grid,
                                const Eigen::Vector2f &grad, Eigen::Matrix2d &H,
                                Eigen::Vector2d &b, size_t &valid_num) {
    const float residual_meter = dist_grid * static_cast<float>(kPixelScale);
    const double w = label_weight * static_cast<double>(huberWeight(residual_meter));
    const Eigen::Vector2d J = grad.cast<double>();
    H.noalias() += w * (J * J.transpose());
    b.noalias() += w * static_cast<double>(dist_grid) * J;
    ++valid_num;
  };

  auto evaluateResiduals = [&](const Eigen::Vector2d &delta_grid, size_t &valid_num,
                               size_t &inlier_num, double &sum_sq_meter) {
    valid_num = 0;
    inlier_num = 0;
    sum_sq_meter = 0.0;

    for (const auto &q : transformed_slot_points) {
      const Eigen::Vector2f q_shifted = q + delta_grid.cast<float>();
      float dist_grid = 0.0f;
      Eigen::Vector2f grad = Eigen::Vector2f::Zero();
      if (!sampleSlotDistanceField(q_shifted, dist_grid, grad) ||
          dist_grid > kDistanceFieldMaxSampleDistGrid) {
        continue;
      }
      ++valid_num;
      if (dist_grid <= kDistanceFieldInlierDistGrid) {
        ++inlier_num;
        const double residual_meter = dist_grid * kPixelScale;
        sum_sq_meter += residual_meter * residual_meter;
      }
    }
    for (const auto &q : transformed_dash_points) {
      const Eigen::Vector2f q_shifted = q + delta_grid.cast<float>();
      float dist_grid = 0.0f;
      Eigen::Vector2f grad = Eigen::Vector2f::Zero();
      if (!pointResidualFromKdtree(q_shifted, kdtree_dash_line_,
                                   dash_line_cloud_in_, dist_grid, grad) ||
          dist_grid > kDistanceFieldMaxSampleDistGrid) {
        continue;
      }
      ++valid_num;
      if (dist_grid <= kDistanceFieldInlierDistGrid) {
        ++inlier_num;
        const double residual_meter = dist_grid * kPixelScale;
        sum_sq_meter += residual_meter * residual_meter;
      }
    }
    for (const auto &q : transformed_arrow_points) {
      const Eigen::Vector2f q_shifted = q + delta_grid.cast<float>();
      float dist_grid = 0.0f;
      Eigen::Vector2f grad = Eigen::Vector2f::Zero();
      if (!pointResidualFromKdtree(q_shifted, kdtree_arrow_line_,
                                   arrow_line_cloud_in_, dist_grid, grad) ||
          dist_grid > kDistanceFieldMaxSampleDistGrid) {
        continue;
      }
      ++valid_num;
      if (dist_grid <= kDistanceFieldInlierDistGrid) {
        ++inlier_num;
        const double residual_meter = dist_grid * kPixelScale;
        sum_sq_meter += residual_meter * residual_meter;
      }
    }
  };

  Eigen::Vector2d delta_grid = Eigen::Vector2d::Zero();
  double final_kappa = std::numeric_limits<double>::quiet_NaN();
  constexpr int max_iter = 8;
  for (int iter = 0; iter < max_iter; ++iter) {
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    Eigen::Vector2d b = Eigen::Vector2d::Zero();
    size_t valid_num = 0;

    for (const auto &q : transformed_slot_points) {
      const Eigen::Vector2f q_shifted = q + delta_grid.cast<float>();
      float dist_grid = 0.0f;
      Eigen::Vector2f grad = Eigen::Vector2f::Zero();
      if (!sampleSlotDistanceField(q_shifted, dist_grid, grad) ||
          dist_grid > kDistanceFieldMaxSampleDistGrid) {
        continue;
      }
      accumulateResidual(w_slot, dist_grid, grad, H, b, valid_num);
    }
    for (const auto &q : transformed_dash_points) {
      const Eigen::Vector2f q_shifted = q + delta_grid.cast<float>();
      float dist_grid = 0.0f;
      Eigen::Vector2f grad = Eigen::Vector2f::Zero();
      if (!pointResidualFromKdtree(q_shifted, kdtree_dash_line_,
                                   dash_line_cloud_in_, dist_grid, grad) ||
          dist_grid > kDistanceFieldMaxSampleDistGrid) {
        continue;
      }
      accumulateResidual(w_dash, dist_grid, grad, H, b, valid_num);
    }
    for (const auto &q : transformed_arrow_points) {
      const Eigen::Vector2f q_shifted = q + delta_grid.cast<float>();
      float dist_grid = 0.0f;
      Eigen::Vector2f grad = Eigen::Vector2f::Zero();
      if (!pointResidualFromKdtree(q_shifted, kdtree_arrow_line_,
                                   arrow_line_cloud_in_, dist_grid, grad) ||
          dist_grid > kDistanceFieldMaxSampleDistGrid) {
        continue;
      }
      accumulateResidual(w_arrow, dist_grid, grad, H, b, valid_num);
    }

    if (valid_num < kMinIcpCorrespondenceForSolve) {
      return false;
    }

    const double prior_w =
        1.0 / (kDistanceFieldRefinePriorSigmaGrid * kDistanceFieldRefinePriorSigmaGrid);
    H(0, 0) += prior_w;
    H(1, 1) += prior_w;
    b(0) += prior_w * delta_grid.x();
    b(1) += prior_w * delta_grid.y();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig_solver(H);
    if (eig_solver.info() == Eigen::Success) {
      const Eigen::Vector2d eig_vals = eig_solver.eigenvalues().cwiseAbs();
      const double eig_min = std::max(1e-9, eig_vals.minCoeff());
      const double eig_max = eig_vals.maxCoeff();
      final_kappa = eig_max / eig_min;
    }

    const Eigen::Vector2d step = -H.ldlt().solve(b);
    if (!step.allFinite()) {
      return false;
    }

    Eigen::Vector2d step_clamped = step;
    step_clamped.x() = std::max(
        -static_cast<double>(kDistanceFieldRefineMaxDeltaStepGrid),
        std::min(static_cast<double>(kDistanceFieldRefineMaxDeltaStepGrid),
                 step_clamped.x()));
    step_clamped.y() = std::max(
        -static_cast<double>(kDistanceFieldRefineMaxDeltaStepGrid),
        std::min(static_cast<double>(kDistanceFieldRefineMaxDeltaStepGrid),
                 step_clamped.y()));
    delta_grid += step_clamped;

    const double delta_norm = delta_grid.norm();
    if (delta_norm > kDistanceFieldRefineMaxCorrectionGrid) {
      delta_grid *= kDistanceFieldRefineMaxCorrectionGrid / delta_norm;
    }

    if (step_clamped.norm() < 1e-3) {
      break;
    }
  }

  size_t valid_num = 0;
  size_t inlier_num = 0;
  double sum_sq_meter = 0.0;
  evaluateResiduals(delta_grid, valid_num, inlier_num, sum_sq_meter);

  if (valid_num < kMinIcpCorrespondenceForSolve ||
      inlier_num < kMinIcpCorrespondenceForSolve) {
    return false;
  }

  quality.matched = inlier_num;
  quality.inlier_ratio =
      static_cast<double>(inlier_num) / static_cast<double>(total_num);
  quality.rmse_meter =
      std::sqrt(sum_sq_meter / static_cast<double>(quality.matched));
  if (!std::isfinite(quality.rmse_meter)) {
    return false;
  }

  const double rmse_norm = std::min(2.0, quality.rmse_meter / 1.0);
  quality.quality_score =
      0.5 * rmse_norm + 0.5 * (1.0 - quality.inlier_ratio);
  quality.hessian_kappa = final_kappa;

  const Eigen::Vector3d refine_t(delta_grid.x() * kPixelScale,
                                 delta_grid.y() * kPixelScale, 0.0);
  refined_pose = Eigen::Translation3d(refine_t) * legacy_pose;
  return true;
}

bool AvpLocalization::imageRegistration(Eigen::Affine3d &T_world_vehicle,
                                        FilterObservationQuality &quality) {
  Eigen::Affine3d legacy_pose = Eigen::Affine3d::Identity();
  FilterObservationQuality legacy_quality;
  const bool legacy_valid = imageRegistrationLegacy(legacy_pose, legacy_quality);

  if (!legacy_valid) {
    quality = legacy_quality;
    return false;
  }

  T_world_vehicle = legacy_pose;
  quality = legacy_quality;

  const bool try_distance_field_refine =
      config_.enable_distance_field_registration;
  if (!try_distance_field_refine) {
    return true;
  }

  Eigen::Affine3d refined_pose = legacy_pose;
  FilterObservationQuality refined_quality;
  const bool refined_valid = refineLegacyTranslationWithDistanceField(
      legacy_pose, refined_pose, refined_quality);
  if (!refined_valid) {
    ROS_WARN_STREAM_THROTTLE(
        1.0,
        "Distance-field refine failed, use legacy ICP. cand="
            << refined_quality.candidates << ", matched="
            << refined_quality.matched);
    return true;
  }

  const double refine_shift =
      (refined_pose.translation().head<2>() -
       legacy_pose.translation().head<2>())
          .norm();

  const bool shift_ok = refine_shift <= 0.25;
  const bool quality_better =
      refined_quality.quality_score + 0.005 < legacy_quality.quality_score;
  const bool rmse_not_worse =
      refined_quality.rmse_meter <= legacy_quality.rmse_meter + 0.02;
  if (shift_ok && quality_better && rmse_not_worse) {
    T_world_vehicle = refined_pose;
    quality = refined_quality;
  }
  return true;
}
