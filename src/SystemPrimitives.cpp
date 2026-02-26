#include "../include/SystemPrimitives.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>

namespace {

State interpolation(const State &start, const State &end, double time) {
  const double factor = (time - start.time) / (end.time - start.time);
  State ret;
  ret.time = time;
  ret.x = start.x + factor * (end.x - start.x);
  ret.y = start.y + factor * (end.y - start.y);
  ret.yaw = start.yaw + factor * (end.yaw - start.yaw);
  ret.P = start.P + factor * (end.P - start.P);
  return ret;
}

double computeCenterDistance(cv::Vec4i line1, cv::Vec4i line2) {
  cv::Point2f p1((line1[0] + line1[2]) / 2.0f, (line1[1] + line1[3]) / 2.0f);
  cv::Point2f p2((line2[0] + line2[2]) / 2.0f, (line2[1] + line2[3]) / 2.0f);
  return cv::norm(p1 - p2);
}

cv::Point2f computeIntersect(cv::Vec4i line1, cv::Vec4i line2) {
  int x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
  int x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];
  float denom =
      static_cast<float>((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
  cv::Point2f intersect((x1 * y2 - y1 * x2) * (x3 - x4) -
                            (x1 - x2) * (x3 * y4 - y3 * x4),
                        (x1 * y2 - y1 * x2) * (y3 - y4) -
                            (y1 - y2) * (x3 * y4 - y3 * x4));
  return intersect / denom;
}

void interpolatePoints(std::vector<Eigen::Vector3d> &points) {
  if (points.size() < 2) {
    points = {};
  }

  std::vector<Eigen::Vector3d> result;
  for (size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector3d &p1 = points[i];
    const Eigen::Vector3d &p2 = points[(i + 1) % 4];

    result.push_back(p1);
    int insertionsPerSegment =
        std::max(1, static_cast<int>((p1 - p2).norm() / 0.05));
    for (int j = 1; j <= insertionsPerSegment; ++j) {
      double t = static_cast<double>(j) / (insertionsPerSegment + 1);
      double nx = p1(0) + t * (p2(0) - p1(0));
      double ny = p1(1) + t * (p2(1) - p1(1));
      double nz = p1(2) + t * (p2(2) - p1(2));
      result.emplace_back(nx, ny, nz);
    }
  }

  result.push_back(points.back());
  result.swap(points);
}

}  // namespace

void StateInterpolation::Push(const State &pose) {
  if (pose.time > EarliestTime()) {
    poses_.push_back(pose);
  }
}

void StateInterpolation::TrimBefore(double time) {
  while (!poses_.empty() && poses_.front().time < time) {
    poses_.pop_front();
  }
}

double StateInterpolation::EarliestTime() const {
  return poses_.empty() ? 0.0 : poses_.front().time;
}

bool StateInterpolation::LookUp(State &pose) const {
  if (poses_.empty() || pose.time < poses_.front().time ||
      pose.time > poses_.back().time) {
    if (poses_.empty()) {
      std::cout << "poses empty!" << std::endl;
    } else {
      std::cout << std::to_string(pose.time)
                << ", poses have: " << std::to_string(poses_.front().time)
                << " -- " << std::to_string(poses_.back().time) << std::endl;
    }
    return false;
  }

  auto end = std::lower_bound(
      poses_.begin(), poses_.end(), pose.time,
      [](const State &_pose, const double t) { return _pose.time < t; });
  if (end == poses_.end()) {
    end = std::prev(end);
  }

  pose = interpolation(*std::prev(end), *end, pose.time);
  return true;
}

FilterBase::FilterBase(double time, double x, double y, double yaw) {
  Qn_ << v_std * v_std, 0, 0, yaw_rate_std * yaw_rate_std;
  Rn_match_ << x_match_std * x_match_std, 0, 0, 0,
      y_match_std * y_match_std, 0, 0, 0, yaw_match_std * yaw_match_std;

  state_.x = x;
  state_.y = y;
  state_.yaw = yaw;
  state_.P = 1.0 * Eigen::Matrix3d::Identity();
  state_.time = time;
  init_ = true;
  state_interpolation_.Push(state_);
}

double FilterBase::normalizeAngle(double angle) {
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

void FilterBase::predict(const double &time, const double &velocity,
                         const double &yaw_rate) {
  const double dt = time - state_.time;
  if (dt == 0.0) {
    return;
  }

  Eigen::Matrix3d A;
  A << 0, 0, -velocity * sin(state_.yaw + M_PI_2), 0, 0,
      velocity * cos(state_.yaw + M_PI_2), 0, 0, 0;
  Eigen::Matrix<double, 3, 2> U;
  U << cos(state_.yaw), 0, sin(state_.yaw), 0, 0, 1;

  const Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + dt * A;
  Eigen::Matrix<double, 3, 2> V;
  V = dt * U;

  state_.x += dt * velocity * cos(state_.yaw + M_PI_2);
  state_.y += dt * velocity * sin(state_.yaw + M_PI_2);
  state_.yaw = normalizeAngle(state_.yaw + dt * yaw_rate);

  state_.P = F * state_.P * F.transpose() + V * Qn_ * V.transpose();
  state_.time = time;
  state_interpolation_.Push(state_);
}

Eigen::Vector3d FilterBase::buildResidual(const double &m_x, const double &m_y,
                                          const double &m_yaw) const {
  Eigen::Vector3d residual;
  residual(0) = m_x - state_.x;
  residual(1) = m_y - state_.y;
  residual(2) = normalizeAngle(m_yaw - state_.yaw);
  return residual;
}

void FilterBase::applyJosephUpdate(const Eigen::Matrix3d &P_pred,
                                   const Eigen::Matrix3d &R_eff,
                                   const Eigen::Vector3d &residual) {
  const Eigen::Matrix3d C = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d S = C * P_pred * C.transpose() + R_eff;
  const Eigen::Matrix3d K = P_pred * C.transpose() * S.inverse();
  const Eigen::Vector3d correct = K * residual;

  state_.x += correct(0);
  state_.y += correct(1);
  state_.yaw = normalizeAngle(state_.yaw + correct(2));

  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  state_.P = (I - K * C) * P_pred * (I - K * C).transpose() +
             K * R_eff * K.transpose();
  state_.P = 0.5 * (state_.P + state_.P.transpose());

  printf("Update t = %.4f, x: %.3f, y: %.3f, yaw: %.2f \n", state_.time,
         state_.x, state_.y, state_.yaw);
}

void FilterBase::setLastStepDiagnostics(double nis, double phi, double gamma) {
  last_step_diag_.nis = nis;
  last_step_diag_.phi = phi;
  last_step_diag_.gamma = gamma;
  constexpr double kNisChi2Threshold = 7.8147;
  last_step_diag_.nis_outlier =
      std::isfinite(nis) && nis > kNisChi2Threshold;
}

EKF::EKF(double time, double x, double y, double yaw)
    : FilterBase(time, x, y, yaw) {}

void EKF::update(const double &m_x, const double &m_y, const double &m_yaw) {
  const Eigen::Matrix3d P_pred = state_.P;
  const Eigen::Matrix3d R_eff = Rn_match_;
  const Eigen::Vector3d residual = buildResidual(m_x, m_y, m_yaw);
  const Eigen::Matrix3d S = P_pred + R_eff;
  const double nis = (residual.transpose() * S.inverse() * residual)(0, 0);
  setLastStepDiagnostics(nis, 1.0, 1.0);
  applyJosephUpdate(P_pred, R_eff, residual);
}

AEKF::AEKF(double time, double x, double y, double yaw)
    : FilterBase(time, x, y, yaw) {}

void AEKF::update(const double &m_x, const double &m_y, const double &m_yaw) {
  const Eigen::Matrix3d P_pred = state_.P;
  Eigen::Matrix3d R_eff = Rn_match_;
  const Eigen::Vector3d residual = buildResidual(m_x, m_y, m_yaw);

  const Eigen::Matrix3d C = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d S = C * P_pred * C.transpose() + R_eff;
  const double nis =
      std::max(0.0, (residual.transpose() * S.inverse() * residual)(0, 0));
  const double gamma =
      std::min(kAekfMaxScale, std::max(1.0, nis / kAekfChi2Threshold));
  R_eff *= gamma;

  const Eigen::Matrix3d S_eff = C * P_pred * C.transpose() + R_eff;
  const double nis_eff =
      std::max(0.0, (residual.transpose() * S_eff.inverse() * residual)(0, 0));
  setLastStepDiagnostics(nis_eff, 1.0, gamma);

  applyJosephUpdate(P_pred, R_eff, residual);
}

STEKF::STEKF(double time, double x, double y, double yaw)
    : FilterBase(time, x, y, yaw) {}

void STEKF::update(const double &m_x, const double &m_y, const double &m_yaw) {
  Eigen::Matrix3d P_pred = state_.P;
  const Eigen::Matrix3d R_eff = Rn_match_;
  const Eigen::Vector3d residual = buildResidual(m_x, m_y, m_yaw);

  const Eigen::Matrix3d C = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d S = C * P_pred * C.transpose() + R_eff;
  const double predicted_energy = std::max(1e-9, S.trace());
  const double innovation_energy = residual.squaredNorm();
  const double phi =
      std::min(kStekfMaxPhi, std::max(1.0, innovation_energy / predicted_energy));
  P_pred *= phi;

  const Eigen::Matrix3d S_eff = C * P_pred * C.transpose() + R_eff;
  const double nis_eff =
      std::max(0.0, (residual.transpose() * S_eff.inverse() * residual)(0, 0));
  setLastStepDiagnostics(nis_eff, phi, 1.0);

  applyJosephUpdate(P_pred, R_eff, residual);
}

DMAEKF::DMAEKF(double time, double x, double y, double yaw,
               const DmaFilterConfig &config)
    : FilterBase(time, x, y, yaw), config_(config) {}

void DMAEKF::setObservationQuality(const FilterObservationQuality &quality) {
  obs_quality_ = quality;
  has_quality_ = true;
}

void DMAEKF::update(const double &m_x, const double &m_y,
                    const double &m_yaw) {
  const Eigen::Vector3d residual = buildResidual(m_x, m_y, m_yaw);

  const Eigen::Matrix3d vvT = residual * residual.transpose();
  const double rho = std::min(0.999, std::max(0.5, config_.ewma_rho));
  innovation_cov_ = rho * innovation_cov_ + (1.0 - rho) * vvT;

  const Eigen::Matrix3d P_pred_base = state_.P;
  const Eigen::Matrix3d S_base = P_pred_base + Rn_match_;
  const double gamma_inn_raw =
      innovation_cov_.trace() / std::max(1e-9, S_base.trace());
  const double phi_instant =
      residual.squaredNorm() / std::max(1e-9, S_base.trace());
  const bool quality_bad =
      has_quality_ && obs_quality_.quality_score > config_.quality_trigger;

  Eigen::Matrix3d P_pred_eff = P_pred_base;
  double phi_used = 1.0;
  if (config_.enable_strong_tracking && !quality_bad) {
    const double phi_raw = std::max(phi_instant, gamma_inn_raw);
    const double phi = std::min(config_.phi_max, std::max(1.0, phi_raw));
    P_pred_eff *= phi;
    phi_used = phi;
  }

  Eigen::Matrix3d R_eff = Rn_match_;
  double gamma_used = 1.0;
  if (config_.enable_observation_weighting) {
    double gamma = 1.0;

    if (has_quality_) {
      const double safe_ratio =
          std::max(0.0, std::min(1.0, obs_quality_.inlier_ratio));
      if (obs_quality_.quality_score > config_.quality_trigger) {
        const double q_excess =
            obs_quality_.quality_score - config_.quality_trigger;
        gamma = std::max(gamma, 1.0 + 3.0 * config_.quality_alpha * q_excess);
      }
      if (safe_ratio < 0.65) {
        gamma = std::max(gamma,
                         1.0 + 4.0 * config_.quality_alpha * (0.65 - safe_ratio));
      }
    }

    if (gamma_inn_raw > config_.innovation_trigger) {
      const double inn_excess = gamma_inn_raw - config_.innovation_trigger;
      gamma = std::max(gamma, 1.0 + config_.innovation_alpha * inn_excess);
    }

    if (has_quality_ && obs_quality_.quality_score < 0.12 &&
        gamma_inn_raw < 0.95) {
      gamma = std::min(gamma, 0.90);
    }

    gamma = std::min(config_.gamma_max, std::max(0.70, gamma));
    R_eff *= gamma;
    gamma_used = gamma;
  }

  const Eigen::Matrix3d S_eff = P_pred_eff + R_eff;
  const double nis_eff =
      std::max(0.0, (residual.transpose() * S_eff.inverse() * residual)(0, 0));
  setLastStepDiagnostics(nis_eff, phi_used, gamma_used);

  applyJosephUpdate(P_pred_eff, R_eff, residual);
}

FilterMode parseFilterMode(const std::string &mode_name) {
  std::string mode = mode_name;
  std::transform(mode.begin(), mode.end(), mode.begin(),
                 [](const unsigned char c) { return std::tolower(c); });
  if (mode == "ekf") {
    return FilterMode::kEKF;
  }
  if (mode == "aekf") {
    return FilterMode::kAEKF;
  }
  if (mode == "stekf") {
    return FilterMode::kSTEKF;
  }
  if (mode == "dmaekf") {
    return FilterMode::kDMAEKF;
  }
  return FilterMode::kEKF;
}

const char *filterModeName(FilterMode mode) {
  switch (mode) {
    case FilterMode::kEKF:
      return "ekf";
    case FilterMode::kAEKF:
      return "aekf";
    case FilterMode::kSTEKF:
      return "stekf";
    case FilterMode::kDMAEKF:
      return "dmaekf";
    default:
      return "ekf";
  }
}

std::unique_ptr<FilterBase> createFilter(FilterMode mode, double time, double x,
                                         double y, double yaw,
                                         const DmaFilterConfig &dma_config) {
  switch (mode) {
    case FilterMode::kEKF:
      return std::unique_ptr<FilterBase>(new EKF(time, x, y, yaw));
    case FilterMode::kAEKF:
      return std::unique_ptr<FilterBase>(new AEKF(time, x, y, yaw));
    case FilterMode::kSTEKF:
      return std::unique_ptr<FilterBase>(new STEKF(time, x, y, yaw));
    case FilterMode::kDMAEKF:
      return std::unique_ptr<FilterBase>(
          new DMAEKF(time, x, y, yaw, dma_config));
    default:
      return std::unique_ptr<FilterBase>(new EKF(time, x, y, yaw));
  }
}

const std::unordered_set<Eigen::Vector3i, Vector3iHash> &
Frame::getSemanticElement(SemanticLabel label) const {
  return semantic_grid_map_.at(label);
}

void Frame::addSemanticElement(SemanticLabel label, const Eigen::Vector3d &pt) {
  Eigen::Vector3i grid_pt;
  grid_pt.x() = std::round(pt.x() * kPixelScaleInv);
  grid_pt.y() = std::round(pt.y() * kPixelScaleInv);
  grid_pt.z() = std::round(pt.z() * kPixelScaleInv);
  semantic_grid_map_[label].insert(grid_pt);
}

void Frame::clearGridMap() {
  for (auto &it : semantic_grid_map_) {
    it.clear();
  }
}

cv::Mat skeletonize(const cv::Mat &img) {
  cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
  cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
  cv::Mat temp, eroded;
  do {
    cv::erode(img, eroded, element);
    cv::dilate(eroded, temp, element);
    cv::subtract(img, temp, temp);
    cv::bitwise_or(skel, temp, skel);
    eroded.copyTo(img);
  } while (0 != cv::countNonZero(img));
  return skel;
}

void removeIsolatedPixels(cv::Mat &src, int min_neighbors) {
  CV_Assert(src.type() == CV_8UC1);
  cv::Mat dst = src.clone();
  for (int y = 1; y < src.rows - 1; ++y) {
    for (int x = 1; x < src.cols - 1; ++x) {
      if (src.at<uchar>(y, x) > 0) {
        int neighbor_count = 0;
        for (int ny = -1; ny <= 1; ++ny) {
          for (int nx = -1; nx <= 1; ++nx) {
            if (ny == 0 && nx == 0) {
              continue;
            }
            if (src.at<uchar>(y + ny, x + nx) > 0) {
              ++neighbor_count;
            }
          }
        }
        if (neighbor_count < min_neighbors) {
          dst.at<uchar>(y, x) = 0;
        }
      }
    }
  }
  src = dst;
}

bool isSlotShortLine(const cv::Point2f &point1, const cv::Point2f &point2,
                     const cv::Mat &image) {
  cv::LineIterator it(image, point1, point2, 4);
  int positiveIndex = 0;
  const double len = cv::norm(point1 - point2);
  int delta = 10;
  if (std::fabs(len - kShortLinePixelDistance) < delta) {
    for (int i = 0; i < it.count; ++i, ++it) {
      int color =
          image.at<uchar>(std::round(it.pos().y), std::round(it.pos().x));
      if (color > 0) {
        positiveIndex++;
      }
    }
    if (positiveIndex > kShortLineDetectPixelNumberMin) {
      return true;
    }
  }
  return false;
}

bool isSlotLongLine(const cv::Mat &line_img, const cv::Point2f &start,
                    const cv::Point2f &dir) {
  int cnt{0};
  cv::Point2f pt(start);
  for (int l = 1; l < kLongLinePixelDistance; ++l) {
    pt += dir;
    if (pt.y <= 0 || pt.y >= kIPMImgHeight || pt.x <= 0 || pt.x >= kIPMImgWidth) {
      continue;
    }
    if (line_img.at<uchar>(pt) > 0) {
      ++cnt;
    }
  }
  return cnt > kLongLineDetectPixelNumberMin;
}

std::vector<Slot> Map::getAllSlots() const {
  std::vector<Slot> all_slots(slots_.size());
  for (int i = 0; i < slots_.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      all_slots[i].corners_[j] = corner_points_[slots_[i][j]].center();
    }
  }
  return std::move(all_slots);
}

const std::unordered_set<Eigen::Vector3i, Vector3iHash> &
Map::getSemanticElement(SemanticLabel label) const {
  return semantic_grid_map_.at(label);
}

void Map::addSemanticElement(SemanticLabel label, const Eigen::Vector3d &pt) {
  Eigen::Vector3i grid_pt;
  grid_pt.x() = std::round(pt.x() * kPixelScaleInv);
  grid_pt.y() = std::round(pt.y() * kPixelScaleInv);
  grid_pt.z() = std::round(pt.z() * kPixelScaleInv);
  semantic_grid_map_[label].insert(grid_pt);
}

void Map::addSlot(const Slot &new_slot) {
  SlotIndex new_slot_index;
  Eigen::Vector3d new_slot_center = Eigen::Vector3d::Zero();
  for (int i = 0; i < 4; ++i) {
    new_slot_index[i] = addSlotCorner(new_slot.corners_[i]);
    new_slot_center += new_slot.corners_[i];
  }
  new_slot_center *= 0.25;

  bool isNewSlot = true;
  Eigen::Vector3d slot_center;
  for (const auto slot_index : slots_) {
    slot_center = Eigen::Vector3d::Zero();
    for (int j = 0; j < 4; ++j) {
      slot_center += corner_points_[slot_index[j]].center();
    }
    slot_center *= 0.25;
    if ((slot_center - new_slot_center).norm() < 1.0) {
      isNewSlot = false;
      break;
    }
  }
  if (isNewSlot) {
    slots_.push_back(new_slot_index);
  }
}

size_t Map::addSlotCorner(const Eigen::Vector3d &pt) {
  size_t index = corner_points_.size();
  for (int i = 0; i < corner_points_.size(); ++i) {
    if (corner_points_[i].absorb(pt, kNeighborDistance * kPixelScale)) {
      index = i;
      break;
    }
  }
  if (corner_points_.size() == index) {
    corner_points_.emplace_back(pt);
    bounding_box_.extend(pt);
  }
  return index;
}

void Map::save(const std::string &filename) const {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs.is_open()) {
    std::cerr << "Unable to open map file: " << filename << std::endl;
    return;
  }

  io::saveVector(ofs, corner_points_);
  io::saveVector(ofs, slots_);
  std::cout << "save corners: " << corner_points_.size() << std::endl;
  std::cout << "save slots: " << slots_.size() << std::endl;

  std::vector<Eigen::Vector3i> semantic_grid;
  for (int i = 0; i < SemanticLabel::kTotalLabelNum; ++i) {
    semantic_grid.assign(semantic_grid_map_[i].begin(), semantic_grid_map_[i].end());
    io::saveVector(ofs, semantic_grid);
    std::cout << "save label_" << i << ": " << semantic_grid.size() << std::endl;
  }
  ofs.close();
  std::cout << "avp map saved to " << filename << std::endl;
}

void Map::load(const std::string &filename) {
  if (!corner_points_.empty()) {
    std::cerr << "map have already loaded, fail to load map file: " << filename
              << std::endl;
    return;
  }
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "Unable to load map file: " << filename << std::endl;
    return;
  }
  io::loadVector(ifs, corner_points_);
  io::loadVector(ifs, slots_);
  for (const auto &corner : corner_points_) {
    bounding_box_.extend(corner.center());
  }
  std::cout << "load corners: " << corner_points_.size() << std::endl;
  std::cout << "load slots: " << slots_.size() << std::endl;

  std::vector<Eigen::Vector3i> semantic_grid;
  for (int i = 0; i < SemanticLabel::kTotalLabelNum; ++i) {
    io::loadVector(ifs, semantic_grid);
    semantic_grid_map_[i].insert(semantic_grid.begin(), semantic_grid.end());
    std::cout << "load label_" << i << ": " << semantic_grid.size() << std::endl;
  }
  ifs.close();
  if (semantic_grid_map_[kSlot].empty()) {
    discretizeLine(getAllSlots());
  }

  std::cout << "avp map loaded from " << filename << std::endl;
}

void Map::discretizeLine(const std::vector<Slot> &slots) {
  for (const auto &slot : slots) {
    std::vector<Eigen::Vector3d> single_slot;
    for (int j = 0; j < 4; ++j) {
      single_slot.push_back(slot.corners_[j]);
    }
    interpolatePoints(single_slot);
    for (const auto &pt : single_slot) {
      addSemanticElement(SemanticLabel::kSlot, pt);
    }
  }
  std::cout << "kSlot discretize points: "
            << getSemanticElement(SemanticLabel::kSlot).size() << std::endl;
}
