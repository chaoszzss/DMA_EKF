#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <opencv2/opencv.hpp>

// parameters for slot
constexpr int kIPMImgHeight = 1000;
constexpr int kIPMImgWidth = 1000;

constexpr int kShortLinePixelDistance = 210;
constexpr int kLongLinePixelDistance = 320;
constexpr int kLongLineDetectPixelNumberMin = 200;
constexpr int kShortLineDetectPixelNumberMin = 50;

constexpr int kNeighborDistance = 100;  // points nearer than this are merged
constexpr double kPixelScale = 0.02;
constexpr double kPixelScaleInv = 1.0 / kPixelScale;

// ipm_label values must stay aligned with avp_mapping map build.
const uchar kDashGray = 127;
const uchar kArrowGray = 225;
const uchar kSlotGray = 151;
const uchar kSlotGray1 = 150;

// Allow small quantization drift from compression / color conversion.
constexpr int kSemanticGrayTolerance = 2;

constexpr double kToRad = M_PI / 180.0;
constexpr double kToDeg = 180.0 / M_PI;

struct Vector3iHash {
  std::size_t operator()(const Eigen::Vector3i &pt) const {
    return (static_cast<size_t>(pt.x()) << 32) | pt.y();
  }
};

enum SemanticLabel {
  kDashLine = 0,
  kArrowLine,
  kSlot,
  kTotalLabelNum
};

struct Slot {
  Eigen::Vector3d corners_[4];
};

class CornerPoint {
 public:
  CornerPoint() : CornerPoint({0, 0, 0}) {}
  explicit CornerPoint(const Eigen::Vector3d &pt) : center_(pt), count_(1) {}
  bool absorb(const Eigen::Vector3d &pt, const double dist) {
    if ((pt - center()).norm() < dist) {
      center_ += pt;
      ++count_;
      return true;
    }
    return false;
  }
  Eigen::Vector3d center() const { return center_ / count_; }

 private:
  Eigen::Vector3d center_;
  unsigned int count_;
};

cv::Mat skeletonize(const cv::Mat &img);
void removeIsolatedPixels(cv::Mat &src, int min_neighbors = 1);
bool isSlotLongLine(const cv::Mat &line_img, const cv::Point2f &start,
                    const cv::Point2f &dir);
bool isSlotShortLine(const cv::Point2f &point1, const cv::Point2f &point2,
                     const cv::Mat &mask);

namespace io {
template <typename T>
void loadVector(std::ifstream &ifs, std::vector<T> &vec) {
  size_t size{0};
  ifs.read(reinterpret_cast<char *>(&size), sizeof(size));
  vec.resize(size);
  ifs.read(reinterpret_cast<char *>(vec.data()), size * sizeof(T));
}

template <typename T>
void saveVector(std::ofstream &ofs, const std::vector<T> &vec) {
  size_t size = vec.size();
  ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
  ofs.write(reinterpret_cast<const char *>(vec.data()), size * sizeof(T));
}
}  // namespace io

class Frame {
 public:
  Frame() : semantic_grid_map_(SemanticLabel::kTotalLabelNum) {}

  const std::unordered_set<Eigen::Vector3i, Vector3iHash> &getSemanticElement(
      SemanticLabel label) const;
  void addSemanticElement(SemanticLabel label, const Eigen::Vector3d &pt);
  void clearGridMap();

  std::vector<std::unordered_set<Eigen::Vector3i, Vector3iHash>>
  getSemanticGridMap() {
    return semantic_grid_map_;
  }

  Eigen::Affine3d T_world_ipm_;
  Eigen::Vector3d t_update;

 private:
  std::vector<std::unordered_set<Eigen::Vector3i, Vector3iHash>>
      semantic_grid_map_;
};

class Map {
  using SlotIndex = std::array<size_t, 4>;

 public:
  Map() : semantic_grid_map_(SemanticLabel::kTotalLabelNum) {}

  int getWidth() const {
    return (bounding_box_.max().x() - bounding_box_.min().x()) / kPixelScale;
  }
  int getHeight() const {
    return (bounding_box_.max().y() - bounding_box_.min().y()) / kPixelScale;
  }
  Eigen::Vector3d getOrigin() const { return origin_; }

  Eigen::AlignedBox3d getBoundingBox() const { return bounding_box_; }
  std::vector<Slot> getAllSlots() const;
  const std::unordered_set<Eigen::Vector3i, Vector3iHash> &getSemanticElement(
      SemanticLabel label) const;
  void addSemanticElement(SemanticLabel label, const Eigen::Vector3d &pt);
  void addSlot(const Slot &new_slot);
  void save(const std::string &filename) const;
  void load(const std::string &filename);
  void discretizeLine(const std::vector<Slot> &slots);

 private:
  Eigen::Vector3d origin_ = Eigen::Vector3d(-50, -50, 0);

  size_t addSlotCorner(const Eigen::Vector3d &pt);

  std::vector<CornerPoint> corner_points_;
  std::vector<SlotIndex> slots_;
  std::vector<std::unordered_set<Eigen::Vector3i, Vector3iHash>>
      semantic_grid_map_;
  Eigen::AlignedBox3d bounding_box_;
};

template <typename T>
T GetYaw(const Eigen::Quaternion<T> &rotation) {
  const Eigen::Matrix<T, 3, 1> direction =
      rotation * Eigen::Matrix<T, 3, 1>::UnitX();
  return atan2(direction.y(), direction.x());
}

class MapViewer {
 public:
  ~MapViewer() {
    count_ = -1;
    thread_.join();
  }

  explicit MapViewer(const Map &avp_map) {
    bounding_box_ = avp_map.getBoundingBox();
    if (bounding_box_.isEmpty()) {
      bounding_box_.extend(Eigen::Vector3d(0, 0, 0));
    }

    Eigen::Vector3d bl = bounding_box_.min() + Eigen::Vector3d(-1, -1, -1);
    Eigen::Vector3d tr = bounding_box_.max() + Eigen::Vector3d(1, 1, 1);
    int width = (tr.x() - bl.x()) * kPixelScaleInv;
    int height = (tr.y() - bl.y()) * kPixelScaleInv;

    map_ = cv::Mat(height, width, CV_8UC3, cv::Vec3b(91, 91, 91));
    to_map_pixel_ = [=](const Eigen::Vector3d &pt) {
      return cv::Point2f((pt.x() - bl.x()) * kPixelScaleInv,
                         height - (pt.y() - bl.y()) * kPixelScaleInv);
    };

    offset_ = to_map_pixel_(Eigen::Vector3d(0, 0, 0));
    drawGrid(map_, avp_map.getSemanticElement(SemanticLabel::kSlot),
             cv::Vec3b(kSlotGray, kSlotGray, kSlotGray));
    drawGrid(map_, avp_map.getSemanticElement(SemanticLabel::kDashLine),
             cv::Vec3b(kDashGray, kDashGray, kDashGray));
    drawGrid(map_, avp_map.getSemanticElement(SemanticLabel::kArrowLine),
             cv::Vec3b(kArrowGray, kArrowGray, kArrowGray));

    thread_ = std::thread([=] { run(); });
  }

  void showFrame(const Frame &frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    frame_ = frame;
    ++count_;
  }

 private:
  void drawGrid(const cv::Mat &map,
                const std::unordered_set<Eigen::Vector3i, Vector3iHash> &grid,
                cv::Vec3b color) const {
    for (const auto &pt : grid) {
      cv::circle(map, offset_ + cv::Point2f(pt.x(), -pt.y()), 1, color, 1);
    }
  }

  void run() {
    int cnt = count_;
    cv::Mat show = map_.clone();
    cv::namedWindow("map", cv::WINDOW_NORMAL);
    cv::resizeWindow("map", 300, 200);
    while (cnt > 0) {
      if (cnt != count_) {
        cnt = count_;
        std::unique_lock<std::mutex> lock(mutex_);
        auto frame = std::move(frame_);
        lock.unlock();
        auto pos = to_map_pixel_(frame.t_update);
        if (!trajectory_.empty()) {
          cv::line(map_, trajectory_.back(), pos, cv::Scalar(255, 0, 155), 10);
        }
        trajectory_.push_back(pos);
        show = map_.clone();
        drawGrid(show, frame.getSemanticElement(SemanticLabel::kSlot),
                 cv::Vec3b(0, 0, 255));
        drawGrid(show, frame.getSemanticElement(SemanticLabel::kDashLine),
                 cv::Vec3b(0, 255, 0));
        drawGrid(show, frame.getSemanticElement(SemanticLabel::kArrowLine),
                 cv::Vec3b(255, 0, 0));
      }
      cv::imshow("map", show);
      cv::waitKey(50);
    }
    cv::destroyAllWindows();
  }

  std::mutex mutex_;
  int count_ = 1;
  Frame frame_;
  std::thread thread_;

  cv::Mat map_;
  Eigen::AlignedBox3d bounding_box_;
  cv::Point2f offset_;
  std::vector<cv::Point2f> trajectory_;
  std::function<cv::Point2f(const Eigen::Vector3d &)> to_map_pixel_;
};

// parameters for filter process noise
constexpr double v_std = 0.1;
constexpr double yaw_rate_std = 0.01;

// parameters for match noise
constexpr double x_match_std = 0.1;
constexpr double y_match_std = 0.1;
constexpr double yaw_match_std = 5.0 / 180.0 * M_PI;

struct State {
  double time;
  double x;
  double y;
  double yaw;
  Eigen::Matrix3d P;
};

struct FilterObservationQuality {
  std::size_t matched{0};
  std::size_t candidates{0};
  double rmse_meter{std::numeric_limits<double>::infinity()};
  double inlier_ratio{0.0};
  double quality_score{std::numeric_limits<double>::infinity()};
  double hessian_kappa{std::numeric_limits<double>::quiet_NaN()};
};

struct FilterStepDiagnostics {
  double nis{std::numeric_limits<double>::quiet_NaN()};
  double phi{1.0};
  double gamma{1.0};
  bool nis_outlier{false};
};

class StateInterpolation {
 public:
  StateInterpolation() = default;
  void Push(const State &pose);
  void TrimBefore(double time);
  double EarliestTime() const;
  bool LookUp(State &pose) const;

 private:
  std::deque<State> poses_;
};

enum class FilterMode {
  kEKF = 0,
  kAEKF,
  kSTEKF,
  kDMAEKF
};

struct DmaFilterConfig {
  bool enable_strong_tracking{true};
  bool enable_observation_weighting{true};
  double ewma_rho{0.95};
  double phi_max{12.0};
  double gamma_max{6.0};
  double quality_alpha{0.2};
  double innovation_alpha{0.8};
  double quality_trigger{0.35};
  double innovation_trigger{1.10};
};

class FilterBase {
 public:
  FilterBase(double time, double x, double y, double yaw);
  virtual ~FilterBase() = default;

  void predict(const double &time, const double &velocity,
               const double &yaw_rate);
  virtual void update(const double &m_x, const double &m_y,
                      const double &m_yaw) = 0;

  virtual void setObservationQuality(const FilterObservationQuality &quality) {
    (void)quality;
  }
  FilterStepDiagnostics getLastStepDiagnostics() const {
    return last_step_diag_;
  }

  State getState() const { return state_; }
  bool isInit() const { return init_; }
  void setState(const State &state) { state_ = state; }

 protected:
  static double normalizeAngle(double angle);

  Eigen::Vector3d buildResidual(const double &m_x, const double &m_y,
                                const double &m_yaw) const;
  void applyJosephUpdate(const Eigen::Matrix3d &P_pred,
                         const Eigen::Matrix3d &R_eff,
                         const Eigen::Vector3d &residual);
  void setLastStepDiagnostics(double nis, double phi, double gamma);

  State state_;
  StateInterpolation state_interpolation_;
  bool init_{false};
  Eigen::Matrix2d Qn_;
  Eigen::Matrix3d Rn_match_;
  FilterStepDiagnostics last_step_diag_;
};

constexpr double kAekfChi2Threshold = 7.8147;
constexpr double kAekfMaxScale = 20.0;

class EKF final : public FilterBase {
 public:
  EKF(double time, double x, double y, double yaw);
  void update(const double &m_x, const double &m_y,
              const double &m_yaw) override;
};

class AEKF final : public FilterBase {
 public:
  AEKF(double time, double x, double y, double yaw);
  void update(const double &m_x, const double &m_y,
              const double &m_yaw) override;
};

constexpr double kStekfMaxPhi = 20.0;

class STEKF final : public FilterBase {
 public:
  STEKF(double time, double x, double y, double yaw);
  void update(const double &m_x, const double &m_y,
              const double &m_yaw) override;
};

class DMAEKF final : public FilterBase {
 public:
  DMAEKF(double time, double x, double y, double yaw,
         const DmaFilterConfig &config = DmaFilterConfig{});

  void setObservationQuality(const FilterObservationQuality &quality) override;
  void update(const double &m_x, const double &m_y,
              const double &m_yaw) override;

 private:
  DmaFilterConfig config_;
  FilterObservationQuality obs_quality_;
  Eigen::Matrix3d innovation_cov_ = 0.1 * Eigen::Matrix3d::Identity();
  bool has_quality_{false};
};

FilterMode parseFilterMode(const std::string &mode_name);
const char *filterModeName(FilterMode mode);
std::unique_ptr<FilterBase> createFilter(
    FilterMode mode, double time, double x, double y, double yaw,
    const DmaFilterConfig &dma_config = DmaFilterConfig{});
