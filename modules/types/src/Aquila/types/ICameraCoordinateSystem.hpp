#pragma once
#include <MetaObject/params/ICoordinateSystem.hpp>
#include <opencv2/core.hpp>

namespace aq {
class ICameraCoordinateSystem : public mo::ICoordinateSystem {
public:
    ICameraCoordinateSystem(const std::string& name);
    virtual const cv::Matx33d& getCameraMatrix() const          = 0;
    virtual const cv::Matx34d& getProjectionMatrix() const      = 0;
    virtual cv::Vec2f project3dTo2D(const cv::Vec3f& xyz) const = 0;
    virtual void project3dTo2D(const cv::Mat& world_points, cv::Mat& image_points) const = 0;
    virtual void project3dTo2D(const cv::cuda::GpuMat& world_points, cv::cuda::GpuMat& image_points, cv::cuda::Stream& stream) const = 0;
    inline double fx() const;
    inline double fy() const;
    inline double cx() const;
    inline double cy() const;
    inline double tx() const;
    inline double ty() const;
};

inline double ICameraCoordinateSystem::fx() const { return getProjectionMatrix()(0, 0); }
inline double ICameraCoordinateSystem::fy() const { return getProjectionMatrix()(1, 1); }
inline double ICameraCoordinateSystem::cx() const { return getProjectionMatrix()(0, 2); }
inline double ICameraCoordinateSystem::cy() const { return getProjectionMatrix()(1, 2); }
inline double ICameraCoordinateSystem::tx() const { return getProjectionMatrix()(0, 3); }
inline double ICameraCoordinateSystem::ty() const { return getProjectionMatrix()(1, 3); }
}
