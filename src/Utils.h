#pragma once

#include "Storage.h"
#ifndef UTILS
#define UTILS

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>

#include <map>
#include <tuple>

namespace nuisken {
namespace houghforests {
typedef storage::CoordinateValue<cv::Vec4f> LocalMaximum;
typedef std::vector<LocalMaximum> LocalMaxima;
}

enum Axis { X = 2, Y = 1, T = 0 };

enum ViewType { SINGLE, MULTI };

enum FeatureType {
    HOGHOF,
    HOG3D,
    HOGHOF_WHITEN,
    HOGHOF_PCA,
    HOGHOF_PCA_EACH,
    DT,
    HOGHOF2,
    HF,
    HF_WITHOUT_FLOW,
    HF_PCA
};

std::vector<std::string> getFeatureNames(FeatureType type);

namespace string {
std::vector<std::string> split(const std::string& str, char delimiter);
}

namespace io {
std::string readFile(const std::string& filePath);

void readSTIPFeatures(const std::string& filePath,
                      std::vector<std::vector<Eigen::MatrixXf>>& features,
                      std::vector<cv::Vec3i>& points);

void saveXYTPoint(cv::FileStorage& fileStorage, const std::string& name, const cv::Vec3f& vec);
void saveXYTPoint(cv::FileStorage& fileStorage, const std::string& name, const cv::Vec3f& vec,
                  int cameraNumber);
void saveXYTPoint(cv::FileStorage& fileStorage, const cv::Vec3f& vec);
void saveRectangle(cv::FileStorage& fileStorage, const cv::Rect& rectangle);
cv::Vec3f loadXYTPoint(const cv::FileStorage& fileStorage, const std::string& name);
cv::Vec3f loadXYTPoint(const cv::FileNode& node);
void loadXYTPoint(const cv::FileNode& node, cv::Vec3f& point, int& cameraNumber);
cv::Rect loadRectangle(const cv::FileStorage& fileStorage, const std::string& name);
}

namespace actionvolume {

std::vector<std::pair<double, cv::Vec3f>> readActionPositions(const std::string& filePath);
void readActionPositions(const std::string& filePath,
                         std::vector<std::pair<double, cv::Vec3f>>& actionPositions,
                         std::vector<std::vector<cv::Vec3i>>& contributionPointsOfActionPosition,
                         double rate = 1.0);
}

namespace cveigen {

template <typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void cv2eigen(const cv::Mat& src,
              Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& dst) {
    CV_DbgAssert(src.rows == _rows && src.cols == _cols);
    if (!(dst.Flags & Eigen::RowMajorBit)) {
        cv::Mat _dst(src.cols, src.rows, cv::DataType<_Tp>::type, dst.data(),
                     (size_t)(dst.stride() * sizeof(_Tp)));
        if (src.type() == _dst.type())
            cv::transpose(src, _dst);
        else if (src.cols == src.rows) {
            src.convertTo(_dst, _dst.type());
            cv::transpose(_dst, _dst);
        } else
            cv::Mat(src.t()).convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    } else {
        cv::Mat _dst(src.rows, src.cols, cv::DataType<_Tp>::type, dst.data(),
                     (size_t)(dst.stride() * sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
}
}
}

#endif