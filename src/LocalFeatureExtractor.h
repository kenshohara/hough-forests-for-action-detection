#ifndef LOCAL_FEATURE_EXTRACTOR
#define LOCAL_FEATURE_EXTRACTOR

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include <vector>

namespace nuisken {
namespace houghforests {

class LocalFeatureExtractor {
   public:
    enum AXIS { X = 2, Y = 1, T = 0 };

    enum FeatureType { INTENSITY, X_DERIVATIVE, Y_DERIVATIVE, T_DERIVATIVE, FLOW };
    static const int N_CHANNELS_;

   private:
    using Descriptor = std::vector<float>;
    using Feature = std::vector<float>;
    using MultiChannelFeature = std::vector<Feature>;
    using Video = std::vector<cv::Mat1b>;

    cv::VideoCapture videoCapture_;
    std::vector<Video> scaleVideos_;
    std::vector<MultiChannelFeature> scaleChannelFeatures_;
    std::vector<double> scales_;
    int localWidth_;
    int localHeight_;
    int localDuration_;
    int xStep_;
    int yStep_;
    int tStep_;
    int width_;
    int height_;
    int storedStartT_;
    int nStoredFeatureFrames_;
    bool isEnd_;

   public:
    LocalFeatureExtractor(){};

    LocalFeatureExtractor(const std::string& videoFilePath, std::vector<double> scales,
                          int localWidth, int localHeight, int localDuration, int xStep, int yStep,
                          int tStep)
            : videoCapture_(videoFilePath),
              scaleVideos_(scales.size()),
              scaleChannelFeatures_(scales.size(), MultiChannelFeature(N_CHANNELS_)),
              scales_(scales),
              localWidth_(localWidth),
              localHeight_(localHeight),
              localDuration_(localDuration),
              xStep_(xStep),
              yStep_(yStep),
              tStep_(tStep),
              storedStartT_(0),
              nStoredFeatureFrames_(0),
              isEnd_(false) {
        makeLocalSizeOdd(localWidth_);
        makeLocalSizeOdd(localHeight_);
        makeLocalSizeOdd(localDuration_);
    }

    LocalFeatureExtractor(const cv::VideoCapture& videoCapture, std::vector<double> scales,
                          int localWidth, int localHeight, int localDuration, int xStep, int yStep,
                          int tStep)
            : videoCapture_(videoCapture),
              scaleVideos_(scales.size()),
              scaleChannelFeatures_(scales.size(), MultiChannelFeature(N_CHANNELS_)),
              scales_(scales),
              localWidth_(localWidth),
              localHeight_(localHeight),
              localDuration_(localDuration),
              xStep_(xStep),
              yStep_(yStep),
              tStep_(tStep),
              storedStartT_(0),
              nStoredFeatureFrames_(0),
              isEnd_(false) {
        makeLocalSizeOdd(localWidth_);
        makeLocalSizeOdd(localHeight_);
        makeLocalSizeOdd(localDuration_);
    }

    void extractLocalFeatures(std::vector<std::vector<cv::Vec3i>>& scalePoints,
                              std::vector<std::vector<Descriptor>>& scaleDescriptors);

    bool isEnd() const { return isEnd_; }
    int getStoredStartT() const { return storedStartT_; }

    static std::vector<std::string> getFeatureNames() {
        return std::vector<std::string>{"intensity",    "x_derivative", "y_derivative",
                                        "t_derivative", "x_flow",       "y_flow"};
    }

   private:
    void makeLocalSizeOdd(int& size) const;
    void readOriginalScaleVideo();
    void generateScaledVideos();
    void denseSampling(int scaleIndex, std::vector<cv::Vec3i>& points,
                       std::vector<Descriptor>& descriptors) const;
    void deleteOldData();

    void extractFeatures(int scaleIndex, int startFrame, int endFrame);
    void extractIntensityFeature(Feature& features, int scaleIndex, int startFrame, int endFrame);
    void extractXDerivativeFeature(Feature& features, int scaleIndex, int startFrame, int endFrame);
    void extractYDerivativeFeature(Feature& features, int scaleIndex, int startFrame, int endFrame);
    void extractTDerivativeFeature(Feature& features, int scaleIndex, int startFrame, int endFrame);
    void extractFlowFeature(Feature& xFeatures, Feature& yFeatures, int scaleIndex, int startFrame,
                            int endFrame);

    Feature extractIntensityFeature(const cv::Mat1b& frame) const;
    Feature extractXDerivativeFeature(const cv::Mat1b& frame) const;
    Feature extractYDerivativeFeature(const cv::Mat1b& frame) const;
    Feature extractTDerivativeFeature(const cv::Mat1b& prev, const cv::Mat1b& next) const;
    std::vector<Feature> extractFlowFeature(const cv::Mat1b& prev, const cv::Mat1b& next) const;

    Descriptor getLocalFeature(int scaleIndex, const cv::Vec3i& topLeftPoint, int width,
                               int height) const;
    int calculateFeatureIndex(int x, int y, int t, int width, int height) const;

    void visualizeDenseFeature(const std::vector<cv::Vec3i>& points,
                               const std::vector<Descriptor>& features, int width, int height,
                               int duration) const;
};
}
}

#endif