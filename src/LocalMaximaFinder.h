#ifndef LOCAL_MAXIMA_FINDER
#define LOCAL_MAXIMA_FINDER

#include "KernelDensityEstimation.h"
#include "VotingSpace.h"
#include "Storage.h"
#include "Utils.h"

#include <vector>
#include <array>

namespace nuisken {
namespace houghforests {

class LocalMaximaFinder {
   private:
    static const int DIMENSION_SIZE_ = 4;
    static const int SPATIAL_DIMENSION_SIZE_ = 2;
    static const int TEMPORAL_DIMENSION_SIZE_ = 1;
    static const int SCALE_DIMENSION_SIZE_ = 1;

    std::vector<int> steps_;
    std::vector<double> scales_;
    double sigma_;
    double tau_;
    double scaleBandwidth_;
    double threshold_;
    int maxIteration_;

    typedef KernelDensityEstimation<float, DIMENSION_SIZE_> KDE;
    typedef std::array<float, DIMENSION_SIZE_> Point;
    typedef std::pair<std::uint32_t, float> Match;

   public:
    LocalMaximaFinder(const std::vector<int>& steps, const std::vector<double>& scales,
                      double sigma, double tau, double scaleBandwidth, double threshold = 0.1,
                      int maxIteration = 50)
            : steps_(steps),
              scales_(scales),
              sigma_(sigma),
              tau_(tau),
              scaleBandwidth_(scaleBandwidth),
              threshold_(threshold),
              maxIteration_(maxIteration){};
    ~LocalMaximaFinder(){};

    LocalMaxima findLocalMaxima(const VotingSpace& votingSpace, std::size_t voteStartT,
                                std::size_t voteEndT) const;

   private:
    std::vector<Point> getGridPoints(std::size_t startT, std::size_t endT, std::size_t startY,
                                     std::size_t endY, std::size_t startX, std::size_t endX,
                                     std::size_t startSIndex, std::size_t endSIndex) const;
    LocalMaxima findLocalMaxima(const VotingSpace& votingSpace,
                                const std::vector<Point>& gridPoints) const;
};
}
}

#endif