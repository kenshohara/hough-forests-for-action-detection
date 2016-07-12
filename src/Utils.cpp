#include "Utils.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>

#include <boost/spirit/include/qi.hpp>
#include <boost/tokenizer.hpp>

#include <fstream>
#include <iostream>
#include <memory>

namespace nuisken {

std::vector<std::string> getFeatureNames(FeatureType type) {
    if (type == HF || type == HF_PCA) {
        return {"intensity", "x_derivatives", "y_derivatives", "t_derivatives", "x_flow", "y_flow"};
    } else if (type == HF_WITHOUT_FLOW) {
        return {"intensity", "x_derivatives", "y_derivatives", "t_derivatives"};
    } else {
        return {};
    }
}

namespace string {

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> results;
    std::size_t current = 0;
    std::size_t found;
    while ((found = str.find_first_of(delimiter, current)) != std::string::npos) {
        results.emplace_back(str, current, found - current);
        current = found + 1;
    }

    results.emplace_back(str, current, str.size() - current);
    return results;
}
}

namespace io {

std::string readFile(const std::string& filePath) {
    std::ifstream inputStream(filePath);
    if (!inputStream.is_open()) {
        return "";
    }

    inputStream.seekg(0, inputStream.end);
    std::size_t size = static_cast<std::size_t>(inputStream.tellg());
    if (size == 0) {
        return "";
    }
    inputStream.seekg(0, inputStream.beg);
    auto file = std::unique_ptr<char[]>(new char[size]);
    file[size - 1] = '\0';
    inputStream.read(file.get(), size);

    return std::string(file.get(), size);
}

void readSTIPFeatures(const std::string& filePath,
                      std::vector<std::vector<Eigen::MatrixXf>>& features,
                      std::vector<cv::Vec3i>& points) {
    std::string file = readFile(filePath);
    std::vector<std::string> lines = string::split(file, '\n');
    for (const auto& line : lines) {
        if (line.find('#') != std::string::npos) {
            continue;
        }

        std::vector<std::string> splitResults = string::split(line, ' ');

        if (splitResults.size() == 1) {
            break;
        } else if (splitResults.size() != 172) {
            break;
        }

        int xIndex = 5;
        int yIndex = 4;
        int tIndex = 6;
        cv::Vec3i centerPoint;
        boost::spirit::qi::parse(std::begin(splitResults.at(tIndex)),
                                 std::end(splitResults.at(tIndex)), boost::spirit::qi::int_,
                                 centerPoint(T));
        boost::spirit::qi::parse(std::begin(splitResults.at(yIndex)),
                                 std::end(splitResults.at(yIndex)), boost::spirit::qi::int_,
                                 centerPoint(Y));
        boost::spirit::qi::parse(std::begin(splitResults.at(xIndex)),
                                 std::end(splitResults.at(xIndex)), boost::spirit::qi::int_,
                                 centerPoint(X));

        int hogBeginIndex = 9;
        int numberOfHogDimension = 72;
        Eigen::MatrixXf hog(1, numberOfHogDimension);
        for (int i = 0; i < numberOfHogDimension; ++i) {
            boost::spirit::qi::parse(std::begin(splitResults.at(i + hogBeginIndex)),
                                     std::end(splitResults.at(i + hogBeginIndex)),
                                     boost::spirit::qi::double_, hog.coeffRef(0, i));
        }

        int hofBeginIndex = 81;
        int numberOfHofDimension = 90;
        Eigen::MatrixXf hof(1, numberOfHofDimension);
        for (int i = 0; i < numberOfHofDimension; ++i) {
            boost::spirit::qi::parse(std::begin(splitResults.at(i + hofBeginIndex)),
                                     std::end(splitResults.at(i + hofBeginIndex)),
                                     boost::spirit::qi::double_, hof.coeffRef(0, i));
        }

        std::vector<Eigen::MatrixXf> featureVectors;
        featureVectors.push_back(hog);
        featureVectors.push_back(hof);

        features.push_back(featureVectors);
        points.push_back(centerPoint);
    }
}
}

namespace actionvolume {

std::vector<std::pair<double, cv::Vec3f>> readActionPositions(const std::string& filePath) {
    std::vector<std::pair<double, cv::Vec3f>> actionPositions;

    std::ifstream inputStream(filePath);
    std::string line;
    while (std::getline(inputStream, line)) {
        if (line.find("[") != std::string::npos) {
            boost::char_separator<char> separator(",[]");
            boost::tokenizer<boost::char_separator<char>> tokenizer(line, separator);
            std::vector<std::string> tokens;
            std::copy(std::begin(tokenizer), std::end(tokenizer), std::back_inserter(tokens));

            cv::Vec3f actionPosition(std::stod(tokens.at(0)), std::stod(tokens.at(1)),
                                     std::stod(tokens.at(2)));
            actionPositions.emplace_back(std::stod(tokens.at(3)), actionPosition);
        }
    }

    return actionPositions;
}

void readActionPositions(const std::string& filePath,
                         std::vector<std::pair<double, cv::Vec3f>>& actionPositions,
                         std::vector<std::vector<cv::Vec3i>>& contributionPointsOfActionPosition,
                         double rate) {
    std::ifstream inputStream(filePath);
    std::string line;

    std::vector<std::pair<double, cv::Vec3i>> contributionPointsAndScores;
    while (std::getline(inputStream, line)) {
        if (line.find("LocalMaximum") != std::string::npos) {
            if (!contributionPointsAndScores.empty()) {
                std::sort(std::begin(contributionPointsAndScores),
                          std::end(contributionPointsAndScores),
                          [=](const std::pair<double, cv::Vec3i>& a,
                              const std::pair<double, cv::Vec3i>& b) { return a.first > b.first; });
                std::vector<cv::Vec3i> contributionPoints(contributionPointsAndScores.size() *
                                                          rate);
                for (auto i = 0; i < contributionPoints.size(); ++i) {
                    contributionPoints.at(i) = contributionPointsAndScores.at(i).second;
                }
                contributionPointsOfActionPosition.push_back(contributionPoints);
                contributionPointsAndScores.clear();
            }

            boost::char_separator<char> separator(",");
            boost::tokenizer<boost::char_separator<char>> tokenizer(line, separator);
            std::vector<std::string> tokens;
            std::copy(std::begin(tokenizer), std::end(tokenizer), std::back_inserter(tokens));

            cv::Vec3f actionPosition(std::stod(tokens.at(1)), std::stod(tokens.at(2)),
                                     std::stod(tokens.at(3)));
            actionPositions.emplace_back(std::stod(tokens.at(4)), actionPosition);
        } else if (!line.empty()) {
            boost::char_separator<char> separator(",");
            boost::tokenizer<boost::char_separator<char>> tokenizer(line, separator);
            std::vector<std::string> tokens;
            std::copy(std::begin(tokenizer), std::end(tokenizer), std::back_inserter(tokens));

            auto frame = std::stoi(tokens.at(0));
            auto contributionScore = std::stod(tokens.at(1));
            auto x = std::stoi(tokens.at(2));
            auto y = std::stoi(tokens.at(3));

            cv::Vec3i point(frame, y, x);
            contributionPointsAndScores.emplace_back(contributionScore, point);
        }
    }

    if (!contributionPointsAndScores.empty()) {
        std::sort(std::begin(contributionPointsAndScores), std::end(contributionPointsAndScores),
                  [=](const std::pair<double, cv::Vec3i>& a,
                      const std::pair<double, cv::Vec3i>& b) { return a.first > b.first; });
        std::vector<cv::Vec3i> contributionPoints(contributionPointsAndScores.size() * rate);
        for (auto i = 0; i < contributionPoints.size(); ++i) {
            contributionPoints.at(i) = contributionPointsAndScores.at(i).second;
        }
        contributionPointsOfActionPosition.push_back(contributionPoints);
        contributionPointsAndScores.clear();
    }
}
}
}