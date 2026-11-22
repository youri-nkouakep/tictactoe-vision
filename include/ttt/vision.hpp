#pragma once

#include <array>
#include <optional>
#include <string>

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "ttt/game.hpp"

namespace ttt {

constexpr int kWarp = 300;
constexpr int kCellPx = kWarp / 3;
constexpr int kInput = 32;
constexpr double kInnerMargin = 0.12;
constexpr float kAreaMm = 150.f;
constexpr float kMarkerMm = 22.f;
constexpr int kMinMarkers = 2;

struct Reading {
    Board board{};
    std::array<float, 9> confidence{};
    cv::Mat homography;  // image -> rectified board
    std::array<cv::Point2f, 4> corners;  // playing area in the image
    int markers = 0;
};

// Rectifies the playing area from every visible ArUco marker (at least two) and classifies the 9 cells.
class BoardReader {
public:
    explicit BoardReader(const std::string& onnx_path);
    std::optional<Reading> read(const cv::Mat& image);
    static cv::Mat normalize_cell(const cv::Mat& cell);

private:
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    cv::Ptr<cv::aruco::DetectorParameters> params_;
    cv::dnn::Net net_;
};

// Draws the detected board, its state, and where the AI wants to play.
void draw_overlay(cv::Mat& frame, const Reading& r, int ai_cell, const std::string& banner);

}  // namespace ttt
