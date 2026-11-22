#include "ttt/vision.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace ttt {

namespace {

// Corners of marker `id` (0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left) in rectified pixels.
std::array<cv::Point2f, 4> marker_model(int id) {
    const float s = kWarp / kAreaMm, m = kMarkerMm, a = kAreaMm;
    const cv::Point2f origins[4] = {{-m, -m}, {a, -m}, {a, a}, {-m, a}};
    const cv::Point2f o = origins[id];
    return {o * s, (o + cv::Point2f(m, 0)) * s, (o + cv::Point2f(m, m)) * s, (o + cv::Point2f(0, m)) * s};
}

cv::Ptr<cv::aruco::DetectorParameters> detector_params() {
    auto p = cv::aruco::DetectorParameters::create();
    p->minMarkerPerimeterRate = 0.01;
    p->adaptiveThreshWinSizeMin = 3;
    p->adaptiveThreshWinSizeMax = 53;
    p->adaptiveThreshWinSizeStep = 4;
    p->perspectiveRemovePixelPerCell = 8;
    p->polygonalApproxAccuracyRate = 0.05;
    return p;
}

}  // namespace

BoardReader::BoardReader(const std::string& onnx_path)
    : dictionary_(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50)),
      params_(detector_params()),
      net_(cv::dnn::readNetFromONNX(onnx_path)) {}

cv::Mat BoardReader::normalize_cell(const cv::Mat& cell) {
    cv::Mat f;
    cell.convertTo(f, CV_32F);
    std::vector<float> v(f.begin<float>(), f.end<float>());
    const size_t k = static_cast<size_t>(0.9 * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    const float paper = v[k];
    cv::Mat ink = (paper - f) / std::max(paper, 1.0f);
    cv::max(ink, 0.0, ink);
    cv::min(ink, 1.0, ink);
    return ink * 2.0;
}

std::optional<Reading> BoardReader::read(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else gray = image;

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    cv::aruco::detectMarkers(gray, dictionary_, marker_corners, ids, params_);
    std::map<int, std::vector<cv::Point2f>> found;
    for (size_t i = 0; i < ids.size(); ++i) found[ids[i]] = marker_corners[i];

    Reading r;
    std::vector<cv::Point2f> src, dst;
    for (const auto& [id, pts] : found) {
        if (id < 0 || id > 3) continue;
        const auto model = marker_model(id);
        src.insert(src.end(), pts.begin(), pts.end());
        dst.insert(dst.end(), model.begin(), model.end());
        ++r.markers;
    }
    if (r.markers < kMinMarkers) return std::nullopt;
    r.homography = cv::findHomography(src, dst, 0);
    if (r.homography.empty()) return std::nullopt;
    const std::vector<cv::Point2f> area = {{0, 0}, {kWarp, 0}, {kWarp, kWarp}, {0, kWarp}};
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(area, projected, r.homography.inv());
    std::copy(projected.begin(), projected.end(), r.corners.begin());
    cv::Mat warped;
    cv::warpPerspective(gray, warped, r.homography, {kWarp, kWarp}, cv::INTER_AREA);

    const int m = static_cast<int>(kInnerMargin * kCellPx);
    std::vector<cv::Mat> cells;
    for (int i = 0; i < 9; ++i) {
        const cv::Rect roi((i % 3) * kCellPx + m, (i / 3) * kCellPx + m, kCellPx - 2 * m, kCellPx - 2 * m);
        cv::Mat small;
        cv::resize(warped(roi), small, {kInput, kInput}, 0, 0, cv::INTER_AREA);
        cells.push_back(normalize_cell(small));
    }
    net_.setInput(cv::dnn::blobFromImages(cells));
    cv::Mat logits = net_.forward();  // 9 x 3

    for (int i = 0; i < 9; ++i) {
        const float* row = logits.ptr<float>(i);
        const int best = static_cast<int>(std::max_element(row, row + 3) - row);
        float denom = 0.f;
        for (int c = 0; c < 3; ++c) denom += std::exp(row[c] - row[best]);
        r.confidence[i] = 1.f / denom;
        r.board[i] = best == 0 ? Cell::Empty : best == 1 ? Cell::X : Cell::O;
    }
    return r;
}

void draw_overlay(cv::Mat& frame, const Reading& r, int ai_cell, const std::string& banner) {
    const cv::Mat inv = r.homography.inv();
    auto to_image = [&](std::vector<cv::Point2f> pts) {
        std::vector<cv::Point2f> out;
        cv::perspectiveTransform(pts, out, inv);
        std::vector<cv::Point> ip(out.begin(), out.end());
        return ip;
    };
    const cv::Scalar green(80, 170, 60), orange(40, 140, 240), white(255, 255, 255);
    std::vector<std::vector<cv::Point>> outline = {to_image({{0, 0}, {kWarp, 0}, {kWarp, kWarp}, {0, kWarp}})};
    cv::polylines(frame, outline, true, green, 2, cv::LINE_AA);

    for (int i = 0; i < 9; ++i) {
        const float x = (i % 3) * kCellPx, y = (i / 3) * kCellPx;
        const auto c = to_image({{x + kCellPx / 2.f, y + kCellPx / 2.f}})[0];
        if (r.board[i] != Cell::Empty) {
            const std::string t(1, static_cast<char>(r.board[i]));
            cv::putText(frame, t, c + cv::Point(-9, -18), cv::FONT_HERSHEY_SIMPLEX, 0.6, green, 2, cv::LINE_AA);
        }
        if (i == ai_cell) {
            std::vector<std::vector<cv::Point>> cell = {
                to_image({{x + 8, y + 8}, {x + kCellPx - 8, y + 8}, {x + kCellPx - 8, y + kCellPx - 8}, {x + 8, y + kCellPx - 8}})};
            cv::Mat layer = frame.clone();
            cv::fillPoly(layer, cell, orange);
            cv::addWeighted(layer, 0.35, frame, 0.65, 0, frame);
            cv::polylines(frame, cell, true, orange, 3, cv::LINE_AA);
            cv::putText(frame, "O", c + cv::Point(-12, 12), cv::FONT_HERSHEY_SIMPLEX, 1.1, orange, 3, cv::LINE_AA);
        }
    }
    if (!banner.empty()) {
        int base = 0;
        const auto size = cv::getTextSize(banner, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &base);
        cv::rectangle(frame, {10, 10}, {30 + size.width, 30 + size.height + base}, cv::Scalar(40, 40, 40), cv::FILLED);
        cv::putText(frame, banner, {20, 20 + size.height}, cv::FONT_HERSHEY_SIMPLEX, 0.8, white, 2, cv::LINE_AA);
    }
}

}  // namespace ttt
