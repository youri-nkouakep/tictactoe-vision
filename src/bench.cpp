// Latency of the full pipeline on a folder of photos: marker detection, rectification, 9-cell CNN.
//   ttt_bench models/cellnet.onnx data/test/*.jpg
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "ttt/minimax.hpp"
#include "ttt/vision.hpp"

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: ttt_bench model.onnx image...\n"; return 1; }
    ttt::BoardReader reader(argv[1]);
    std::vector<double> read_ms, ai_ms;
    for (int i = 2; i < argc; ++i) {
        const cv::Mat img = cv::imread(argv[i]);
        if (img.empty()) continue;
        reader.read(img);  // warm-up
        auto t0 = std::chrono::steady_clock::now();
        const auto r = reader.read(img);
        read_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        if (r && ttt::to_move(r->board)) {
            t0 = std::chrono::steady_clock::now();
            ttt::best_move(r->board);
            ai_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    auto report = [](const char* name, std::vector<double> v) {
        if (v.empty()) return;
        std::sort(v.begin(), v.end());
        double mean = 0;
        for (double x : v) mean += x;
        std::cout << name << ": mean " << mean / v.size() << " ms, median " << v[v.size() / 2] << " ms, p95 "
                  << v[static_cast<size_t>(0.95 * (v.size() - 1))] << " ms (" << v.size() << " frames)\n";
    };
    report("board reading", read_ms);
    report("AI move", ai_ms);
    return 0;
}
