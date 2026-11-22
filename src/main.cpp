// TicTacToe Vision: play on paper against a perfect AI.
//   ttt_vision --camera 0             live webcam
//   ttt_vision --video game.mp4       recorded video (writes an annotated copy with --out)
//   ttt_vision --image photo.jpg      single photo: board state and the AI move
#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "ttt/minimax.hpp"
#include "ttt/tracker.hpp"
#include "ttt/vision.hpp"

namespace {
struct Args {
    std::string model = "models/cellnet.onnx";
    std::string image, video, out;
    int camera = -1;
    bool show = true;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (k == "--model") a.model = next();
        else if (k == "--image") a.image = next();
        else if (k == "--video") a.video = next();
        else if (k == "--camera") a.camera = std::stoi(next());
        else if (k == "--out") a.out = next();
        else if (k == "--no-window") a.show = false;
    }
    return a;
}

int run_image(ttt::BoardReader& reader, const Args& a) {
    cv::Mat img = cv::imread(a.image);
    if (img.empty()) { std::cerr << "cannot read " << a.image << "\n"; return 1; }
    const auto t0 = std::chrono::steady_clock::now();
    auto r = reader.read(img);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!r) { std::cerr << "board not found (the four markers must be visible)\n"; return 2; }
    const auto move = ttt::best_move(r->board);
    std::cout << "board: " << ttt::to_string(r->board) << "  (" << ms << " ms)\n";
    if (move.cell >= 0) std::cout << "best move for " << static_cast<char>(*ttt::to_move(r->board)) << ": cell " << move.cell << "\n";
    ttt::draw_overlay(img, *r, ttt::to_move(r->board) == ttt::Cell::O ? move.cell : -1, ttt::to_string(r->board));
    if (!a.out.empty()) cv::imwrite(a.out, img);
    if (a.show) { cv::imshow("TicTacToe Vision", img); cv::waitKey(0); }
    return 0;
}

int run_stream(ttt::BoardReader& reader, const Args& a) {
    cv::VideoCapture cap = a.camera >= 0 ? cv::VideoCapture(a.camera) : cv::VideoCapture(a.video);
    if (!cap.isOpened()) { std::cerr << "cannot open the video source\n"; return 1; }
    cv::VideoWriter writer;
    ttt::GameTracker tracker;
    cv::Mat frame;
    while (cap.read(frame)) {
        if (auto r = reader.read(frame)) {
            tracker.update(*r);
            ttt::Reading shown = *r;
            shown.board = tracker.board();
            ttt::draw_overlay(frame, shown, tracker.ai_cell(), tracker.message());
        } else {
            cv::putText(frame, "Show the whole board (4 markers)", {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(40, 140, 240), 2, cv::LINE_AA);
        }
        if (!a.out.empty()) {
            if (!writer.isOpened())
                writer.open(a.out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 20, frame.size());
            writer.write(frame);
        }
        if (a.show) {
            cv::imshow("TicTacToe Vision", frame);
            if (cv::waitKey(1) == 27) break;
        }
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    ttt::BoardReader reader(a.model);
    if (!a.image.empty()) return run_image(reader, a);
    if (!a.video.empty() || a.camera >= 0) return run_stream(reader, a);
    std::cerr << "usage: ttt_vision (--camera 0 | --video file | --image file) [--out file] [--no-window]\n";
    return 1;
}
