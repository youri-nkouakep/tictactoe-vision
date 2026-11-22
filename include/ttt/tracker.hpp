#pragma once

#include <string>

#include "ttt/game.hpp"
#include "ttt/vision.hpp"

namespace ttt {

// Turns noisy per-frame readings into a game: a new board is accepted only after it has been
// read identically, with high confidence, on several consecutive frames, and only if it is a
// legal continuation of the current game (exactly one new mark).
class GameTracker {
public:
    explicit GameTracker(int stable_frames = 6, float min_confidence = 0.8f);

    void update(const Reading& r);
    const Board& board() const { return board_; }
    int ai_cell() const { return ai_cell_; }
    std::string message() const;
    bool warning() const { return !warning_.empty(); }

private:
    Board board_{};
    Board candidate_{};
    int streak_ = 0;
    int ai_cell_ = -1;
    std::string warning_;
    int stable_frames_;
    float min_confidence_;

    void accept(const Board& b);
};

}  // namespace ttt
