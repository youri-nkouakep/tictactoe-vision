#include "ttt/tracker.hpp"

#include <algorithm>

#include "ttt/minimax.hpp"

namespace ttt {

GameTracker::GameTracker(int stable_frames, float min_confidence)
    : stable_frames_(stable_frames), min_confidence_(min_confidence) {
    board_.fill(Cell::Empty);
    candidate_.fill(Cell::Empty);
}

void GameTracker::update(const Reading& r) {
    const bool confident = std::all_of(r.confidence.begin(), r.confidence.end(),
                                       [&](float c) { return c >= min_confidence_; });
    if (!confident) {
        streak_ = 0;
        return;
    }
    streak_ = r.board == candidate_ ? streak_ + 1 : 1;
    candidate_ = r.board;
    if (streak_ == stable_frames_ && candidate_ != board_) accept(candidate_);
}

void GameTracker::accept(const Board& b) {
    int added = 0, removed = 0;
    for (int i = 0; i < 9; ++i) {
        if (board_[i] == Cell::Empty && b[i] != Cell::Empty) ++added;
        else if (board_[i] != b[i]) ++removed;
    }
    const bool fresh_start = std::count(b.begin(), b.end(), Cell::Empty) == 9;
    if (!fresh_start && (removed > 0 || added != 1 || !is_reachable(b))) {
        warning_ = "Unexpected change: one new mark at a time, please";
        return;
    }
    if (!fresh_start && to_move(board_).has_value() && b != board_) {
        for (int i = 0; i < 9; ++i) {
            if (board_[i] == Cell::Empty && b[i] != Cell::Empty && b[i] != *to_move(board_)) {
                warning_ = std::string("It was ") + static_cast<char>(*to_move(board_)) + "'s turn";
                return;
            }
        }
    }
    warning_.clear();
    board_ = b;
    ai_cell_ = to_move(board_) == Cell::O ? best_move(board_).cell : -1;
}

std::string GameTracker::message() const {
    if (!warning_.empty()) return warning_;
    switch (status(board_)) {
        case Status::XWins: return "You win!";
        case Status::OWins: return "The AI wins";
        case Status::Draw: return "Draw";
        default: break;
    }
    return ai_cell_ >= 0 ? "AI move: draw an O in the highlighted cell" : "Your turn: draw an X";
}

}  // namespace ttt
