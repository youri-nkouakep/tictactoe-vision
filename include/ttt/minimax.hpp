#pragma once

#include "ttt/game.hpp"

namespace ttt {

struct Move {
    int cell = -1;
    int score = 0;          // +10 - depth: win for the side to move, 0: draw, negative: loss
    long nodes = 0;         // positions visited
};

// Negamax with alpha-beta pruning and a transposition table. Prefers faster wins and slower losses.
Move best_move(const Board& b);

}  // namespace ttt
