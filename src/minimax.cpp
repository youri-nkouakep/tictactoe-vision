#include "ttt/minimax.hpp"

#include <limits>
#include <unordered_map>

namespace ttt {

namespace {
struct Search {
    std::unordered_map<int, int> table;
    long nodes = 0;

    static int key(const Board& b) {
        int k = 0;
        for (Cell c : b) k = k * 3 + (c == Cell::Empty ? 0 : c == Cell::X ? 1 : 2);
        return k;
    }

    int negamax(Board& b, Cell me, int depth, int alpha, int beta) {
        ++nodes;
        const Status s = status(b);
        if (s == Status::Draw) return 0;
        if (s != Status::InProgress) return -(10 - depth);  // the previous player just won
        const int k = key(b);
        if (auto it = table.find(k); it != table.end()) return it->second;
        const Cell other = me == Cell::X ? Cell::O : Cell::X;
        const int alpha0 = alpha;
        int best = std::numeric_limits<int>::min();
        for (int i = 0; i < 9 && alpha < beta; ++i) {
            if (b[i] != Cell::Empty) continue;
            b[i] = me;
            const int v = -negamax(b, other, depth + 1, -beta, -alpha);
            b[i] = Cell::Empty;
            best = std::max(best, v);
            alpha = std::max(alpha, v);
        }
        if (best > alpha0 && best < beta) table[k] = best;  // exact value only, not a bound
        return best;
    }
};
}  // namespace

Move best_move(const Board& board) {
    Move result;
    const auto side = to_move(board);
    if (!side) return result;
    Board b = board;
    Search search;
    const Cell other = *side == Cell::X ? Cell::O : Cell::X;
    int depth = 0;
    for (Cell c : b) depth += c != Cell::Empty;
    result.score = std::numeric_limits<int>::min();
    for (int i = 0; i < 9; ++i) {
        if (b[i] != Cell::Empty) continue;
        b[i] = *side;
        const int v = -search.negamax(b, other, depth + 1, -1000, 1000);
        b[i] = Cell::Empty;
        if (v > result.score) result = {i, v, 0};
    }
    result.nodes = search.nodes;
    return result;
}

}  // namespace ttt
