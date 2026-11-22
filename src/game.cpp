#include "ttt/game.hpp"

#include <algorithm>
#include <stdexcept>

namespace ttt {

namespace {
constexpr int kLines[8][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
                              {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};

bool wins(const Board& b, Cell p) {
    return std::any_of(std::begin(kLines), std::end(kLines), [&](const int* l) {
        return b[l[0]] == p && b[l[1]] == p && b[l[2]] == p;
    });
}

int count(const Board& b, Cell p) { return static_cast<int>(std::count(b.begin(), b.end(), p)); }
}  // namespace

Status status(const Board& b) {
    if (wins(b, Cell::X)) return Status::XWins;
    if (wins(b, Cell::O)) return Status::OWins;
    if (count(b, Cell::Empty) == 0) return Status::Draw;
    return Status::InProgress;
}

std::optional<Cell> to_move(const Board& b) {
    if (status(b) != Status::InProgress) return std::nullopt;
    return count(b, Cell::X) == count(b, Cell::O) ? Cell::X : Cell::O;
}

bool is_reachable(const Board& b) {
    const int x = count(b, Cell::X), o = count(b, Cell::O);
    if (x != o && x != o + 1) return false;
    const bool xw = wins(b, Cell::X), ow = wins(b, Cell::O);
    if (xw && ow) return false;
    if (xw && x != o + 1) return false;
    if (ow && x != o) return false;
    return true;
}

std::string to_string(const Board& b) {
    std::string s;
    for (Cell c : b) s += static_cast<char>(c);
    return s;
}

Board from_string(const std::string& s) {
    if (s.size() != 9) throw std::invalid_argument("board string must have 9 characters");
    Board b{};
    for (int i = 0; i < 9; ++i) {
        switch (s[i]) {
            case 'X': case 'x': b[i] = Cell::X; break;
            case 'O': case 'o': b[i] = Cell::O; break;
            case '.': case '-': case ' ': b[i] = Cell::Empty; break;
            default: throw std::invalid_argument("invalid board character");
        }
    }
    return b;
}

}  // namespace ttt
