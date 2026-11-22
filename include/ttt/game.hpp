#pragma once

#include <array>
#include <optional>
#include <string>

namespace ttt {

enum class Cell : char { Empty = '.', X = 'X', O = 'O' };

using Board = std::array<Cell, 9>;

enum class Status { InProgress, XWins, OWins, Draw };

Status status(const Board& b);
std::optional<Cell> to_move(const Board& b);
bool is_reachable(const Board& b);
std::string to_string(const Board& b);
Board from_string(const std::string& s);

}  // namespace ttt
