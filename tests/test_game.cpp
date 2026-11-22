#include <doctest/doctest.h>

#include <functional>

#include "ttt/game.hpp"
#include "ttt/minimax.hpp"

using namespace ttt;

TEST_CASE("status and turn") {
    CHECK(status(from_string("XXX.OO...")) == Status::XWins);
    CHECK(status(from_string("XOXXOOOXX")) == Status::Draw);
    CHECK(to_move(from_string(".........")) == Cell::X);
    CHECK(to_move(from_string("X........")) == Cell::O);
    CHECK_FALSE(to_move(from_string("OOOXX.X..")).has_value());
}

TEST_CASE("reachability") {
    CHECK(is_reachable(from_string("XO.......")));
    CHECK_FALSE(is_reachable(from_string("XX.......")));
    CHECK_FALSE(is_reachable(from_string("XXXOOO...")));
}

TEST_CASE("AI takes a win and blocks a threat") {
    CHECK(best_move(from_string("XX.OO.X..")).cell == 5);  // O completes its row
    CHECK(best_move(from_string("XX..O....")).cell == 2);  // O blocks X
}

TEST_CASE("perfect play: the AI never loses, whatever the human plays") {
    long games = 0, losses = 0;
    std::function<void(Board)> play = [&](Board b) {
        if (status(b) != Status::InProgress) {
            ++games;
            losses += status(b) == Status::XWins;
            return;
        }
        if (to_move(b) == Cell::O) {
            b[best_move(b).cell] = Cell::O;
            play(b);
            return;
        }
        for (int i = 0; i < 9; ++i) {
            if (b[i] != Cell::Empty) continue;
            Board next = b;
            next[i] = Cell::X;
            play(next);
        }
    };
    play(from_string("........."));
    MESSAGE("games explored with the human playing every possible move: " << games);
    CHECK(games > 0);
    CHECK(losses == 0);
}
