#include <doctest/doctest.h>

#include "ttt/tracker.hpp"

using namespace ttt;

namespace {
Reading reading(const std::string& s, float confidence = 0.99f) {
    Reading r;
    r.board = from_string(s);
    r.confidence.fill(confidence);
    return r;
}
}  // namespace

TEST_CASE("a move is accepted only after several identical frames") {
    GameTracker t(3);
    t.update(reading("X........"));
    t.update(reading("X........"));
    CHECK(to_string(t.board()) == ".........");
    t.update(reading("X........"));
    CHECK(to_string(t.board()) == "X........");
    CHECK(t.ai_cell() >= 0);
}

TEST_CASE("low-confidence frames and illegal changes are ignored") {
    GameTracker t(2);
    t.update(reading("X........", 0.5f));
    t.update(reading("X........", 0.5f));
    CHECK(to_string(t.board()) == ".........");
    t.update(reading("XX......."));
    t.update(reading("XX......."));
    CHECK(to_string(t.board()) == ".........");
    CHECK(t.warning());
}
