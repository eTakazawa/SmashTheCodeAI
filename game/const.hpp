#pragma once

#ifdef DEBUG
#define DEBUG_PRINTF(fmt, ...)  printf(fmt, __VA_ARGS__);                   
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

#define RANGE_ASSERT(x, y) assert(x >= 0 && x < BOARD_WIDTH && y >= 0 && y < BOARD_HEIGHT)

typedef char Color;
class BLOCK {
public:
  enum {
    SKULL = '0',
    BLUE = '1',
    GREEN = '2',
    PINK = '3',
    RED = '4',
    YELLOW = '5',
    EMPTY = '.',
  };
  constexpr static int NUM_COLORS = 5;
  constexpr static Color COLORS [NUM_COLORS] = {
    BLOCK::BLUE,
    BLOCK::GREEN,
    BLOCK::PINK,
    BLOCK::RED,
    BLOCK::YELLOW
  };
};

constexpr static int BOARD_WIDTH = 6;
constexpr static int BOARD_HEIGHT = 12;
constexpr static int NUM_ROTATION = 4;
constexpr static int NUM_NEXTS = 8;
constexpr static int END_TURN = 200;

constexpr static double NUISANCE_PER_SCORE = 70.0;

constexpr static int colorA_dx[] = {0, 0, 0, 0};
constexpr static int colorA_dy[] = {0, 0, 0, 1};
constexpr static int colorB_dx[] = {1, 0, -1, 0};
constexpr static int colorB_dy[] = {0, 1, 0, 0};

constexpr static int dx[] = {0, 1, 0, -1};
constexpr static int dy[] = {1, 0, -1, 0};

constexpr static int NUM_PLAYERS = 2;
