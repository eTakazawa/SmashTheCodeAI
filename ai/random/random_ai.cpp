#include <iostream>
#include <random>
#include "../../game/const.hpp"

using namespace std;

int mt_rand() {
  static random_device rd;
  static mt19937 mt(rd());
  return mt();
}

int main(void) {
  for (;;) {
    string tmp;
    // 8先まで
    for (int i = 0; i < NUM_NEXTS; i++) {
      cin >> tmp >> tmp;
    }
    // 各プレイヤーの盤面情報（最初が自分）
    for (int player_id = 0; player_id < NUM_PLAYERS; player_id++) {
      cin >> tmp;
      for (int y = BOARD_HEIGHT - 1; y >= 0; y--) {
        string tmp;
        cin >> tmp;
      }
    }
    int x = mt_rand() % BOARD_WIDTH;
    int rotate = mt_rand() % NUM_ROTATION;
    while ((x == 0 && rotate == 2) || (x == BOARD_WIDTH - 1 && rotate == 0)) {
      x = mt_rand() % BOARD_WIDTH;
      rotate = mt_rand() % NUM_ROTATION;
    }
    cout << x << " " << rotate << endl;
  }
}