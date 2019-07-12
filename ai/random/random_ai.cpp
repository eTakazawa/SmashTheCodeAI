#include <iostream>
using namespace std;

const static int BOARD_WIDTH = 6;
const static int BOARD_HEIGHT = 12;
const static int NUM_ROTATION = 4;
const static int NUM_NEXTS = 8;
const static int NUM_PLAYERS = 2;

int main(void) {
  srand(time(NULL));
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
    int x = rand() % BOARD_WIDTH;
    int rotate = rand() % NUM_ROTATION;
    while ((x == 0 && rotate == 2) || (x == BOARD_WIDTH - 1 && rotate == 0)) {
      x = rand() % BOARD_WIDTH;
      rotate = rand() % NUM_ROTATION;
    }
    cout << x << " " << rotate << endl;
  }
}