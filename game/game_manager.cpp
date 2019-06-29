#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <set>
#include <cmath>
#include <cassert>
using namespace std;

#define DEBUG
#define DEBUG_BOARD

#ifdef DEBUG
#define DEBUG_PRINTF(fmt, ...)  printf(fmt, __VA_ARGS__);                   
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

#define RANGE_ASSERT(x, y) assert(x >= 0 && x < BOARD_WIDTH && y >= 0 && y < BOARD_HEIGHT)

const static int BOARD_WIDTH = 6;
const static int BOARD_HEIGHT = 12;
const static int NUM_ROTATION = 4;

const static int colorA_dx[] = {0, 0, 0, 0};
const static int colorA_dy[] = {0, 0, 0, 1};
const static int colorB_dx[] = {1, 0, -1, 0};
const static int colorB_dy[] = {0, 1, 0, 0};

const static int dx[] = {0, 1, 0, -1};
const static int dy[] = {1, 0, -1, 0};

enum BLOCK {
  SKULL = '0',
  BLUE = '1',
  GREEN = '2',
  PINK = '3',
  RED = '4',
  YELLOW = '5',
  EMPTY = '.',
};

typedef char Color;

class Board {
private:
  Color cells_[BOARD_HEIGHT][BOARD_WIDTH]; // 左下が(0, 0)
  int empty_height_[BOARD_WIDTH]; // 各列の'0'の高さ(Blockを落としたときに落ちる高さ)
public:
  Color at(int x, int y) const {
    RANGE_ASSERT(x, y);
    return cells_[y][x];
  }
  void set(int x, int y, Color c) {
    RANGE_ASSERT(x, y);
    cells_[y][x] = c;
  }
  /**
   * colorA, colorBのBlockを落とす
   * 高さが足りず置けない場合は，falseを返す．
   * rotateとxの組が間違っている場合も，falseを返す
   */
  bool putBlock(int x, Color colorA, Color colorB, int rotate) {
    assert(x >= 0 && x < BOARD_WIDTH);
    assert(rotate >= 0 && rotate < NUM_ROTATION);
    if (x == 0 && rotate == 2) return false;
    if (x == BOARD_WIDTH - 1 && rotate == 0) return false;

    int y = empty_height_[x];
    int colorA_x = x + colorA_dx[rotate], colorA_y = y + colorA_dy[rotate];
    int colorB_x = x + colorB_dx[rotate], colorB_y = y + colorB_dy[rotate];

    if (colorA_y >= BOARD_HEIGHT || colorB_y >= BOARD_HEIGHT) return false;

    cells_[colorA_y][colorA_x] = colorA;
    cells_[colorB_y][colorB_x] = colorB;

    empty_height_[colorA_x]++;
    empty_height_[colorB_x]++;
    return true;
  }
  /**
   * 1ラインSKULLを落とす
   * もう置けない場合, falseを返す
   */
  bool putSkullLine() {
    for (int x = 0; x < BOARD_WIDTH; x++) {
      if (empty_height_[x] == BOARD_HEIGHT) return false;
      cells_[empty_height_[x]][x] = BLOCK::SKULL;
      empty_height_[x]++;
    }
    return true;
  }
  /**
   * コンボ後などEMTPYがブロックの隙間に入っているときにそれを詰める
   */
  void pack() {
    for(int x = 0; x < BOARD_WIDTH; x++) {
      int empty_y = -1;
      for(int y = 0; y < empty_height_[x]; y++) {
        if (empty_y == -1) {
          if( cells_[y][x] == BLOCK::EMPTY ) empty_y = y;
        }
        else {
          if (cells_[y][x] != BLOCK::EMPTY ) {
            cells_[empty_y][x] = cells_[y][x];
            cells_[y][x] = BLOCK::EMPTY;
            empty_y++;
          }
        }
      }
      if (empty_y != -1) empty_height_[x] = empty_y;
    }
  }
  bool isOut(int x, int y) const {
    return (x < 0 || x >= BOARD_WIDTH || y < 0 || y >= BOARD_HEIGHT);
  }
  void debugPrint() {
#ifdef DEBUG_BOARD
    for (int y = BOARD_HEIGHT - 1; y >= 0; y--) {
      for (int x = 0; x < BOARD_WIDTH; x++) {
        cout << at(x, y) << endl;
      }
    }
#endif
  }
  Board() {
    memset(cells_, 0, sizeof(cells_));
    memset(empty_height_, 0, sizeof(empty_height_));
  }
};

class ComboUtility {
private:
  int B; // The number of deleted Blocks
  int CP; // Chain Power
  int CB; // Color Bonus
  int GB; // Group Bonus
  int deleted_colors[5]; // EMPTYを抜いて5種（SKULLは使わない）

  /**
   * 4つ以上繋がっている部分をBFSで探す
   */
  void bfs(const Board* board, int sx, int sy, int searched[BOARD_HEIGHT][BOARD_WIDTH],
           vector<int>& deleted_x, vector<int>& deleted_y) {
    queue<int> qy, qx;
    qx.push(sx);
    qy.push(sy);
    searched[sy][sx] = 1;
    while(!qy.empty()) {
      int x = qx.front(); qx.pop();
      int y = qy.front(); qy.pop();
      deleted_y.push_back(y);
      deleted_x.push_back(x);
      for(int k = 0; k < 4; k++) {
        int ny = y + dy[k], nx = x + dx[k];
        if(board->isOut(x, y))continue;
        if(board->at(nx, ny) != board->at(x, y) || searched[ny][nx])continue;
        qx.push(nx);
        qy.push(ny);
        searched[ny][nx] = 1;
      }
    }
  }

  /**
   * 1combo以上したか
   */
  bool deleteConnectedColorBlocks(Board* board) {
    bool isCombo = false;
    int searched[BOARD_HEIGHT][BOARD_WIDTH];
    memset(searched, 0, sizeof(searched));
    for (int y = 0; y < BOARD_HEIGHT; y++) for (int x = 0; x < BOARD_HEIGHT; x++) {
      if(board->at(x, y) != BLOCK::SKULL && board->at(x, y) != BLOCK::EMPTY
          && searched[y][x] == 0) {
        vector<int> deleted_x, deleted_y;
        bfs(board, x, y, searched, deleted_x, deleted_y);
        int deleted_count = (int)deleted_x.size();

        // 4連結以上であれば消す
        if(deleted_count >= 4) {
          isCombo = true;
          Color deleted_color = board->at(deleted_x[0], deleted_y[0]);
          assert(deleted_color >= '1' && deleted_color <= '5');
          
          B += deleted_count; // 消した数を純粋カウント
          GB += min(deleted_count - 4, 8); // 一度に消した数が多いと良い（２ダブとか）
          deleted_colors[deleted_color - '1'] = 1; // 消した色をカウント

          // 連結マスと連結マスと隣接しているSKULLを削除
          for(int i = 0; i < deleted_count; i++) {
            for(int d = 0; d < 4; d++) {
              int ny = deleted_y[i] + dy[d], nx = deleted_x[i] + dx[d];
              if (board->isOut(nx, ny)) continue;
              if (board->at(nx, ny) == BLOCK::SKULL) board->set(nx, ny, BLOCK::EMPTY);
            }
            board->set(x, y, BLOCK::EMPTY);
          }
        }
      }
    }
    return isCombo;
  }

public:
  /**
   * 盤面に対してコンボシミュレーションを行い，スコアを返す
   */
  double getScoreByCombo(Board* board) {
    double score = 0.0;
    B = CP = CB = GB = 0;
    memset(deleted_colors, 0, sizeof(deleted_colors));
    // Group Bonus and B(# of deleted blocks)
    bool firstCombo = true;
    while(deleteConnectedColorBlocks(board)) {
      // 詰める
      board->pack();
      
      // Chainr Power (1combo: 0, 2combo: 8, 3combo~ 直前のCP*2)
      if(CP == 0 && (!firstCombo)) CP = 8;
      else CP *= 2;
      
      // Color Bonus
      int num_deleted_color_kind = 0;
      for(int i = 0; i < 5; i++) { // SKULLは除く
        if (deleted_colors[i]) num_deleted_color_kind++;
      }
      if( num_deleted_color_kind == 1 ) CB = 0;
      else CB = pow( 2, num_deleted_color_kind - 1);
      
      score += (10 * B) * max(1, min(999,(CP + CB + GB)));

      B = CB = GB = 0; // CPは消してはいけない
      memset(deleted_colors, 0, sizeof(deleted_colors));
      if( firstCombo ) firstCombo = false;
    }
    return score / 70.0;
  }
};

int main(void) {


  return 0;
}