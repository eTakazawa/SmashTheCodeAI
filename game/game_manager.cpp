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
#include <memory>
#include <random>

#include <boost/process.hpp>

namespace bp = boost::process;
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
const static int NUM_NEXTS = 8;
const static int NUM_COLORS = 5;
const static double NUISANCE_PER_SCORE = 70.0;

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
const static Color colors[NUM_COLORS] = {
  BLOCK::BLUE,
  BLOCK::GREEN,
  BLOCK::PINK,
  BLOCK::RED,
  BLOCK::YELLOW
};

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
  void fillCellsEmpty() {
    memset(empty_height_, 0, sizeof(empty_height_));
    memset(cells_, BLOCK::EMPTY, sizeof(cells_));
  }
  void resetEmptyHeight() {
    for (int x = 0; x < BOARD_WIDTH; x++) {
      for (int y = BOARD_HEIGHT - 1; y >= 0; y--) {
        if (cells_[y][x] != BLOCK::EMPTY) {
          empty_height_[x] = y + 1;
          break;
        }
      }
    }
  }
  /**
   * colorA, colorBのBlockを落とす
   * 高さが足りず置けない場合は，falseを返す．
   * rotateとxの組が間違っている場合も，falseを返す
   * 
   * rotate 0: 右には置けない
   * rotate 2: 左には置けない
   */
  bool putBlock(int x, Color colorA, Color colorB, int rotate) {
    assert(x >= 0 && x < BOARD_WIDTH);
    assert(rotate >= 0 && rotate < NUM_ROTATION);
    if (x == 0 && rotate == 2) return false;
    if (x == BOARD_WIDTH - 1 && rotate == 0) return false;

    int colorA_x = x + colorA_dx[rotate];
    int colorB_x = x + colorB_dx[rotate];
    assert(colorA_x < BOARD_WIDTH && colorA_x >= 0);
    assert(colorB_x < BOARD_WIDTH && colorB_x >= 0);

    int A_y = empty_height_[colorA_x];
    int B_y = empty_height_[colorB_x];
    int colorA_y = A_y + colorA_dy[rotate];
    int colorB_y = B_y + colorB_dy[rotate];

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
        cout << at(x, y);
      }
      cout << endl;
    }
#endif
  }
  Board() {
    fillCellsEmpty();
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
      deleted_x.push_back(x);
      deleted_y.push_back(y);
      for(int d = 0; d < 4; d++) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        if(board->isOut(nx, ny)) continue;
        if(board->at(nx, ny) != board->at(x, y) || searched[ny][nx]) continue;
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
    for (int y = 0; y < BOARD_HEIGHT; y++) for (int x = 0; x < BOARD_WIDTH; x++) {
      if(board->at(x, y) != BLOCK::SKULL && board->at(x, y) != BLOCK::EMPTY
          && searched[y][x] == 0) {
        vector<int> deleted_x, deleted_y;
        bfs(board, x, y, searched, deleted_x, deleted_y);
        int deleted_count = (int)deleted_x.size();

        // 4連結以上であれば消す
        if(deleted_count >= 4) {
          // DEBUG_PRINTF("deleted_count: %d at (%2d, %2d)\n", deleted_count, x, y);
          
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
            board->set(deleted_x[i], deleted_y[i], BLOCK::EMPTY);
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
  int getNuisancePointsAfterComboSim(Board* board) {
    int score = 0;
    int combo_count = 0;
    B = CP = CB = GB = 0;
    memset(deleted_colors, 0, sizeof(deleted_colors));
    // Group Bonus and B(# of deleted blocks)
    bool firstCombo = true;
    // DEBUG_PRINTF("combo count: %d\n", combo_count++);
    while(deleteConnectedColorBlocks(board)) {
      // DEBUG_PRINTF("combo count: %d\n", combo_count++);
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
    return score;
  }
};

class GameManager {
public:
  const static int NUM_PLAYERS = 2;
  unique_ptr<Board> boards_[NUM_PLAYERS];
  unique_ptr<ComboUtility> combo_util_;
  Color nexts_colors_[NUM_NEXTS][2];

  vector<int> scores_;
  vector<int> curr_scores_;
  vector<int> deads_;
  vector<double> nuisance_points_;
  int turn_;

  typedef pair<int,int> Action;
  
  GameManager(): scores_(NUM_PLAYERS), curr_scores_(NUM_PLAYERS), deads_(NUM_PLAYERS), nuisance_points_(NUM_PLAYERS) {
    for (int i = 0; i < NUM_PLAYERS; i++) {
      boards_[i] = make_unique<Board>();
    }
    combo_util_ = make_unique<ComboUtility>();
    for (int i = 0; i < NUM_NEXTS; i++) {
      nexts_colors_[i][0] = getRandomColor();
      nexts_colors_[i][1] = getRandomColor();
    }
    turn_ = 0;
  }

  void setStatusFromStdin(int ai_player_id) {
    // 8先まで
    for (int i = 0; i < NUM_NEXTS; i++) {
      cin >> nexts_colors_[i][0] >> nexts_colors_[i][1];
    }
    // 各プレイヤーの盤面情報（最初が自分）
    for (int player_id_offset = 0; player_id_offset < NUM_PLAYERS; player_id_offset++) {
      int player_id = (ai_player_id + player_id_offset) % NUM_PLAYERS;
      cin >> scores_[player_id];
      for (int y = BOARD_HEIGHT - 1; y >= 0; y--) {
        string tmp;
        cin >> tmp;
        for (int x = 0; x < BOARD_WIDTH; x++) {
          boards_[player_id]->set(x, y, tmp[x]);
        }
      }
      boards_[player_id]->resetEmptyHeight();
    }
  }

  /**
   * player_idの1ターン分のシミュレーション
   * (死んでいないか，スコア)を返す
   */
  pair<bool, int> simulatePlayerTurn(int player_id, int x, int rotate, int num_skull_line) {
    assert(player_id >= 0 && player_id < NUM_PLAYERS);

    // 現在のplayerの変数をとっておく
    Board *board = boards_[player_id].get();
    Color *nexts = nexts_colors_[0];

    // skullを落とす，nuisance points更新
    for (int s = 0; s < num_skull_line; s++) {
      bool can_skull_drop = board->putSkullLine();
      if (!can_skull_drop) return make_pair(false, 0);
    }

    // ブロックを落とす
    bool is_put_block = board->putBlock(x, nexts[0], nexts[1], rotate);
    if (!is_put_block) return make_pair(false, 0);
    
    // score取得して，メンバ変数を更新
    int score = combo_util_->getNuisancePointsAfterComboSim(board);
    return make_pair(true, score);
  }
  /**
   * 全員分の1ターンのシミュレーションをする
   * @return 生きているプレイヤー数 @todo スコアの更新も分離した方が良い
   */
  int simulateAllPlayerTurn(vector<Action>& curr_actions) {
    assert(!isAllDead());
    // シミュレーションして次の盤面へ
    for (int player_id = 0; player_id < NUM_PLAYERS; player_id++) {
      int op_player_id = (player_id + 1) % NUM_PLAYERS;
      // 敵から送られてくるskull数計算
      int num_skull_lines = (int)(nuisance_points_[op_player_id] / 6);
      nuisance_points_[op_player_id] -= 6 * num_skull_lines;

      // シミュレーション
      int x = curr_actions[player_id].first;
      int rotate = curr_actions[player_id].second;
      
      bool is_alive;
      int score;
      tie(is_alive, score) = simulatePlayerTurn(player_id, x, rotate, num_skull_lines);
      deads_[player_id] = !is_alive; // @todo deads_の更新はここでも平気？
      curr_scores_[player_id] = score;
    }

    // スコアの更新（ここでやらないとコンボ後のスコアが入ってしまう）
    for (int player_id = 0; player_id < NUM_PLAYERS; player_id++) {
      scores_[player_id] += curr_scores_[player_id];
      nuisance_points_[player_id] += curr_scores_[player_id] / NUISANCE_PER_SCORE;
    }
    return countAlives();
  }
  template<class T1, class T2>
  vector<Action> communicateToAllAITurn(vector<T1*> &from_ais, vector<T2*> &to_ais) {
    vector<Action> curr_actions(NUM_PLAYERS);
    for (int player_id = 0; player_id < NUM_PLAYERS; player_id++) {
      if (deads_[player_id]) continue; // 死んでたら飛ばす
      // player_idのAIへ盤面情報を出力・返答待ち
      curr_actions[player_id] = communicateToAIOneTurn(from_ais[player_id], to_ais[player_id], player_id);
    }
    return curr_actions;
  }
  template<class T1, class T2>
  Action communicateToAIOneTurn(T1* from_ai_p, T2* to_ai_p, int ai_player_id) {
    DEBUG_PRINTF("communicate to player_id: %d\n", ai_player_id);

    T1 &from_ai = *from_ai_p;
    T2 &to_ai = *to_ai_p;

    // 8先まで
    for (int i = 0; i < NUM_NEXTS; i++) {
      to_ai << nexts_colors_[i][0] << " " << nexts_colors_[i][1] << endl;
    }
    // 各プレイヤーの盤面情報（最初が自分）
    for (int player_id_offset = 0; player_id_offset < NUM_PLAYERS; player_id_offset++) {
      int player_id = (ai_player_id + player_id_offset) % NUM_PLAYERS;
      to_ai << scores_[player_id] << endl;
      for (int y = BOARD_HEIGHT - 1; y >= 0; y--) {
        for (int x = 0; x < BOARD_WIDTH; x++) {
          to_ai << boards_[player_id]->at(x, y);
        }
        to_ai << endl;
      }
    }
    // AIからの入力待ち
    int from_ai_x, from_ai_rotate;
    // @todo ここで時間計測
    from_ai >> from_ai_x >> from_ai_rotate;
    return Action(from_ai_x, from_ai_rotate);
  }

  template<class T1, class T2>
  int communicateAndSimulateAllPlayerTurn(vector<T1*> &from_ais, vector<T2*> &to_ais) {
    vector<Action> curr_action = communicateToAllAITurn(from_ais, to_ais);
    return simulateAllPlayerTurn(curr_action);
  }
  void toNextTurn() {
    turn_++;
    for (int i = 0; i < NUM_NEXTS - 1; i++) {
      nexts_colors_[i][0] = nexts_colors_[i + 1][0];
      nexts_colors_[i][1] = nexts_colors_[i + 1][1];
    }
    nexts_colors_[NUM_NEXTS - 1][0] = getRandomColor();
    nexts_colors_[NUM_NEXTS - 1][1] = getRandomColor();

    fill(curr_scores_.begin(), curr_scores_.end(), 0);
  }

  bool isEndGame() {
    int count_alives = countAlives();
    if (count_alives <= 1) return true;
    return false;
  }

  bool isAllDead() {
    for (int i = 0; i < NUM_PLAYERS; i++) {
      if (!deads_[i]) return false;
    }
    return true;
  }
  int countAlives() {
    int cnt = 0;
    for (int i = 0; i < NUM_PLAYERS; i++) {
      if (!deads_[i]) cnt++;
    }
    return cnt;
  }

private:
  Color getRandomColor() {
    static random_device rd;
    static mt19937 mt(rd());
    static uniform_int_distribution<int> dice(0, NUM_COLORS - 1);
    return colors[dice(mt)];
  }
};

static void random_vs_submitted() {
  GameManager gm;
  bp::ipstream from_ai_1, from_ai_2;
  bp::opstream to_ai_1, to_ai_2;
  bp::child ai1("../ai/random/a.out", bp::std_in < to_ai_1, bp::std_out > from_ai_1);
  bp::child ai2("../ai/submitted/a.out", bp::std_in < to_ai_2, bp::std_out > from_ai_2, bp::std_err > bp::null);
  vector<bp::ipstream*> is;
  vector<bp::opstream*> os;
  is.push_back(&from_ai_1);is.push_back(&from_ai_2);
  os.push_back(&to_ai_1);os.push_back(&to_ai_2);

  for (int i = 0; i < 100; i++) {
    cerr << "phase: " << i << endl;
    gm.communicateAndSimulateAllPlayerTurn(is, os);
    gm.boards_[1]->debugPrint();
    gm.toNextTurn();

    if (gm.isEndGame()) break;
  }

  ai1.terminate();
  ai2.terminate();
}

static void test_codingame() {
  GameManager gm;
  gm.setStatusFromStdin(0);

  int x, rotate;
  cin >> x >> rotate;
  pair<bool, int> res = gm.simulatePlayerTurn(0, x, rotate, 0);
  cerr << res.first << " " << res.second << endl;
  gm.boards_[0]->debugPrint();
}

static void test_game_manager_npstream() {
  GameManager gm;
  bp::ipstream from_ai_1, from_ai_2;
  bp::opstream to_ai_1, to_ai_2;
  bp::child ai1("../ai/submitted/a.out", bp::std_in < to_ai_1, bp::std_out > from_ai_1);
  bp::child ai2("../ai/submitted/a.out", bp::std_in < to_ai_2, bp::std_out > from_ai_2, bp::std_err > bp::null);

  vector<bp::ipstream*> is;
  vector<bp::opstream*> os;
  is.push_back(&from_ai_1);is.push_back(&from_ai_2);
  os.push_back(&to_ai_1);os.push_back(&to_ai_2);

  for (int i = 0; i < 15; i++) {
    cerr << "phase: " << i << endl;
    gm.communicateAndSimulateAllPlayerTurn(is, os);
    gm.boards_[0]->debugPrint();
    gm.toNextTurn();
  }

  ai1.terminate();
  ai2.terminate();
}

static void test_game_manager_iostream() {
  GameManager gm;
  vector<istream*> is;
  vector<ostream*> os;
  is.push_back(&cin);is.push_back(&cin);
  os.push_back(&cout);os.push_back(&cout);

  for (int i = 0; i < 100; i++) {
    gm.communicateAndSimulateAllPlayerTurn(is, os);
    gm.boards_[0]->debugPrint();
  }
}

static void test_Board_put() {
  unique_ptr<Board> board(new Board());
  unique_ptr<ComboUtility> combo_util(new ComboUtility());

  cout << "test 1" << endl;
  cout << board->putSkullLine() << endl;
  board->debugPrint();
  cout << endl;

  cout << "test 2" << endl;
  cout << board->putBlock(0, BLOCK::RED, BLOCK::RED, 0) << endl;
  board->debugPrint();
  cout << endl;

  cout << "test 3" << endl;
  cout << board->putBlock(BOARD_WIDTH - 1, BLOCK::RED, BLOCK::RED, 0) << endl;
  board->debugPrint();
  cout << endl;


  cout << "test 4" << endl;
  cout << board->putBlock(BOARD_WIDTH - 1, BLOCK::RED, BLOCK::RED, 0) << endl;
  cout << board->putBlock(0, BLOCK::RED, BLOCK::RED, 2) << endl;
  cout << endl;

  cout << "test 5" << endl;
  board->fillCellsEmpty();
  board->debugPrint();
  cout << board->putBlock(0, BLOCK::RED, BLOCK::RED, 1) << endl;
  cout << board->putBlock(1, BLOCK::RED, BLOCK::RED, 1) << endl;
  cout << board->putBlock(1, BLOCK::RED, BLOCK::YELLOW, 1) << endl;
  cout << board->putBlock(2, BLOCK::YELLOW, BLOCK::YELLOW, 1) << endl;
  cout << board->putBlock(2, BLOCK::YELLOW, BLOCK::BLUE, 1) << endl;
  cout << board->putBlock(3, BLOCK::BLUE, BLOCK::BLUE, 1) << endl;
  cout << board->putBlock(3, BLOCK::BLUE, BLOCK::PINK, 1) << endl;
  cout << board->putBlock(4, BLOCK::PINK, BLOCK::PINK, 1) << endl;
  cout << board->putBlock(4, BLOCK::PINK, BLOCK::GREEN, 1) << endl;
  board->debugPrint();
  cout << combo_util->getNuisancePointsAfterComboSim(board.get()) << endl;
  board->debugPrint();
  cout << endl;
}

static void test(void) {
  // test_game_manager_npstream();
  // test_game_manager_iostream();
  // test_codingame();
  // random_vs_submitted();
  return;
}