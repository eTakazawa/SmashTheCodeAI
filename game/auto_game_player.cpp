#include "game_manager.hpp"
#include <iostream>

using namespace std;

void battle(vector<string>& commands) {
  // プロセス・pstream
  vector<bp::child> ai_processes(NUM_PLAYERS);
  vector<bp::ipstream> from_ai_pstreams(NUM_PLAYERS);
  vector<bp::opstream> to_ai_pstreams(NUM_PLAYERS);
  // 受け渡し用にアドレスで渡す
  vector<bp::ipstream*> p_froms(NUM_PLAYERS);
  vector<bp::opstream*> p_tos(NUM_PLAYERS);
  // 各プレイヤーの起動・設定
  for (int pid = 0; pid < NUM_PLAYERS; pid++) {
    // プロセスの起動
    ai_processes.emplace_back(
      commands[pid],
      bp::std_in < to_ai_pstreams[pid],
      bp::std_out > from_ai_pstreams[pid],
      bp::std_err > bp::null
    );
    p_froms[pid] = &from_ai_pstreams[pid];
    p_tos[pid] = &to_ai_pstreams[pid];
  }
  
  GameManager gm;
  // 初期状態をログへ出力
  // gm.outputState();

  // 対戦シミュレーション開始
  while(gm.getGameState() != GameManager::DOING) {
    gm.communicateAndSimulateAllPlayerTurn(p_froms, p_tos);
    gm.toNextTurn();
    // gm.outputState();
  }

  // プロセスを落とす
  for (int pid = 0; pid < NUM_PLAYERS; pid++) {
    ai_processes[pid].terminate();
  }
}

// auto_game.out "使用AIを起動するコマンド" "対戦回数" "ログ出力保存先"
int main(void) {
  // コマンドライン引数から受け取る
  vector<string> commands(NUM_PLAYERS);
  int num_battles;
  string log_path;

  // 対戦
  for (int battle_cnt = 0; battle_cnt < num_battles; battle_cnt++) {
    battle(commands);
  }

  return 0;
}