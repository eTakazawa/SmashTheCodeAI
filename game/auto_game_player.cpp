#include "game_manager.hpp"
#include <iostream>
#include <filesystem>
#include <boost/format.hpp>

using namespace std;
namespace fs = std::filesystem;

void battle(vector<string>& commands, string log_root_path) {
  // プロセス・pstream
  vector<bp::child> ai_processes(NUM_PLAYERS);
  vector<bp::ipstream> from_ai_pstreams(NUM_PLAYERS);
  vector<bp::opstream> to_ai_pstreams(NUM_PLAYERS);
  // 受け渡し用にアドレスで渡す（@todo 生のアドレスで渡すのは避けたい）
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
  // 初期状態をログへ出力（@todo いい感じにログ出力するクラスがあると良い）
  const string log_dir_path = log_root_path;
  fs::create_directories(log_dir_path);
  
  vector<ofstream> ofs(NUM_PLAYERS);
  ofstream result(log_dir_path + "result.txt");
  // ファイル名は各プレイヤーID
  for (int i = 0; i < NUM_PLAYERS; i++) {
    ofs[i] = move(ofstream(log_dir_path + to_string(i) + ".txt"));
  }

  // 対戦シミュレーション開始
  while(gm.getGameState() == GameManager::DOING) {
    // 現在の状態を出力
    for (int i = 0; i < NUM_PLAYERS; i++) {
      ofs[i] << "[turn" << gm.turn_ << "]" << endl; 
      gm.outputState(ofs[i], i);
    }

    gm.communicateAndSimulateAllPlayerTurn(p_froms, p_tos);
    // 直前にした行動を出力
    for (int i = 0; i < NUM_PLAYERS; i++) gm.outputLastAction(ofs[i], i);

    gm.toNextTurn(); // 注意：next_colors等が更新されてしまう
  }
  // 最終状態を出力
  for (int i = 0; i < NUM_PLAYERS; i++) {
    ofs[i] << "[endturn]" << endl; 
    gm.outputState(ofs[i], i);
  }
  
  // 結果と合わせてコマンドも出力
  result << "win: " << gm.getGameState() << endl;
  for (int i = 0; i < NUM_PLAYERS; i++) {
    result << commands[i] << endl;
  }

  // プロセスを落とす pipeをclose
  for (int pid = 0; pid < NUM_PLAYERS; pid++) {
    ofs[pid].close();
    ai_processes[pid].terminate();
    from_ai_pstreams[pid].pipe().close();
    to_ai_pstreams[pid].pipe().close();
  }
  result.close();
}

// auto_game.out "使用AIを起動するコマンド" "対戦回数" "ログ出力保存先"
int main(int argc, char* argv[]) {
  if (argc < 4) {
    cerr << "usage: auto_game.out log_root_path command0 command1 ";
    cerr << "[num_battle<=999999]" << endl << endl;
    cerr << "(e.g.) ./auto_game.out ../log/`date '+%Y_%m_%d-%H_%M_%S'`/ ../ai/submitted/a.out ../ai/random/a.out 1000" << endl;
    exit(1);
  }

  // コマンドライン引数から受け取る
  vector<string> commands(NUM_PLAYERS);
  string log_root_path = argv[1]; // "../log/test/";
  commands[0] = argv[2]; //"../ai/random/a.out";
  commands[1] = argv[3]; // "../ai/random/a.out";
  int num_battles = 100;
  if (argc >= 5) {
    num_battles = stoi(argv[4]);
  }
  
  // 対戦
  for (int battle_cnt = 0; battle_cnt < num_battles; battle_cnt++) {
    string battle_cnt_str = (boost::format("%06d") % battle_cnt).str();
    string battle_path = log_root_path + battle_cnt_str + "/";
    battle(commands, battle_path);
  }

  return 0;
}