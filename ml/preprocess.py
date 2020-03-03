import os.path
import sys
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt

from game_const import *

def get_aciont_tbl():
  action_id = 0
  action_tbl = {}
  for x in range(BOARD_WIDTH):
    for rotate in range(NUM_ROTATE):
      if x == 0 and rotate == 2: continue
      if x == BOARD_WIDTH - 1 and rotate == 0: continue
      action_tbl[(x, rotate)] = action_id
      action_id += 1
  return action_tbl

# 各ターンの情報
# 入力：盤面(codingameの入力), 出力：アクション（最終ターンはない）
class TurnInfo():
  action_tbl = get_aciont_tbl()

  def __init__(self, turn, nexts, boards, scores, ai_output):
    self.turn = turn
    self.nexts = nexts          # 共通のネクスト
    self.boards = boards        # [0]が自盤面， [1]が敵盤面
    self.scores = scores        # [0]が自スコア， [1]が敵スコア
    self.ai_output = ai_output  # 自分の出力
  
  def get_board_supervised(self):
    board_np = np.zeros((NUM_COLORS + 1, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    board = self.boards[0] # 自分の盤面
    for r, row in enumerate(board):
      for c, cell in enumerate(row):
        if cell == '.':
          continue
        else:
          assert(0 <= int(cell) and int(cell) <= NUM_COLORS)
          board_np[int(cell)][r][c] = 1
    return board_np
  
  def get_nexts_supervised(self):
    nexts_np = np.zeros((NUM_COLORS, NUM_NEXTS, NEXT_SIZE), dtype=np.float32)
    for r, row in enumerate(self.nexts):
      for c, cell in enumerate(row):
        assert(1 <= int(cell) and int(cell) <= NUM_COLORS)
        nexts_np[int(cell) - 1][r][c] = 1
    return nexts_np

  def get_output_supervised(self):
    if self.ai_output is None:
      return None
    return TurnInfo.action_tbl[(int(self.ai_output[0]), int(self.ai_output[1]))]

  def to_supervised(self):
    return ((self.get_board_supervised(),\
             self.get_nexts_supervised()),\
             self.get_output_supervised())

class LogContent():
  # log_root_dir/%06d ディレクトリ内
  #   0.txt, 1.txt: それぞれのAIの標準入出力
  #   result.txt:   下記三行
  #                 - win: (0 | 1 | 2) (2 = DRAW)
  #                 - ../ai/submitted/a.out (0's実行ファイル名)
  #                 - ../ai/random/random.out (1's実行ファイル名)
  def __init__(self, log_path):
    self.winner = None # result.txtから読み取った勝者
    self.commands = [] # result.txtから読み取ったコマンド
    self.ai_logs_raw = []  # 0.txt, 1.txtを全部readlinesで読み込んだ文字列
    self.turn_infos = [] # ターンごとの情報に分けたもの
    self.read_log_contents_from_files(log_path) # 扱いやすい形に変形
  
  def read_log_contents_from_files(self, log_path):
    # result.txtの読み込み
    result_path = os.path.join(log_path, 'result.txt')
    with open(result_path) as result_txt:
      lines = result_txt.readlines()
      self.winner = lines[0].strip().split()[1]
      self.commands.append(lines[1].strip())
      self.commands.append(lines[2].strip())
    # AIの標準入出力の読み込み
    ai_files = ['0.txt', '1.txt']
    for ai_file in ai_files:
      ai_path = os.path.join(log_path, ai_file)
      with open(ai_path) as ai_txt:
        lines = ai_txt.readlines()
        self.ai_logs_raw.append(lines)
        self.turn_infos.append(self.parse_ai_log(lines))
  
  # my_board/score, op_board/score, nexts, outputsに分ける
  # 色ごとにチャンネルに分割するのは別の関数でやる
  def parse_ai_log(self, ai_log):
    turn_infos = []
    it = iter(ai_log) # ai_logには生のデータが各行に入っている
    while True:
      # 入力
      ## turn情報の読み取り
      turn = next(it).strip()
      ## nexts
      nexts = []
      for _ in range(NUM_NEXTS):
        nexts.append(next(it).strip().split())
      ## scores, boards
      scores = []
      boards = []
      for _ in range(NUM_PLAYERS):
        # score
        score = int(next(it).strip())
        scores.append(score)
        # 盤面
        board = []
        for _ in range(BOARD_HEIGHT):
          board.append(next(it).strip())
        boards.append(board)
      
      # [endturn]後は出力がないので，break
      if turn == "[endturn]":
        turn_infos.append(TurnInfo(turn, nexts, boards, scores, None))
        break
      else:
        # 出力
        output = next(it).strip().split()
        turn_infos.append(TurnInfo(turn, nexts, boards, scores, output))

    return turn_infos

# @param `log_root_dir` (log_root_dir / %06d / *.txt)
# @return Array of LogContent
def load_smash_the_code_log(log_root_dir):
  files = os.listdir(log_root_dir)
  dirs = [dir for dir in files if os.path.isdir(os.path.join(log_root_dir, dir))]
  dirs.sort()

  log_contents = []
  repatter = re.compile(r'\d{6}')
  for dir_id, dir in enumerate(dirs):
    if not repatter.match(dir): continue
    log_path = os.path.join(log_root_dir, dir)
    log_contents.append(LogContent(log_path))

  return log_contents

# Create dataset from log.txt (教師あり学習用（現状のネットワーク）の入力・正解のデータセットを作る)
def log_content_to_supervised_data(log_content):
  # None If Draw (勝者なしの場合)
  if not (log_content.winner == '0' or log_content.winner == '1'):
    return None
  # @return ((board_x , next_x), t)形式へ変換する
  data = []
  for turn_info in log_content.turn_infos[int(log_content.winner)]:
    data.append(turn_info.to_supervised())
  return data

# @param Array of LogContent
# @return [((board_x, next_x), (t))_0, ((board_x, next_x), (t))_1, ...] 
# board_x, next_x, t : np.array
def transform_logs_for_supervised_learning(log_contents):
  dataset_per_one_game = []
  for log_id, log_content in enumerate(log_contents):
    data = log_content_to_supervised_data(log_content)
    if data is not None: # None if Draw (引き分けのときNone)
      dataset_per_one_game.append(data)
  
  dataset_per_turn = []
  for game_data in dataset_per_one_game:
    for turn_data in game_data:
      if turn_data[1] is None: # Reject if output is None (outputがNoneは弾く)
        continue
      dataset_per_turn.append(turn_data)
  return dataset_per_turn

def concat_samples(train_batch, gpu_id):
  # Transform [(x1,t1), (x2, t2), ...] to [[x1,x2,x3,...], [t1,t2,t3,...]]
  # [(x1,t1), (x2, t2), ...]形式から，[[x1,x2,x3,...], [t1,t2,t3,...]]形式へ
  # @todo dataset.concat_examplesの使い方調べる
  #       x, t = chainer.dataset.concat_examples(train_batch, gpu_id)
  x0 = []
  x1 = []
  t = []
  for data in train_batch:
    x0.append(data[0][0]) # 盤面
    x1.append(data[0][1]) # Nexts (ネクスト)
    t.append(data[1])     # Action

  x0 = np.array(x0)
  x1 = np.array(x1)
  t = np.array(t)

  # # Use GPU
  # if gpu_id is not None:
  #   x0 = chainer.dataset.to_device(gpu_id, x0)
  #   x1 = chainer.dataset.to_device(gpu_id, x1)
  #   t = chainer.dataset.to_device(gpu_id, t)
  
  return x0, x1, t

def save_dataset_bin(dataset, save_dir):
  np.save(save_dir, dataset)

def main(log_dir, dataset_save_dir):
  # Load log files in in log_dir
  print('*** [START] load files in {} ***'.format(log_dir), file=sys.stderr)
  log_contents = load_smash_the_code_log(log_dir)
  print('*** [ END ] load files in {} ***'.format(log_dir), file=sys.stderr)
  print(file=sys.stderr)

  # Transform LogContent to Dataset for learning
  print('*** [START] transform raw log.txt to dataset ***', file=sys.stderr)
  dataset = transform_logs_for_supervised_learning(log_contents)
  print('*** len(dataset) = {} ***'.format(len(dataset)), file=sys.stderr)
  print('*** [ END ] transform raw log.txt to dataset ***', file=sys.stderr)
  print(file=sys.stderr)

  # Save the dataset by binary
  print('*** Save dataset by bianary ***', file=sys.stderr)
  save_dataset_bin(dataset, dataset_save_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--log_dir", help="input to ml algorithm (log_dir/%%06d/*.txt)", type=str, required=True)
  parser.add_argument("--dataset_save_path", help="save the dataset transformed from log.txt in dataset_save_path", type=str, required=True)
  args = parser.parse_args()

  main(args.log_dir, args.dataset_save_path)