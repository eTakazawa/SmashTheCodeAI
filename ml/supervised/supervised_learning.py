import os.path
import sys
import re

import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.links as L
import chainer.functions as F

from chainer.datasets import TupleDataset
from chainer.datasets import split_dataset_random
from chainer.iterators import SerialIterator
from chainer import optimizers
from chainer import serializers

NUM_PLAYERS = 2
BOARD_HEIGHT = 12
BOARD_WIDTH = 6
NUM_COLORS = 5
NUM_NEXTS = 8
NEXT_SIZE = 2
NUM_ROTATE = 4
NUM_ACTIONS = BOARD_WIDTH * NUM_ROTATE - 2

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

# @param `log_root_dir` を指定 (log_root_dir / %06d / *.txt)
# @return LogContentの配列
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

# 教師あり学習用（現状のネットワーク）の入力・正解のデータセットを作る
def log_content_to_supervised_data(log_content):
  # 勝者なしの場合, None
  if not (log_content.winner == '0' or log_content.winner == '1'):
    return None
  # @return ((board_x , next_x), t)形式へ変換する
  data = []
  for turn_info in log_content.turn_infos[int(log_content.winner)]:
    data.append(turn_info.to_supervised())
  return data

# @param LogContentの配列
# @return [((board_x, next_x), (t))_0, ((board_x, next_x), (t))_1, ...] 
# board_x, next_x, t : np.array
def transform_logs_for_supervised_learning(log_contents):
  dataset_per_one_game = []
  for log_id, log_content in enumerate(log_contents):
    data = log_content_to_supervised_data(log_content)
    if data is not None: # 引き分けのときNone
      dataset_per_one_game.append(data)
  
  dataset_per_turn = []
  for game_data in dataset_per_one_game:
    for turn_data in game_data:
      if turn_data[1] is None: # outputがNoneは弾く
        continue
      dataset_per_turn.append(turn_data)
  return dataset_per_turn

DROP_OUT_RATIO = 0.3 # 0.3~0.4くらいが良いらしい?
# Full pre-activation https://www.slideshare.net/masatakanishimori/res-net
class BaseBlock(chainer.Chain):
  def __init__(self, n_in, n_out, stride=1, use_dropout=False):
    super(BaseBlock, self).__init__()
    self.n_in = n_in
    self.n_out = n_out

    w = chainer.initializers.HeNormal()
    self.use_dropout = use_dropout
    
    # パラメータの持つ層の登録
    with self.init_scope():
      # L.BatchNormalization(size, decay, eps, dtype, ...)
      # L.Convolution2D(in_channels, out_channels, ksize, stride=1, pad=0, nobias=False, ...)
      self.bn1   = L.BatchNormalization(n_in)
      self.conv1 = L.Convolution2D(n_in, n_out, 3, stride, 1, True, w)
      self.bn2   = L.BatchNormalization(n_out)
      self.conv2 = L.Convolution2D(n_out, n_out, 3, 1, 1, True, w)
  
  def __call__(self, x):
    # 推論
    residual_x = x 
    if self.n_in != self.n_out: # チャンネル数が合わないときはpadding
      residual_x = F.pad_sequence(residual_x, self.n_out)

    x = F.leaky_relu(self.bn1(x)) # reluよりleaky_reluが良いらしい
    x = F.leaky_relu(self.bn2(self.conv1(x)))
    if self.use_dropout:
      x = F.dropout(x, DROP_OUT_RATIO)
    x = self.conv2(x)

    # print('{} {} {} '.format(x.shape, residual_x.shape, (x + residual_x).shape))
    return x + residual_x

class FullPreActivationBlock(chainer.Chain):
  def __init__(self, n_in, n_out, n_block, stride=1, use_dropout=False):
    super(FullPreActivationBlock, self).__init__()
    self.n_block = n_block
    self.use_dropout = use_dropout
    
    with self.init_scope():
      self.block0 = BaseBlock(n_in, n_out, stride, use_dropout)
      for block_id in range(1, self.n_block):
        bx = BaseBlock(n_out, n_out)
        setattr(self, 'block{}'.format(str(block_id)), bx)
  
  def __call__(self, x):
    x = self.block0(x)
    for block_id in range(1, self.n_block):
      block_attr = getattr(self, 'block{}'.format(str(block_id)))
      x = block_attr(x)
      # print('block{}, n_in:{} n_out:{}'.format(str(block_id), block_attr.n_in, block_attr.n_out))
    return x

class SmashTheCodeNet(chainer.Chain):
  def __init__(self, n_board_mid=256, n_nexts_mid=128, n_board_blocks=15, n_nexts_blocks=5):
    # n_mid: フィルタ（チャンネル）数 層を深くするより，フィルタ数を増やした方が良いらしい?
    
    super(SmashTheCodeNet, self).__init__()
    w = chainer.initializers.HeNormal()
    
    # 全結合層のノード数
    ## 盤面
    n_board_flatten_in = BOARD_HEIGHT * BOARD_WIDTH * n_board_mid
    n_board_fc0_out = 256
    n_board_fc1_out = n_board_fc0_out // 2
    ## ネクスト
    n_nexts_flatten_in = NUM_NEXTS * NEXT_SIZE * n_nexts_mid
    n_nexts_fc0_out = 128
    n_nexts_fc1_out = n_nexts_fc0_out // 2
    ## concated
    n_all_fc0_in = n_board_fc1_out + n_nexts_fc1_out
    n_all_fc0_out = 256
    n_all_fc1_out = 128

    with self.init_scope():
      # 盤面入力：(B,H,W,C)=(None,12,6,6) (H×W=12×6, 5色+SKULL)
      self.board_resblock = FullPreActivationBlock(NUM_COLORS + 1, n_board_mid, n_board_blocks, use_dropout=True)
      self.board_fc0 = L.Linear(n_board_flatten_in, n_board_fc0_out) #, initialW=w)
      self.board_fc1 = L.Linear(n_board_fc0_out,    n_board_fc1_out) #, initialW=w)
      
      # ネクスト入力：(B,H,W,C)=(None,8,2,5) 8個先まで見える
      self.nexts_resblock = FullPreActivationBlock(NUM_COLORS, n_nexts_mid, n_nexts_blocks, use_dropout=True)
      self.nexts_fc0 = L.Linear(n_nexts_flatten_in, n_nexts_fc0_out) #, initialW=w)
      self.nexts_fc1 = L.Linear(n_nexts_fc0_out,    n_nexts_fc1_out) #, initialW=w)

      # 盤面とネクスト結合
      self.all_fc0 = L.Linear(n_all_fc0_in,  n_all_fc0_out) #, initialW=w)
      self.all_fc1 = L.Linear(n_all_fc0_out, n_all_fc1_out) #, initialW=w)
      self.all_fc2 = L.Linear(n_all_fc1_out, NUM_ACTIONS) #, initialW=w)

  # x0: 盤面, x1: ネクスト
  def __call__(self, x0, x1):
    batch_size = len(x0)
    assert(len(x0) == len(x1))

    y0 = self.board_resblock(x0)
    # print('board_resblock y0 {}'.format(y0.shape))
    y0 = F.reshape(y0, (batch_size, -1))
    # print('reshape y0 {}'.format(y0.shape))
    y0 = F.relu(self.board_fc0(y0))
    y0 = F.relu(self.board_fc1(y0))

    y1 = self.nexts_resblock(x1)
    # print('nexts_resblock y1 {}'.format(y1.shape))
    y1 = F.reshape(y1, (batch_size, -1))
    # print('reshape y1 {}'.format(y1.shape))
    y1 = F.relu(self.nexts_fc0(y1))
    y1 = F.relu(self.nexts_fc1(y1))

    y2 = F.concat((y0, y1))
    y2 = F.relu(self.all_fc0(y2))
    y2 = F.relu(self.all_fc1(y2))
    y2 = self.all_fc2(y2)
    return y2

def concat_samples(train_batch, gpu_id):
  # [(x1,t1), (x2, t2), ...]形式から，[[x1,x2,x3,...], [t1,t2,t3,...]]形式へ
  # @todo dataset.concat_examplesの使い方調べる
  #       x, t = chainer.dataset.concat_examples(train_batch, gpu_id)
  x0 = []
  x1 = []
  t = []
  for data in train_batch:
    x0.append(data[0][0]) # 盤面
    x1.append(data[0][1]) # ネクスト
    t.append(data[1])     # Action

  x0 = np.array(x0)
  x1 = np.array(x1)
  t = np.array(t)

  # GPU化
  if gpu_id is not None:
    x0 = chainer.dataset.to_device(gpu_id, x0)
    x1 = chainer.dataset.to_device(gpu_id, x1)
    t = chainer.dataset.to_device(gpu_id, t)
  
  return x0, x1, t

# def data_argumenting(dataset):
#   order_x0 = random.shuffle(range(NUM_COLORS + 1))
#   order_x1 = random.shuffle(range(NUM_COLORS))

#   for data in dataset:
#     x0 = data[0][0]
#     x1 = data[0][1]
#     t = data[1]

#   pass

GPU_ID = 0
def run_train(net, optimizer, dataset, save_path):
  gpu_id = GPU_ID # 使用するGPU番号
  
  n_batch = 64
  n_epoch = 10

  SAVE_MODEL_PER_ITER = 1000

  # GPUに転送
  if gpu_id is not None:
    net.to_gpu(gpu_id)
  # log
  results_train, results_valid = {}, {}
  results_train['loss'], results_train['accuracy'] = [], []
  results_valid['loss'], results_valid['accuracy'] = [], []

  # 入力データを分割
  train_val, test_data = split_dataset_random(dataset, int(len(dataset) * 0.8), seed=0)
  train, valid = split_dataset_random(train_val, int(len(train_val) * 0.8), seed=0)

  # iteration数出力
  print('# of epoch:', n_epoch)
  print('# of batch:', n_batch)
  print('# of iteration:', max(n_epoch, n_epoch * len(train) / n_batch), '\n')

  # ぷよぷよAIを参考にbatch_sizeは64
  train_iter = SerialIterator(train, batch_size=n_batch, repeat=True, shuffle=True)
  
  count = 0
  for epoch in range(n_epoch):
    while True:
      # ミニバッチの取得
      train_batch = train_iter.next()

      # [(x1,t1), (x2, t2), ...]形式から，[[x1,x2,x3,...], [t1,t2,t3,...]]形式へ
      x0_train, x1_train, t_train = concat_samples(train_batch, gpu_id)

      # 予測値と目的関数の計算
      y_train = net(x0_train, x1_train)
      loss_train = F.softmax_cross_entropy(y_train, t_train)
      acc_train = F.accuracy(y_train, t_train)

      # 勾配の初期化と勾配の計算
      net.cleargrads()
      loss_train.backward()
      # パラメータの更新
      optimizer.update()
      
      # iteration カウントアップ
      count += 1

      # 1000 iterationごとにモデルを保存
      if count % SAVE_MODEL_PER_ITER == 0:
        # 各epochのモデルの保存
        save_filename = os.path.join(save_path, 'net_{:03d}.npz'.format(count))
        save_model(net, save_filename)
        print('save model (iteration {}) to {}\n'.format(count, save_filename))

      # 1エポック終えたら、valid データで評価する
      if train_iter.is_new_epoch or count % SAVE_MODEL_PER_ITER == 0:
        # 検証用データに対する結果の確認
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
          # x_valid, t_valid = chainer.dataset.concat_examples(valid, gpu_id)
          x0_valid, x1_valid, t_valid = concat_samples(valid, gpu_id)
          y_valid = net(x0_valid, x1_valid)
          loss_valid = F.softmax_cross_entropy(y_valid, t_valid)
          acc_valid = F.accuracy(y_valid, t_valid)
        # 注意：GPU で計算した結果はGPU上に存在するため、CPU上に転送します
        if gpu_id is not None:
          loss_train.to_cpu()
          loss_valid.to_cpu()
          acc_train.to_cpu()
          acc_valid.to_cpu()
        # 結果の表示
        print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}\n'
              'acc (train): {:.4f}, acc (valid): {:.4f}\n'.format(
              epoch, count, loss_train.array.mean(), loss_valid.array.mean(),
              acc_train.array.mean(), acc_valid.array.mean()))
        if train_iter.is_new_epoch:
          # 可視化用に保存
          results_train['loss'] .append(loss_train.array)
          results_train['accuracy'] .append(acc_train.array)
          results_valid['loss'].append(loss_valid.array)
          results_valid['accuracy'].append(acc_valid.array)
          break

  # 各epochのモデルの保存
  save_filename = os.path.join(save_path, 'net_final.npz')
  save_model(net, save_filename)
  print('save model to {}\n'.format(count, save_filename))

  # 損失 (loss)
  plt.plot(results_train['loss'], label='train')  # label で凡例の設定
  plt.plot(results_valid['loss'], label='valid')  # label で凡例の設定
  plt.legend()  # 凡例の表示
  plt.savefig(os.path.join(save_path, 'loss.png'))
  plt.figure()
  # 精度 (accuracy)
  plt.plot(results_train['accuracy'], label='train')  # label で凡例の設定
  plt.plot(results_valid['accuracy'], label='valid')  # label で凡例の設定
  plt.legend()  # 凡例の表示
  plt.savefig(os.path.join(save_path, 'accuracy.png'))

def save_model(net, filename='net.npz'):
  if GPU_ID:
    net.to_cpu()
  serializers.save_npz(filename, net)

def main(log_dir, gpu_id, file_npz, save_path):
  # log_dir以下のファイルを読み込む
  print('*** [START] load files in {} ***'.format(log_dir), file=sys.stderr)
  log_contents = load_smash_the_code_log(log_dir)
  print('*** [ END ] load files in {} ***'.format(log_dir), file=sys.stderr)
  print(file=sys.stderr)

  # 教師あり学習のデータセットに変換
  print('*** [START] transform raw logtxt to dataset ***', file=sys.stderr)
  dataset = transform_logs_for_supervised_learning(log_contents)
  print('*** len(dataset) = {} ***'.format(len(dataset)), file=sys.stderr)
  print('*** [ END ] transform raw logtxt to dataset ***', file=sys.stderr)
  print(file=sys.stderr)

  # ネットワーク定義
  print('*** [START] define neural network ***', file=sys.stderr)
  net = SmashTheCodeNet()
  if file_npz:
    serializers.load_npz(file_npz, net)

  # 最適化手法の設定 (momentum参考：http://tadaoyamaoka.hatenablog.com/entry/2017/11/03/095652)
  # lr=0.1だとoverflowした
  optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
  optimizer.setup(net)
  print('*** [ END ] define neural network ***', file=sys.stderr)
  print(file=sys.stderr)

  # 学習
  print('*** [START] training ***', file=sys.stderr)
  run_train(net, optimizer, dataset, save_path)
  print('*** [ END ] training ***', file=sys.stderr)

if __name__ == "__main__":
  argc = len(sys.argv)
  file_npz = None
  save_path = None

  if argc < 2:
    print("usage: python supervised_learning.py log_root_dir [save_path] [net.npz]")
  if argc >= 3:
    save_path = sys.argv[2]
  if argc >= 4:
    if sys.argv[3] is not "None":
      file_npz = sys.argv[3]
  if argc > 5:
    print("too many arg")
    exit(0)

  main(sys.argv[1], None, file_npz, save_path)