import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.links as L
import chainer.functions as F

from chainer.datasets import TupleDataset
from chainer.datasets import split_dataset_random
from chainer.iterators import SerialIterator
from chainer import optimizers

BOARD_HEIGHT = 12
BOARD_WIDTH = 6
NUM_COLORS = 6
NUM_NEXTS = 8
NEXT_SIZE = 2

NUM_ACTIONS = 1

DROP_OUT_RATIO = 0.3 # 0.3~0.4くらいが良いらしい?

# Full pre-activation https://www.slideshare.net/masatakanishimori/res-net
class BaseBlock(chainer.Chain):
  def __init__(self, n_in, n_out, stride=1, use_dropout=False):
    super(BaseBlock, self).__init__()
    
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
    x = F.leaky_relu(self.bn1(x)) # reluよりleaky_reluが良いらしい
    x = F.leaky_relu(self.bn2(self.conv1(x)))
    if self.use_dropout:
      x = F.dropout(x, DROP_OUT_RATIO)
    x = self.conv2(x)
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
      x = getattr(self, 'block{}'.format(str(block_id)))(x)
    return x

class SmashTheCodeNet(chainer.Chain):
  def __init__(self, n_board_mid=256, n_nexts_mid=128, n_board_blocks=12, n_nexts_blocks=5):
    # n_mid: フィルタ（チャンネル）数 層を深くするより，フィルタ数を増やした方が良いらしい?
    
    super(SmashTheCodeNet, self).__init__()
    w = chainer.initializers.HeNormal()
    
    # 全結合層のノード数
    ## 盤面
    n_board_flatten_in = BOARD_HEIGHT * BOARD_WIDTH * n_board_mid
    n_board_fc0_out = 512
    n_board_fc1_out = n_board_fc0_out / 2
    ## ネクスト
    n_nexts_flatten_in = NUM_NEXTS * NEXT_SIZE * n_nexts_mid
    n_nexts_fc0_out = 128
    n_nexts_fc1_out = n_nexts_fc0_out / 2
    ## concated
    n_all_fc0_in = n_board_fc1_out + n_nexts_fc1_out
    n_all_fc0_out = 256
    n_all_fc1_out = 128

    with self.init_scope():
      # 盤面入力：(B,H,W,C)=(None,12,6,6) (H×W=12×6, 5色+SKULL)
      self.board_resblock = FullPreActivationBlock(NUM_COLORS + 1, n_board_mid, n_board_blocks)
      self.board_fc0 = L.Linear(n_board_flatten_in, n_board_fc0_out, initialW=w)
      self.board_fc1 = L.Linear(n_board_fc0_out,    n_board_fc1_out, initialW=w)
      
      # ネクスト入力：(B,H,W,C)=(None,8,2,5) 8個先まで見える
      self.nexts_resblock = FullPreActivationBlock(NUM_COLORS, n_nexts_mid, n_nexts_blocks)
      self.nexts_fc0 = L.Linear(n_nexts_flatten_in, n_nexts_fc0_out, initialW=w)
      self.nexts_fc1 = L.Linear(n_nexts_fc0_out,    n_nexts_fc1_out, initialW=w)

      # 盤面とネクスト結合
      self.all_fc0 = L.Linear(n_all_fc0_in,  n_all_fc0_out, initialW=w)
      self.all_fc1 = L.Linear(n_all_fc0_out, n_all_fc1_out, initialW=w)
      self.all_fc2 = L.Linear(n_all_fc1_out, NUM_ACTIONS, initialW=w)


  # x0: 盤面, x1: ネクスト
  def __call__(self, x0, x1):
    batch_size = len(x0[0])
    assert(len(x0[0]) == len(x1[0]))

    y0 = self.board_resblock(x0)
    y0 = F.reshape(y0, (batch_size, -1))
    y0 = F.relu(self.board_fc0(y0))
    y0 = F.relu(self.board_fc1(y0))

    y1 = self.nexts_resblock(x1)
    y1 = F.reshape(y1, (batch_size, -1))
    y1 = F.relu(self.nexts_fc0(y1))
    y1 = F.relu(self.nexts_fc1(y1))

    y2 = F.concat((y0, y1))
    y2 = F.relu(self.all_fc0(y2))
    y2 = F.relu(self.all_fc1(y2))
    y2 = self.all_fc2(y2)
    return y2

def train(net, optimizer, dataset):
  gpu_id = 0 # 使用するGPU番号
  n_batch = 64
  n_epoch = 10

  # GPUに転送
  net.to_gpu(gpu_id)
  # log
  results_train, results_valid = {}, {}
  results_train['loss'], results_train['accuracy'] = [], []
  results_valid['loss'], results_valid['accuracy'] = [], []

  # 入力データを分割
  train_val, test_data = split_dataset_random(dataset, int(len(dataset) * 0.8), seed=0)
  train, valid = split_dataset_random(train_val, int(len(train_val) * 0.8), seed=0)
  # ぷよぷよAIを参考にbatch_sizeは64
  train_iter = SerialIterator(train, batch_size=n_batch, repeat=True, shuffle=True)

  count = 1
  for epoch in range(n_epoch):
    while True:
      # ミニバッチの取得
      train_batch = train_iter.next()
      # x と t に分割 GPUにデータを送るため，gpu_id指定
      # [(x1,t1), (x2, t2), ...]形式から，[[x1,x2,x3,...], [t1,t2,t3,...]]形式へ
      x_train, t_train = chainer.dataset.concat_examples(train_batch, gpu_id)
      # 予測値と目的関数の計算
      y_train = net(x_train)
      loss_train = F.softmax_cross_entropy(y_train, t_train)
      acc_train = F.accuracy(y_train, t_train)

      # 勾配の初期化と勾配の計算
      net.cleargrads()
      loss_train.backward()
      # パラメータの更新
      optimizer.update()
      # カウントアップ
      count += 1

      # 1エポック終えたら、valid データで評価する
      if train_iter.is_new_epoch:
          # 検証用データに対する結果の確認
          with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
              x_valid, t_valid = chainer.dataset.concat_examples(valid, gpu_id)
              y_valid = net(x_valid)
              loss_valid = F.softmax_cross_entropy(y_valid, t_valid)
              acc_valid = F.accuracy(y_valid, t_valid)
          # 注意：GPU で計算した結果はGPU上に存在するため、CPU上に転送します
          loss_train.to_cpu()
          loss_valid.to_cpu()
          acc_train.to_cpu()
          acc_valid.to_cpu()
          # 結果の表示
          print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'
                'acc (train): {:.4f}, acc (valid): {:.4f}'.format(
              epoch, count, loss_train.array.mean(), loss_valid.array.mean(),
                acc_train.array.mean(), acc_valid.array.mean()))
          # 可視化用に保存
          results_train['loss'] .append(loss_train.array)
          results_train['accuracy'] .append(acc_train.array)
          results_valid['loss'].append(loss_valid.array)
          results_valid['accuracy'].append(acc_valid.array)
          break

  # 損失 (loss)
  plt.plot(results_train['loss'], label='train')  # label で凡例の設定
  plt.plot(results_valid['loss'], label='valid')  # label で凡例の設定
  plt.legend()  # 凡例の表示
  plt.show()

  # 精度 (accuracy)
  plt.plot(results_train['accuracy'], label='train')  # label で凡例の設定
  plt.plot(results_valid['accuracy'], label='valid')  # label で凡例の設定
  plt.legend()  # 凡例の表示
  plt.show()

def load_smash_the_code_log(log_dir):
  pass

def transform_logs_for_supervised_learning(logs):
  pass

def main(log_dir):
  # log_dir以下のファイルを読み込む
  logs = load_smash_the_code_log(log_dir)
  # 教師あり学習のデータセット(TupleDataset)に変換
  dataset = transform_logs_for_supervised_learning(logs)
  # ネットワーク定義
  net = SmashTheCodeNet()
  # 最適化手法の設定 (momentum参考：http://tadaoyamaoka.hatenablog.com/entry/2017/11/03/095652)
  optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
  optimizer.setup(net)

if __name__ == "__main__":
  main()