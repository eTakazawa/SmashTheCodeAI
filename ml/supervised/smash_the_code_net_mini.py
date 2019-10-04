import chainer
import chainer.links as L
import chainer.functions as F

from game_const import *

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

class SmashTheCodeMiniNet(chainer.Chain):
  def __init__(self, n_board_mid=256, n_nexts_mid=128):
    # n_mid: フィルタ（チャンネル）数 層を深くするより，フィルタ数を増やした方が良いらしい?
    
    super(SmashTheCodeMiniNet, self).__init__()
    # w = chainer.initializers.HeNormal()
    
    # 全結合層のノード数
    ## 盤面
    n_board_flatten_in = BOARD_HEIGHT * BOARD_WIDTH * n_board_mid
    n_board_fc0_out = n_board_flatten_in
    n_board_fc1_out = n_board_fc0_out // 2
    ## ネクスト
    n_nexts_flatten_in = NUM_NEXTS * NEXT_SIZE * n_nexts_mid
    n_nexts_fc0_out = n_nexts_flatten_in
    n_nexts_fc1_out = n_nexts_fc0_out // 2
    ## concated
    n_all_fc0_in = n_board_fc1_out + n_nexts_fc1_out
    n_all_fc0_out = 512
    n_all_fc1_out = 256

    with self.init_scope():
      # 盤面入力：(B,H,W,C)=(None,12,6,6) (H×W=12×6, 5色+SKULL)
      self.board_resblock = BaseBlock(NUM_COLORS + 1, n_board_mid, use_dropout=False)
      self.board_fc0 = L.Linear(n_board_flatten_in, n_board_fc0_out) #, initialW=w)
      self.board_fc1 = L.Linear(n_board_fc0_out,    n_board_fc1_out) #, initialW=w)
      
      # ネクスト入力：(B,H,W,C)=(None,8,2,5) 8個先まで見える
      self.nexts_resblock = BaseBlock(NUM_COLORS, n_nexts_mid, use_dropout=False)
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