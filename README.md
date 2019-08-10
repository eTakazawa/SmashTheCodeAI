# 概要
深層強化学習を使ってCodingameのSmash the CodeでLegend 1位を目指す

# ディレクトリ構成
- ai（過去に作ったAI）
- game（自動対戦）
- ml（機械学習用）
  - supervised（教師あり学習）
- doc
- experiments（実験用）

# Build & Run
WIP

# ToDo
- [x] 作成したAIを戦わせる環境を作る
  - [x] テスト：実際の環境とスコアの一致×3
- [ ] 対戦用弱めのAIの作成
  - [x] random AI
  - [ ] 浅めの探索AI
- [ ] 対戦ログを集める
  - [x] ログ形式の整理
  - [ ] submitted v.s. randomの1000戦分作成
- [ ] 教師あり学習
  - [ ] 自盤面とnextのみを入力とする
    - 3コンボぐらいを目指す

- [x] 定数をどこかにまとめる
  - [ ] クラスごとに必要な定数を分離（テンプレートが良い？）
- [ ] パイプ閉じる
```
terminate called after throwing an instance of 'boost::process::process_error'
  what():  pipe(2) failed: Too many open files
Abort trap: 6
```

- [ ] 対戦AIのタイムアウト設定（止まらない時がある）

## Refacter
- [ ] パイプ通信をutilにする
  - [ ] 送信・受信関数
  - [ ] GM IOみたいなクラスでも良さそう
- [ ] クラス・関数分け
  - [ ] Board, Action
- [x] 定数分離
  - [ ] クラスにまとめる