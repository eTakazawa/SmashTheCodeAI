# -*- coding:utf-8 -*-
# AIプログラムとゲームマネージャー間の通信を取り持つ
import subprocess
import sys
import traceback

class GameObjectIO(object):
  def __init__(self):
    self.colors = []
    self.curr_scores = []
    self.boards = []
  
  def receive_from_manager_process(self, manager_process):
    # First 8 lines: 2 space separated integers colorA and colorB: 
    # The colors of the blocks you must place in your grid by order 
    # of appearance.
    for _ in range(8):
      line_colorA_B = manager_process.stdout.readline()
      self.colors = line_colorA_B.decode('utf-8').strip().split()
    
    playerNum = 2
    for _ in range(playerNum):
      # Next line: * current score.
      curr_score = manager_process.stdout.readline().decode('utf-8').strip()
      self.curr_scores.append(curr_score)

      # Next 12 lines: 6 characters representing one 
      # line of * grid, top to bottom.
      board = []
      for _ in range(12):
        blocks_line = manager_process.stdout.readline().decode('utf-8').strip()
        board.append(list(blocks_line))
      self.boards.append(board)
    
  def __repr__(self):
    return repr(self.boards)

class GameServer(object):
  def __init__(self, commands):
    # 起動commandsは3つ managerプログラムと対戦するAIプログラム
    assert len(commands) == 3, "python game_server.py manager.out ai1.out ai2.out"
    self.manager_process = None
    self.ai_processes = []

    try:
      self.manager_process = subprocess.Popen(commands[0], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
      self.ai_processes.append(subprocess.Popen(commands[1], stdin=subprocess.PIPE, stdout=subprocess.PIPE))
      self.ai_processes.append(subprocess.Popen(commands[2], stdin=subprocess.PIPE, stdout=subprocess.PIPE))
    except:
      traceback.print_exc()

  def run(self):
    try:
      # manager_process.stdout から ai_processes[0] への入力を受け取り
      manager_process.stdout.readline()

      # ai_processes[i].stdin へ送信
      pass
      # ai_processes[i].stdout から応答を受信
      pass

    except:
      traceback.print_exc()
      # proc.kill()
      return False
    return True
  
  def close(self):
    self.manager_process.kill()
    for ai_process in self.ai_processes:
      ai_process.kill()


def game_object_test():
  command = input().strip()
  manager_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  io = GameObjectIO()
  io.receive_from_manager_process(manager_process)
  print(io)

def main():
  game_object_test()

if __name__ == "__main__":
  main()