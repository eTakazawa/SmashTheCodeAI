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
using namespace std;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
int dr[] = {0,1,0,-1};
int dc[] = {1,0,-1,0};
vector<char> colorAs(8),colorBs(8); // color of the first block,color of the attached block
void printErrBoard(vector<string>& board) {
  for(int i=0; i<12; i++) {
    for(int j=0; j<6; j++) {
      cerr << board[i][j] << " ";
    }
    cerr << endl;
  }
}
struct ZobristHash{
  int table[12][6][6];
  ZobristHash(){
    srand(time(0));
    for(int i=0;i<12;i++)for(int j=0;j<6;j++)for(int k=0;k<6;k++)
      table[i][j][k] = rand();
  }
  int hash(const vector<string>& board){
    int h = 0;
    for(int i=0;i<12;i++){
      for(int j=0;j<6;j++){
        if( board[i][j] != '.' ){
          int k = board[i][j] - '0';
          h ^= table[i][j][k];
        }
      }
    }
    return h;
  }
    
};
// utility of Combo
struct ComboUtility {
  int comboSeed3,comboSeed2,comboCount,blocksCount,hakkaHeight;
  int newBlockColumn[2],newBlockRow[2];
  int B,CP,CB,GB;
  int deletedColor[6];
  bool putSkullOnBoard(vector<string>& board, int column){
    int row = -1;
    for(int i=11; i>=0; i--){
      if( board[i][column] == '.' ){
        row = i;
        break;
      }
    }
    if(row == -1)return false;
    board[row][column] = '0';
    return true;
  }
  void putSkullLinesOnBoard(vector<string>& board,int numOfLines){
    for(int i=0;i<numOfLines;i++){
      for(int j=0;j<6;j++){
        putSkullOnBoard( board, j);
      }
    }
  }
  bool putBlocksOnBoard(vector<string>& board,int column,int colA,int colB,int rotate) {
    if( column == 5 && rotate == 0 || column == 0 && rotate == 2 )return false;
    
    if( rotate == 0 ){
      int row1 = -1;
      for(int i=11; i>=0; i--){
        if( board[i][column] == '.' ){
          row1 = i;
          break;
        }
      }
      if( row1 == -1 )return false;
      int row2 = -1;
      for(int i=11; i>=0; i--){
        if( board[i][column + 1] == '.'){
          row2 = i;
          break;
        }
      }
      if( row2 == -1 )return false;
      board[row1][column] = colA;
      board[row2][column + 1] = colB;
      newBlockRow[0] = row1;newBlockColumn[0] = column;
      newBlockColumn[1] = column + 1;newBlockRow[1] = row2;
    }else if( rotate == 2 ){
      int row1 = -1;
      for(int i=11; i>=0; i--){
        if( board[i][column] == '.' ){
          row1 = i;
          break;
        }
      }
      if( row1 == -1 )return false;
      int row2 = -1;
      for(int i=11; i>=0; i--){
        if( board[i][column - 1] == '.'){
          row2 = i;
          break;
        }
      }
      if( row2 == -1 )return false;
      board[row1][column] = colA;
      board[row2][column - 1] = colB;
      newBlockRow[0] = row1;newBlockColumn[0] = column;
      newBlockRow[1] = row2;newBlockColumn[1] = column - 1;
    }else{
      int row = -1;
      for(int i=11; i>=1; i--){
        if( board[i][column] == '.' ){
          row = i;
          break;
        }
      }
      if( row == -1 )return false;
      
      if( rotate == 1 )swap(colA,colB);
      board[row][column] = colB;
      board[row-1][column] = colA;
      newBlockRow[0] = row;newBlockColumn[0] = column;
      newBlockRow[1] = row - 1;newBlockColumn[1] = column;
    }
    
    return true;
  }
  int deleteBlocks(vector<string>& board) {
    bool isCombo = false;
    vector<vector<int>> used(12,vector<int>(6));
    blocksCount = 0;
    for(int i=0; i<12; i++) {
      for(int j=0; j<6; j++) {
        if(board[i][j] != '.')blocksCount++;
        if( board[i][j] != '.' && board[i][j] != '0' && used[i][j] == 0) {
          vector<int> delRow,delColumn;
          queue<int> qr,qc;
          qr.push(i);qc.push(j);
          used[i][j] = 1;
          
          // breadth search for connectivity
          while( !qr.empty() ) {
            int r = qr.front(), c = qc.front();qr.pop();qc.pop();
            delRow.push_back(r);delColumn.push_back(c);
            for(int k=0; k<4; k++) {
              int nr = r + dr[k], nc = c + dc[k];
              if( nr < 0 || nc < 0 || nr >= 12 || nc >= 6)continue;
              if( board[nr][nc] != board[r][c] || used[nr][nc])continue;
              qr.push(nr);qc.push(nc);
              used[nr][nc] = 1;
            }
          }
          // if connect 4 same color blocks delete and ComboCount + 1
          if(delRow.size() == 2)comboSeed2++;
          else if(delRow.size() == 3)comboSeed3++;
          else if(delRow.size() >= 4) {
            for(int k=0;k<2;k++){
              if( newBlockRow[k] == i && newBlockColumn[k] == j )
                hakkaHeight = min( hakkaHeight, i);
            }
            B += delRow.size();
            GB += min((int)delRow.size() - 4, 8);
            deletedColor[board[delRow[0]][delColumn[0]] - '0'] = 1;
            isCombo = true;
            for(int k=0; k<delRow.size(); k++) {
              for(int l=0; l<4; l++) {
                int nr = delRow[k] + dr[l], nc = delColumn[k] + dc[l];
                if( nr < 0 || nc < 0 || nr >= 12 || nc >= 6)continue;
                if( board[nr][nc] == '0' ) board[nr][nc] = '.';
              }
              board[delRow[k]][delColumn[k]] = '.';
            }
          }
        }
      }
    }
    return isCombo;
  }
  void packComboedBoard(vector<string>& board) {
    for(int j=0; j<6; j++) {
      int emptyPos = 0;
      for(int i=11; i>=0; i--) {
        if( board[i][j] == '.' && emptyPos < i ) emptyPos = i;
        else if( board[i][j] != '.' && emptyPos > i) {
          board[emptyPos][j] = board[i][j];
          board[i][j] = '.';
          emptyPos--;
        }
      }
    }
  }
  double comboBoard(vector<string>& board) {
    double score = 0.0;
    comboCount = comboSeed2 = comboSeed3 = 0;
    B = CP = CB = GB = 0;
    memset( deletedColor, 0, sizeof(deletedColor) );
    // for(int i=0;i<6;i++)deletedColor[i] = 0;
    hakkaHeight = 12;
    // GB and B
    bool firstCombo = true;
    while(deleteBlocks(board)) {
      packComboedBoard(board);
      
      // CP
      if( CP == 0 && (!firstCombo))CP = 8;
      else CP *= 2;
      if( firstCombo )firstCombo = false;
      
      // CB
      int tmp = 0;
      for(int i=1;i<6;i++)tmp += deletedColor[i];
      if( tmp == 1 )CB = 0;
      else CB = pow( 2, tmp-1);
      
      score += (10 * B)*max(1,min(999,(CP + CB + GB)));
      B = CB = GB = 0;
      memset( deletedColor, 0, sizeof(deletedColor) );
      // for(int i=0;i<6;i++)deletedColor[i] = 0;
      comboCount++;
    }
    
    
     // cerr << "B CP CB GB COMBO" << endl;
  //  cerr << B << " " << CP << " " << CB << " " << GB << " " << 
    //  (10 * B)*max(1,min(999,(CP + CB + GB))) << endl;
    // cerr << comboCounter << endl;
    
    return score / 70.;
  }
};

bool opPinchi;
struct State {
  static ComboUtility comboUtil;
  static ZobristHash zobristHash;
  static int nowTurn;
  vector<string> board,comboedBoard;
  int comboSeed2,comboSeed3,comboCount,blocksCount;
  int tate2,tate3;
  int maxHeight,maxmaxHeight,opMaxHeight,hakkaHeight;
  double colorDiff;
  double nuisancePoints;
  bool check,isDamaged,isNext;
  int turn,firstColumn,firstRotate,skullCount;
  double value;
  int getHash(){
    return zobristHash.hash(board);
  }
  State(vector<string> board_,int turn_) {
    turn = turn_;
    board = board_;
    firstColumn = -1;
    firstRotate = -1;
    comboedBoard = board_;
    nuisancePoints = comboUtil.comboBoard(comboedBoard);
    nuisancePoints = -100000;
    isNext = isDamaged = check = false;
    comboSeed2 = comboUtil.comboSeed2;
    comboSeed3 = comboUtil.comboSeed3;
    comboCount = comboUtil.comboCount;
    blocksCount = comboUtil.blocksCount;
    hakkaHeight = comboUtil.hakkaHeight;
    skullCount = tate2 = tate3 = 0;
    getInfoForValue();
  }
  State(vector<string> board_,int turn_,int firstColumn_,int firstRotate_) {
    turn = turn_;
    board = board_;
    firstColumn = firstColumn_;
    firstRotate = firstRotate_;
    comboedBoard = board_;
    nuisancePoints = comboUtil.comboBoard(comboedBoard);
    isNext = isDamaged = check = false;
    comboSeed2 = comboUtil.comboSeed2;
    comboSeed3 = comboUtil.comboSeed3;
    comboCount = comboUtil.comboCount;
    blocksCount = comboUtil.blocksCount;
    hakkaHeight = comboUtil.hakkaHeight;
    skullCount =tate2 = tate3 = 0;
    getInfoForValue();
  }
  
  double getValue() const{
    // cerr << "nuisancePoints: "<< nuisancePoints << endl;
    double sikii = 4.0;
    if(isDamaged)sikii = 2.0;
    // if(skullCount >= 36)sikii = 1.0;
    if(check)return -1000;
    // if( maxHeight <= 2 )return 500000 + nuisancePoints*100 - turn*1000 + comboSeed2*5 + comboSeed3*10;//return nuisancePoints*10 + comboSeed2 + comboSeed3*2;
    
    if( turn <= 9 && nuisancePoints/6 >= 3)return 1000000 + min(12,(int)(nuisancePoints/6))*5 - turn*1000;// + hakkaHeight/100.; // or * nuisance
    if( opPinchi && nuisancePoints/6 >= 1 )return 1100000 + min(12,(int)(nuisancePoints/6))*5 - turn*1000;// + hakkaHeight/100.; // or * nuisance//200000 + nuisancePoints + comboSeed2 + comboSeed3*2 - turn*1000;
    if( (nuisancePoints/6) >= sikii )return 1000000 + min(12,(int)(nuisancePoints/6))*5 - turn*1000;// + hakkaHeight/100.; // or * nuisance
    if( comboCount == 1 || comboCount == 2 )return -10;
    if( nuisancePoints/6 < sikii )return (!isDamaged && turn < 10 ? tate3 + tate2 - maxmaxHeight/10. : 0) + comboSeed2*5 + comboSeed3*10 - colorDiff/3. - turn;/// (1+turn/500.);
    return nuisancePoints*10;  //+ comboSeed3*2 + maxHeight/6.0;// / (1+turn/500.);
  }
  void getInfoForValue(){
    maxHeight = -1;
    maxmaxHeight = 13;
    colorDiff = 0;
    int left[] = {6,6,6,6,6};
    int right[] = {0,0,0,0,0};
    int maxLeft = 6,maxRight = 0,prevtate3 = -1;
    for(int i=0;i<12;i++){
      for(int j=0;j<6;j++){
        if(board[i][j] != '.'){
          if(board[i][j] != '0'){
            maxmaxHeight = min(maxmaxHeight, i);
            left[board[i][j] - '1'] = min( left[board[i][j] - '1'], j);
            right[board[i][j] - '1'] = max( right[board[i][j] - '1'], j);
            maxLeft = min( j, maxLeft);
            maxRight = max( j, maxRight);
            
            if( i < 10 && ( j == 2 || j == 3) && !isDamaged){
              if( board[i][j] == board[i+1][j] && board[i+1][j] == board[i+2][j]){
                tate3 += 100;
                if(  prevtate3 == i-3 ){
                  if( prevtate3 > 1 && board[prevtate3-1][j] == board[i][j] )
                    tate3 += 10;
                }
                prevtate3 = i;
              }else if(board[i][j] == board[i+1][j]){
                tate2 += 50;
              }else if( board[i][j] != board[i+1][j] && board[i+1][j] != board[i+2][j]){
                tate3 -= 30;
                tate2 -= 30;
              }
            }
          }else{
            isDamaged = true;
            skullCount++;
          }
        }else maxHeight = i;
      }
    }
    // if(maxLeft != 6)colorDiff += (maxRight - maxLeft);
    if(maxHeight == -1)maxHeight = 12;
    for(int i=0;i<5;i++)
      if(left[i] != 6)colorDiff += (right[i] - left[i] <= 1 ? 0 : right[i] - left[i]);
  }
  vector<State*> generateNextStates(){
    vector<State*> resStates;
    
    static const int ord[] = {2,3,1,4,0,5};
    // if( comboCount >= 3 )return resStates;
    // if( !isDamaged && comboCount > 0)return resStates;
    if(isNext)return resStates;
    isNext = true;
    for(int rot=0;rot<4;rot++){
      for(int col=0; col<2; col++){
        vector<string> temp = comboedBoard;
        if(!comboUtil.putBlocksOnBoard(temp, ord[col], colorAs[turn], colorBs[turn], rot))continue;
        if(firstColumn == -1)resStates.push_back(new State(temp, turn + 1, ord[col], rot));
        else resStates.push_back(new State(temp, turn + 1, firstColumn, firstRotate));
      }
    }
    if(resStates.size() == 0 || isDamaged || turn >= 10 ){
      for(int rot=0;rot<4;rot++){
        for(int col=2; col<6; col++){
          vector<string> temp = comboedBoard;
          if(!comboUtil.putBlocksOnBoard(temp, ord[col], colorAs[turn], colorBs[turn], rot))continue;
          if(firstColumn == -1)resStates.push_back(new State(temp, turn + 1, ord[col], rot));
          else resStates.push_back(new State(temp, turn + 1, firstColumn, firstRotate));
         }
      }
    }
    if(resStates.size() == 0)check = true;
    return resStates;
  }
  bool operator<(const State& state)const {
    return getValue() < state.getValue();
  }
};
ComboUtility State::comboUtil;
ZobristHash State::zobristHash;
int State::nowTurn;

struct StateCompare {
  bool operator()(State* s1,State* s2)const {
    return *s1 < *s2;
  }
};
struct Timer{
  chrono::system_clock::time_point startTime;
  Timer(){
     startTime = chrono::system_clock::now();
  }
  int getDuration(){
     return chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - startTime).count();
  }
};

int score = 0;
void solve(vector<string> board,int turn) {
  Timer timer;
  
  State::nowTurn = turn;
  int MaxTurn = 8,beamBreadth = 25;
  // int MaxTurn =6,beamBreadth = 30;
  vector<priority_queue<State*,vector<State*>,StateCompare>> nowStatesQue(MaxTurn+1);
  nowStatesQue[0].push(new State(board, turn));
  
  set<int> searchedStates;
  searchedStates.insert(nowStatesQue[0].top()->getHash());
  
  int chokudaiBreadth = 2;
  
  while(timer.getDuration() <= 90){
    for(int t=0; t < MaxTurn; t++){
      for(int i=0; i < chokudaiBreadth; i++){
        if( nowStatesQue[t].empty() )break;
        if( timer.getDuration() >= 95 )break;
        State* nowState = nowStatesQue[t].top();nowStatesQue[t].pop();
        if(nowState->firstColumn != -1)nowStatesQue[t+1].push(nowState);
        // generate next states
        vector<State*> nextStates = nowState->generateNextStates();
        for(int j=0;j<nextStates.size();j++){
          State* nextState = nextStates[j];
          if(searchedStates.find(nextState->getHash()) != searchedStates.end()){
            continue;
          }
          nowStatesQue[t+1].push(nextState);
          searchedStates.insert(nextState->getHash());
        }
      }
    }
  }
  
  if(nowStatesQue[MaxTurn].empty())cout << 0 << " " << 0 << " !? " << endl;
  else{
    State* bestState = nowStatesQue[MaxTurn].top();
    cout << bestState->firstColumn << " " << bestState->firstRotate << " ";
    // if(bestState->turn - turn > 1)cout << "About " << bestState->turn - turn - 1 << " turns left." << endl;
    // else if(bestState->getValue() == 0)cout << "Help!!!" << endl;
    // else cout << "Fire!!" << endl;
    cout << turn+1 << "/" << bestState->turn << "/";
    cout << (int)bestState->getValue() << "/";
    
    cout << (int)bestState->nuisancePoints*70 << endl;
    cerr << "hakka:" << bestState->hakkaHeight << endl;
    auto msec = timer.getDuration();
    cerr << msec << " ms" << endl;    
  }
}


int main()
{
  // game loop
  int turn = 0;
  ComboUtility comboutil;
  while (1) {
    vector<int> colorCounter(6);
    vector<string> myBoard(12,"......"),opBoard(12);
    if(turn == 0){
      for (int i = 0; i < 8; i++) {
        cin >> colorAs[i] >> colorBs[i];
        cin.ignore();
      }
    }else{
      for (int i = 0; i < 8; i++) {
        char cA,cB;
        cin >> cA >> cB;
        cin.ignore();
        if( i == 7){
          colorAs.push_back(cA);
          colorBs.push_back(cB);
        }
      }
    }
int tsc;
cin >> tsc;
    for (int i = 0; i < 12; i++) {
      cin >> myBoard[i];
      cin.ignore();
    }
cin >> tsc;
    for (int i = 0; i < 12; i++) {
      // One line of the map ('.' = empty, '0' = skull block, '1' to '5' = colored block)
      cin >> opBoard[i];
      cin.ignore();
    }
     
    opPinchi = false;
    int mhtmp = 0;
    for(int i=0;i<12;i++)for(int j=0;j<6;j++)
      if( opBoard[i][j] == '.' )mhtmp = i;
    if( mhtmp <= 7 )opPinchi = true;
    
    // State* state = new State(opBoard, turn);
    // vector<State*> nextStates = state->generateNextStates();
    // double maxSkull = 0;
    // for(int i=0;i<nextStates.size();i++)
    //   maxSkull = max( maxSkull, nextStates[i]->nuisancePoints/6 );
    
    // for(int i=0;i<3;i++){
    //   comboutil.putSkullLinesOnBoard( opBoard, 1);
    // }
  
    // Write an action using cout. DON'T FORGET THE "<< endl"
    // To debug: cerr << "Debug messages..." << endl;
    cerr << "turn: "<< turn << endl;
    solve(myBoard, turn);
    turn++;
    // exit(0);
  }
}

/*
  State* testState = new State(board, turn);
  vector<State*> nextTestStates = testState->generateNextStates();
  for(int i=0;i<nextTestStates.size();i++){
    cerr << i << endl;
    printErrBoard( nextTestStates[i]->board );
  }
  return;
*/
/*
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;
*/
/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
 /*
int main()
{

  // game loop
  while (1) {
    for (int i = 0; i < 8; i++) {
      int colorA; // color of the first block
      int colorB; // color of the attached block
      cin >> colorA >> colorB; cin.ignore();
    }
    int score1;
    cin >> score1; cin.ignore();
    for (int i = 0; i < 12; i++) {
      string row;
      cin >> row; cin.ignore();
    }
    int score2;
    cin >> score2; cin.ignore();
    for (int i = 0; i < 12; i++) {
      string row; // One line of the map ('.' = empty, '0' = skull block, '1' to '5' = colored block)
      cin >> row; cin.ignore();
    }

    // Write an action using cout. DON'T FORGET THE "<< endl"
    // To debug: cerr << "Debug messages..." << endl;

    cout << "0" << endl; // "x": the column in which to drop your blocks
  }
}
*/