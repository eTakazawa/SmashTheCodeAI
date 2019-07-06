#include <bits/stdc++.h>
#include <boost/process.hpp>
#include <boost/filesystem.hpp>

namespace bp = boost::process;
using namespace std;

void test_system() {
  int ret = bp::system("ls");
  // a.out manager.cpp
  // と端末に出力
}

void test_pipe() {
  bp::ipstream is; //reading pipe-stream
  bp::child c(bp::search_path("nm"), "a.out", bp::std_out > is);

  std::vector<std::string> data;
  std::string line;

  while (c.running() && std::getline(is, line) && !line.empty())
    cout << line << endl;

  c.wait();
}

int main(void) {
  test_pipe();
  return 0;
}