/**
 * g++ -lboost_filesystem-mt manager.cpp
 */

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

void test_pipe_2() {
  // https://www.boost.org/doc/libs/1_70_0/doc/html/process.html
  bp::ipstream from_sum_out;
  bp::opstream to_sum_out;
  bp::child c("sum.out", bp::std_in < to_sum_out, bp::std_out > from_sum_out);

  std::string line;

  for (int j = 0; j < 100; j++) {
    int n = 10;
    to_sum_out << n << endl;
    for (int i = 1; i <= n; i++) {
      to_sum_out << i << endl;
    }

    if (from_sum_out) {
      std::getline(from_sum_out, line);
      std::cerr << line << std::endl;
    }
  }

  c.terminate();
}

int main(void) {
  test_pipe_2();
  return 0;
}