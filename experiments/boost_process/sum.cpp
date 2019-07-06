#include <iostream>

using namespace std;

void sum() {
  int n;
  cin >> n;
  int sum = 0, tmp;
  for (int i = 0; i < n; i++) {
    cin >> tmp;
    sum += tmp;
  }
  cout << sum << endl;
}

int main(void) {
  for (;;) {
    sum();
  }

  return 0;
}