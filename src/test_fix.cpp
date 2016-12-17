#include "sg14/fixed_point"
#include <iostream>
using namespace std;
typedef sg14::fixed_point<int32_t,-20> myfp;
int main(){
  float a=1;
  myfp x=a;
  myfp y;
  y=x*x;
  y=y/x;
  cout<<y<<endl;
}
