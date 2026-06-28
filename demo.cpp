#include <iostream>
#include <cmath>
#include <string>
#include <bits/stdc++.h>

using namespace std;
// Problem no: 

int main()
{
    uint8_t flag = 0b00001000;
    cout<<(flag<<2)<<endl;
    cout<<(bitset<8>(flag<<2))[2]<<endl; 

}