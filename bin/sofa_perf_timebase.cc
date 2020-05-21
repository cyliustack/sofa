#include<sys/time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include<iostream>
#include<fstream>

double gettime(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    double t = tp.tv_sec + tp.tv_usec / 1000000.0; //get current timestamp in milliseconds
    return t;
}
int main(int argc, char* argv[]){
    double time = gettime();
    system("perf record -F 999 ls -lah /usr/bin > /dev/null");
    printf("%.6lf %.6lf\n",time,time);
    system("perf script 2>&1");
    return 0;
}

