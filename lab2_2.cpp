#include<iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <openacc.h>

//#define ITER 1000000
#define ACC 0.000001
//#define N 128
int main(int arg, char** argv) {
    //init
    int N = std::atoi(argv[1]);
    int ITER = std::atoi(argv[2]);
    double** mas = new double* [N];
    double** anew = new double* [N];
    for (int i = 0; i < N; i++)
    {
        mas[i] = new double[N];
        anew[i] = new double[N];
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            mas[i][j] = 0;
            anew[i][j] = 0;
        }
    }
    //corners
    int rep = 0;
    double err = 1.0;
    mas[0][0] = 10;
    mas[N - 1][N - 1] = 30;
    mas[0][N - 1] = 20;
    mas[N - 1][0] = 20;
    anew[0][0] = 10;
    anew[N - 1][N - 1] = 30;
    anew[0][N - 1] = 20;
    anew[N - 1][0] = 20;
    clock_t befin = clock();
#pragma acc enter data copyin (err, mas[0:N][0:N], anew[0:N][0:N])
#pragma acc data present ( anew, mas)
#pragma acc parallel loop gang num_gangs (256) vector vector_length(256) async(1)
    //borders
for (int i = 1; i < N - 1; i++)
{
    mas[0][i] = mas[0][0] + (mas[0][N - 1] - mas[0][0]) / (N - i);
    mas[i][0] = mas[0][0] + (mas[0][N - 1] - mas[0][0]) / (N - i);
    mas[i][N - 1] = mas[0][N - 1] + (mas[N - 1][N - 1] - mas[N - 1][0]) / (N - i);
    mas[N - 1][i] = mas[N - 1][0] + (mas[N - 1][N - 1] - mas[0][N - 1]) / (N - i);
    anew[0][i] = mas[0][0] + (mas[0][N - 1] - mas[0][0]) / (N - i);
    anew[i][0] = mas[0][0] + (mas[0][N - 1] - mas[0][0]) / (N - i);
    anew[i][N - 1] = mas[0][N - 1] + (mas[N - 1][N - 1] - mas[N - 1][0]) / (N - i);
    anew[N - 1][i] = mas[N - 1][0] + (mas[N - 1][N - 1] - mas[0][N - 1]) / (N - i);
}
#pragma acc wait(1) async(2)
std::cout << "Initialization Time: " << 1.0 * (clock() - befin) / CLOCKS_PER_SEC << std::endl;
{
    clock_t befca = clock();
    while (rep < ITER && err >= ACC)
    {
#pragma acc kernels async(2)
        err = 0;
#pragma acc update device (err) async(2)
        rep++;
#pragma acc data present(anew, mas,err)
#pragma acc parallel loop collapse(2) independent vector vector_length(256) gang num_gangs(256) reduction (max : err) async(2)
        //matrix
        for (int i = 1; i < N - 2; i++)
        {
            for (int j = 1; j < N - 2; j++)
            {
                anew[i][j] = (mas[i - 1][j] + mas[i][j - 1] + mas[i + 1][j] + mas[i][j + 1]) / 4;
                err = fmax(err, fabs(mas[i][j] - anew[i][j]));
            }
        }
#pragma acc update host(err) async(2)
#pragma acc wait(2)
        //swap
        double** c = mas;
        mas = anew;
        anew = c;
        std::cout << rep << std::endl;
    }
    std::cout << "Calculation time: " << 1.0 * (clock() - befca) / CLOCKS_PER_SEC << std::endl;
}
std::cout << "Iteration: " << rep << " " << "Error: " << err << std::endl;
//free
delete[] mas;
delete[] anew;
return 0;
}
