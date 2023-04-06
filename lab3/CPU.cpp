#include<iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <openacc.h>

#define ACC 0.000001

int main(int arg, char** argv) {
    int N = std::atoi(argv[1]);
    int ITER = std::atoi(argv[2]);
    double* mas = new double[N * N];
    double* anew = new double[N * N];

    for (int i = 0; i < N * N; i++)
    {
        mas[i] = 0;
        anew[i] = 0;
    }

    int rep = 0;
    double err = 1.0;
    mas[0] = 10;
    mas[N - 1] = 30;
    mas[N * N - 1] = 20;
    mas[N * (N - 1)] = 20;
    anew[0] = 10;
    anew[N * N - 1] = 30;
    anew[N * (N - 1)] = 20;
    anew[N * N - 1] = 20;
    clock_t befin = clock();
    for (int i = 1; i < N - 1; i++)
    {
        mas[i] = mas[i - 1] + (mas[N - 1] - mas[0]) / N;
        mas[N * (N - 1) + i] = mas[N * (N - 1) + i - 1] + (mas[N * N - 1] - mas[N * (N - 1)]) / N;
        mas[N * i] = mas[(i - 1) * N] + (mas[N * N - 1] - mas[N * (N - 1)]) / N;
        mas[N * i + (N - 1)] = mas[N * (i - 1) + (N - 1)] + (mas[N * N - 1] - mas[N - 1]) / N;
        anew[i] = mas[i];
        anew[N * (N - 1) + i] = mas[N * (N - 1) + i];
        anew[N * i] = mas[N * i];
        anew[N * i + (N - 1)] = mas[N * i + (N - 1)];
    }
#pragma acc enter data copyin(err, anew[0:N*N], mas[0:N*N])
    {
        std::cout << "Initialization Time: " << 1.0 * (clock() - befin) / CLOCKS_PER_SEC << std::endl;
        clock_t befca = clock();
        while (rep < ITER && err >= ACC)
        {
            rep++;
            err = 0;
#pragma acc update device(err)
            //std::cout<<err<<std::endl; 
#pragma acc data present(anew, mas,err)
#pragma acc parallel loop independent collapse(2) async ()
            for (int i = 1; i < N - 2; i++)
            {
                for (int j = 1; j < N - 2; j++)
                {
                    anew[i * N + j] = (mas[i * N + j - 1] + mas[(i - 1) * N + j] + mas[(i + 1) * N + j] + mas[i * N + j + 1]) / 4;
                    err = fmax(err, fabs(mas[i * N + j] - anew[i * N + j]));
                }
            }
#pragma acc update host (err)
            if (rep % 100 == 0) {
#pragma acc wait ()
#pragma acc data present(anew, mas)
            }
            double* c = mas;
            mas = anew;
            anew = c;
            std::cout << rep << "  " << err << std::endl;
        }
        std::cout << "Calculation time: " << 1.0 * (clock() - befca) / CLOCKS_PER_SEC << std::endl;
    }
    std::cout << "Iteration: " << rep << " " << "Error: " << err << std::endl;
    delete[] mas;
    delete[] anew;
    return 0;
}
