#include<iostream>
#include <string>
#include <cmath>
#include <ctime>

#define ACC 0.000001

int main(int arg, char** argv) {
    int N = std::atoi(argv[1]); //arguments
    int ITER = std::atoi(argv[2]); 
    double** mas = new double* [N]; //memory
    double** anew = new double* [N];
    for (int i = 0; i < N; i++)
    {
        mas[i] = new double[N];
        anew[i] = new double[N];
    }
    for (int i = 0; i < N; i++) //zeros
    {
        for (int j = 0; j < N; j++)
        {
            mas[i][j] = 0;
            anew[i][j] = 0;
        }
    }
    int rep = 0; //variables
    double err = 1.0;
    mas[0][0] = 10; //corners
    mas[N - 1][N - 1] = 30;
    mas[0][N - 1] = 20;
    mas[N - 1][0] = 20;
    anew[0][0] = 10;
    anew[N - 1][N - 1] = 30;
    anew[0][N - 1] = 20;
    anew[N - 1][0] = 20;
    clock_t befin = clock();
    //copying data to GPU
#pragma acc enter data copyin (err, mas[0:N][0:N], anew[0:N][0:N]) 
    {
#pragma acc data present (anew, mas)
#pragma acc parallel loop gang vector
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
#pragma acc wait(1)
        std::cout << "Initialization Time: " << 1.0 * (clock() - befin) / CLOCKS_PER_SEC << std::endl;
        clock_t befca = clock();
        //matrix and finding error
        while (rep < ITER && err >= ACC)
        {
            rep++;
            err = 0;
#pragma acc update device (err) //updates in variable in GPU from CPU
#pragma acc data present(anew, mas,err)
#pragma acc parallel loop collapse(2) independent vector gang reduction(max:err)		  
            for (int i = 1; i < N - 2; i++)
            {
                for (int j = 1; j < N - 2; j++)
                {
                    anew[i][j] = (mas[i - 1][j] + mas[i][j - 1] + mas[i + 1][j] + mas[i][j + 1]) / 4;
                    err = fmax(err, fabs(mas[i][j] - anew[i][j]));
                }
            }

#pragma acc update host(err) //updates in variable in CPU from GPU
#pragma acc wait(2) 
            double** c = mas; //updates matrix
            mas = anew;
            anew = c;
            //    std::cout << rep << std::endl;
        }
        std::cout << "Calculation time: " << 1.0 * (clock() - befca) / CLOCKS_PER_SEC << std::endl;
    }
    std::cout << "Iteration: " << rep << " " << "Error: " << err << std::endl;
    delete[] mas; //free
    delete[] anew;
    return 0;
}
