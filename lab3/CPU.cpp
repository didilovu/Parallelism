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

    if (N == 0 || N < 0){
	std::cout<<"N error"<<std::endl;
	return EXIT_FAILURE;

    }

    if (ITER > 1000000 || ITER < 0)
    {
	std::cout<<"ITER error"<<std::endl;
	return EXIT_FAILURE;
    }

    if (ACC < 0.0000001){
	std::cout<<"ACC error"<<std::endl;
	return EXIT_FAILURE;
    }

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
    for (int i = 1; i < N-1; i++)
        mas[i] = mas[i-1] + (mas[N - 1] - mas[0]) / (N-1);

    for (int i = 1; i < N-1; i++)
    {
        mas[N * (N - 1) + i] = mas[N * (N - 1) + i-1] + (mas[N * N-1] - mas[N * (N - 1)]) / (N-1);
        mas[N*i] = mas[N*i-N] + (mas[N * N-1] - mas[N-1]) / (N-1);
        mas[(N)*i+(N-1)] = mas[(N)*(i-1)+(N-1)] + (mas[N * N - 1] - mas[N - 1]) / (N-1);
    }
  
	for (int i = 0; i < N*N; i++)
		anew[i] = mas[i];

    {
        std::cout << "Initialization Time: " << 1.0 * (clock() - befin) / CLOCKS_PER_SEC << std::endl;
        clock_t befca = clock();
        while (rep < ITER && err >= ACC)
        {
            rep++;
            err = 0;
            for (int i = 1; i < N - 2; i++)
            {
                for (int j = 1; j < N - 2; j++)
                {
                    anew[i * N + j] = (mas[i * N + j - 1] + mas[(i - 1) * N + j] + mas[(i + 1) * N + j] + mas[i * N + j + 1]) / 4;
                    err = fmax(err, fabs(mas[i * N + j] - anew[i * N + j]));
                }
            }
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
