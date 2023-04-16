#include<iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <openacc.h>
#include <cublas_v2.h>

int main(int arg, char** argv) {
    int N = std::atoi(argv[1]);
    int ITER = std::atoi(argv[2]);
    float ACC = std::atof(argv[3]);


    cublasHandle_t handle;
    cublasStatus_t stat;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
  	std::cout<<"CUBLAS initialization failed"<<std::endl;
	return EXIT_FAILURE;
    }

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
    mas[N - 1] = 20;
    mas[N * (N - 1)] = 20; 
    mas[N * N-1] = 30;

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
#pragma acc enter data copyin(anew[0:N*N], mas[0:N*N])
    {
        std::cout << "Initialization Time: " << 1.0 * (clock() - befin) / CLOCKS_PER_SEC << std::endl;
        clock_t befca = clock();
	double a = -1;
	int index = 0;

        while (rep < ITER && err >= ACC)
        {
	    #pragma acc data present(anew, mas)
            #pragma acc parallel loop independent collapse(2) async ()
            for (int i = 1; i < N - 1; i++)
                for (int j = 1; j < N - 1; j++)
                    anew[i * N + j] = (mas[i * N + j - 1] + mas[(i - 1) * N + j] + mas[(i + 1) * N + j] + mas[i * N + j + 1]) *0.25;
	   if (rep % 100 == 0)
	    {
		#pragma acc wait ()
		#pragma acc host_data use_device(anew,mas)
		{
			stat = cublasDaxpy(handle, N * N, &a, anew, 1, mas, 1);
			if (stat != CUBLAS_STATUS_SUCCESS) {
				std::cout<<"CUBLAS initialization failed1"<<std::endl;
				return EXIT_FAILURE;}
			stat =cublasIdamax(handle, N*N, mas, 1, &index);
			if (stat != CUBLAS_STATUS_SUCCESS) {
			std::cout<<"CUBLAS initialization failed2"<<std::endl;
				return EXIT_FAILURE;}
            	}
	   #pragma acc update host(mas[index - 1]) 
	   err = fabs(mas[index - 1]);

           #pragma acc host_data use_device(mas, anew)
	   stat = cublasDcopy(handle, N * N, anew, 1, mas, 1);
	   if (stat != CUBLAS_STATUS_SUCCESS) {
		std::cout<<"CUBLAS initialization failed2"<<std::endl;
		return EXIT_FAILURE;}
	   }
           double* c = mas;
           mas = anew;
           anew = c;
           rep++;
           std::cout << rep << "  " << err << std::endl;
        }
        std::cout << "Calculation time: " << 1.0 * (clock() - befca) / CLOCKS_PER_SEC << std::endl;
    }
    std::cout << "Iteration: " << rep << " " << "Error: " << err << std::endl;
    delete[] mas;
    delete[] anew;
    return 0;
}
