#include <iostream>
#include <string>
#include <cmath>
#include <ctime> 
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda.h>

using namespace std;

__global__
void countnewmatrix(double* mas, double* anew, size_t n) //считаем новую матрицу
{
    size_t i = blockIdx.x; //получаем индексы, блок
    size_t j = threadIdx.x; //нить

    if (!(blockIdx.x == 0 || threadIdx.x == 0) && (i * n + j<= n*n))
        anew[i * n + j] = (mas[i * n + j - 1] + mas[(i - 1) * n + j] + mas[(i + 1) * n + j] + mas[i * n + j + 1]) * 0.25; //считаем поэлементно

}

__global__
void finderr(double* mas, double* anew, double* outMatrix, size_t n) //обновляем значение ошибки
{				//разм-ть блока
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; //получаем индекс элемента
    if (!(blockIdx.x == 0 || threadIdx.x == 0) && (idx <= n * n))
        outMatrix[idx] = fabs(anew[idx] - mas[idx]); //берём по модулю

}

int main(int arg, char** argv) {
    int N = atoi(argv[1]); //параметр, размер сетки
    int ITER = std::atoi(argv[2]); //параметр, макс количество итераций
    float ACC = std::atof(argv[3]); //параметр, мин значение ошибки

    //защита от пользователя
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


    double* mas = new double[N * N]; //создаем массив, который будет содержать старую копию  
    double* anew = new double[N * N]; //массив, который будет содержать новую версию

    for (int i = 0; i < N * N; i++)  //заполняю оба массива нулями, чтобы избавиться от возможного мусора
    {
        mas[i] = 0;
        anew[i] = 0;
    }

    int rep = 0; //инициализация переменной, отвечающей за отслеживания количества итераций 
    double err= 1.0; //инициализация переменной, отвечающей за отслеживания ошибки на определённой итерации

    mas[0] = 10; //левый верхний угол матрицы
    mas[N - 1] = 20; //правый верхний угол матрицы
    mas[N * (N - 1)] = 20; //нижний левый угол матрицы
    mas[N * N-1] = 30; //нижний правый угол матрицы

    clock_t befin = clock(); //начнаю отсчёт времени инициализации 
    for (int i = 1; i < N-1; i++) //запоняю верхнюю границу матрицы
        mas[i] = mas[i-1] + (mas[N - 1] - mas[0]) / (N-1);

    for (int i = 1; i < N-1; i++) //заполняю левую, правую и нижнуюю границы матрицы
    {
        mas[N * (N - 1) + i] = mas[N * (N - 1) + i-1] + (mas[N * N-1] - mas[N * (N - 1)]) / (N-1); 
        mas[N*i] = mas[N*i-N] + (mas[N * N-1] - mas[N-1]) / (N-1);
        mas[(N)*i+(N-1)] = mas[(N)*(i-1)+(N-1)] + (mas[N * N - 1] - mas[N - 1]) / (N-1);
    }
  
	for (int i = 0; i < N*N; i++) //копирую
		anew[i] = mas[i];

        cout << "Initialization Time: " << 1.0 * (clock() - befin) / CLOCKS_PER_SEC << endl; //вывод потраченного на инициализацию времени
        clock_t befca = clock();//начало отсчёта для выполнения вычислений

	//создаю переменные и матрицы для работы на девайсе
        double* mas_dev, * anew_dev, * deviceError, * errorMatrix, * tempStorage = NULL;
        size_t tempStorageSize = 0;
	
	//выделяю память для матриц и переменных
        cudaMalloc((void**)(&mas_dev), sizeof(double) * N * N);
        cudaMalloc((void**)(&anew_dev), sizeof(double) * N * N);
        cudaMalloc((void**)&deviceError, sizeof(double));
        cudaMalloc((void**)&errorMatrix, sizeof(double) * N * N);
	
	//копирую из хоста на девайс
        cudaMemcpy(mas_dev, mas, sizeof(double) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(anew_dev, anew, sizeof(double) * N * N, cudaMemcpyHostToDevice);

	bool graphCreated = false;
	cudaStream_t stream, memoryStream;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&memoryStream);
	cudaGraph_t graph;
	cudaGraphExec_t instance;


	//выделние памяти для временного хранения CUB
 	cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, N * N, stream); //NULL
        cudaMalloc(&tempStorage, tempStorageSize);

	//вычисление наиболее подходящего размер блока
	int blockS, minGridSize;
        int maxSize = 1024;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockS, countnewmatrix, 0, maxSize);
        dim3 blockSize = dim3(32, 32); //сведенья об устр-ве. 1024 макс. 1.кол-во потоков, 2.кол-во блоков.
        dim3 gridSize = dim3((N+blockSize .x-1)/blockSize .x,(N+blockSize .y-1)/blockSize .y);


	
        while ((rep < ITER) && (err > ACC)) //начинаем вычислять матрицу
        {
      	    	if (!graphCreated) {
			cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal); //захват графов на потоке в режиме 0
			for (size_t i = 0; i < 50; i++) {
				countnewmatrix <<<gridSize, blockSize, 0, stream >>> (mas_dev, anew_dev, N);
				countnewmatrix <<<gridSize, blockSize, 0, stream >>> (anew_dev, mas_dev, N);
			}
			cudaStreamEndCapture(stream, &graph);//заканчивает захват, возвр охваченный граф
			cudaGraphInstantiate(&instance, graph, NULL, NULL, 0); //Creates an executable graph from a graph.
			graphCreated = true;

		}
		else {
			cudaGraphLaunch(instance, stream); //загружает выполняемый граф в поток

           		finderr<<<gridSize, blockSize, 0, stream >>> (mas_dev, anew_dev, errorMatrix, N);
			cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, N*N, stream);
			cudaMemcpy(&err, deviceError, sizeof(double), cudaMemcpyDeviceToHost);

			rep += 100;
	    	}
            cout << rep << "  " << err << endl;  //вывод итерации и значения ошибки 
        }

    cout << "Calculation time: " << 1.0 * (clock() - befca) / CLOCKS_PER_SEC << endl; //вывод времени, потраченного на вычисления матрицы
    cout << "Iteration: " << rep << " " << "Error: " << err << endl; //сколько всего итераций потребовалось и достигнутое значение ошибки
    delete[] mas;  //освобождаем память
    delete[] anew;
    cudaFree(mas_dev);
    cudaFree(anew_dev);
    cudaFree(errorMatrix);
    cudaFree(tempStorage);
    return 0; //программа завершена успешно
}
