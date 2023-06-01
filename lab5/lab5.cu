#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <iomanip>

#include <mpi.h> 

//расчёт пограничных значений матрицы на месте "среза"
__global__ void calculateBoundaries(double* mas, double* anew, size_t n, size_t sizePerGpu)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (j == 0 || j > n - 2) return;

    if (j < n)
    {
        anew[i * n + j] = 0.25 * (mas[i * n + j - 1] + mas[(i - 1) * n + j] + mas[(i + 1) * n + j] + mas[i * n + j + 1]);
        anew[(sizePerGpu - 2) * n + i] = 0.25 * (mas[(sizePerGpu - 2) * n + i - 1] + mas[((sizePerGpu - 2) - 1) * n + i] + mas[((sizePerGpu - 2) + 1) * n + i] + mas[(sizePerGpu - 2) * n + i + 1]);
    }
}


//расчёт внутренних значений матрицы 
__global__ void calculationMatrix(double* anew, const double* mas, size_t n, size_t groupSize)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < groupSize - 1 && j > 0 && j < n - 1) //промежуток подсчета исключая границы
    {
        anew[i * n + j] = 0.25 * (mas[i * n + j - 1] + mas[(i - 1) * n + j] +
            mas[(i + 1) * n + j] + mas[i * n + j + 1]);
    }
}


// Функция, подсчитывающая разницу матриц
__global__ void getErrorMatrix(double* mas, double* anew, double* outputMatrix, size_t n, size_t sizePerGpu)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

	size_t idx = i * n + j;
	if(!(j == 0 || i == 0 || j == n - 1 || i == sizePerGpu - 1))
	{
		outputMatrix[idx] = std::abs(anew[idx] - mas[idx]);
	}
}
int main(int argc, char** argv) {

    double* mas, * anew, * d_mas, * d_anew, * deviceError, * errorMatrix, * tempStorage = NULL;
    int rank, sizeOfTheGroup; //номер текущего процесса, кол-во всех возможных процессов

    MPI_Init(&argc, &argv); //кол-во процессов и 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //инициализация
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfTheGroup);

    int numOfDevices = 0;
    cudaGetDeviceCount(&numOfDevices);
    if (sizeOfTheGroup > numOfDevices || sizeOfTheGroup < 1)
    {
        std::cout << "Invalid number of devices!";
        std::exit(-1);
    }

    cudaSetDevice(rank);//устанавливает

    int N = atoi(argv[1]); //параметр, размер сетки SIZE
    int ITER = std::atoi(argv[2]); //параметр, макс количество итераций iter max
    float ACC = std::atof(argv[3]); //параметр, мин значение ошибки
    int totalSize = N * N;

    //защита от пользователя
    if (N == 0 || N < 0) {
        std::cout << "N error" << std::endl;
        return EXIT_FAILURE;
    }

    if (ITER > 1000000 || ITER < 0)
    {
        std::cout << "ITER error" << std::endl;
        return EXIT_FAILURE;
    }

    if (ACC < 0.0000001) {
        std::cout << "ACC error" << std::endl;
        return EXIT_FAILURE;
    }

    //часть SIZE для каждого процесса
    size_t sizeOfAreaForOneProcess = N / sizeOfTheGroup;

    //начало части 
    size_t startYIdx = sizeOfAreaForOneProcess * rank;

    // выделение памяти на хосте 
    cudaMallocHost(&mas, sizeof(double) * totalSize);
    cudaMallocHost(&anew, sizeof(double) * totalSize);

    std::memset(mas, 0, N * N * sizeof(double));//нулями заполняем


    //создание массивов на хосте 

    mas[0] = 10; //левый верхний угол матрицы
    mas[N - 1] = 20; //правый верхний угол матрицы
    mas[N * (N - 1)] = 20; //нижний левый угол матрицы
    mas[N * N - 1] = 30; //нижний правый угол матрицы

    clock_t befin = clock(); //начнаю отсчёт времени инициализации 
    for (int i = 1; i < N - 1; i++) //запоняю верхнюю границу матрицы
        mas[i] = mas[i - 1] + (mas[N - 1] - mas[0]) / (N - 1);

    for (int i = 1; i < N - 1; i++) //заполняю левую, правую и нижнуюю границы матрицы
    {
        mas[N * (N - 1) + i] = mas[N * (N - 1) + i - 1] + (mas[N * N - 1] - mas[N * (N - 1)]) / (N - 1);
        mas[N * i] = mas[N * i - N] + (mas[N * N - 1] - mas[N - 1]) / (N - 1);
        mas[(N)*i + (N - 1)] = mas[(N) * (i - 1) + (N - 1)] + (mas[N * N - 1] - mas[N - 1]) / (N - 1);
    }

    for (int i = 0; i < N * N; i++) //копирую
        anew[i] = mas[i];

    std::memcpy(anew, mas, totalSize * sizeof(double));//

    //double* deviceMatrixAPtr, *d_anew, *deviceError, *errorMatrix, *tempStorage = NULL;


      // Расчитываем, сколько памяти требуется процессу
    if (rank != 0 && rank != sizeOfTheGroup - 1)
    {
        sizeOfAreaForOneProcess += 2; // 2 еденицы памяти на добавление границ
    }
    else
    {
        sizeOfAreaForOneProcess += 1; //otherwise
    }

    size_t sizeOfAllocatedMemory = N * sizeOfAreaForOneProcess;

    //выделяем память на gpu 
    cudaMalloc((void**)&d_mas, sizeOfAllocatedMemory * sizeof(double));
    cudaMalloc((void**)&d_anew, sizeOfAllocatedMemory * sizeof(double));
    cudaMalloc((void**)&errorMatrix, sizeOfAllocatedMemory * sizeof(double));
    cudaMalloc((void**)&deviceError, sizeof(double));

    // Копируем часть заполненной матрицы в выделенную память, начиная с 1 строки
    size_t offset = (rank != 0) ? N : 0; //тернарный опрератор (если ранк не = 0, то сайз)

    cudaMemcpy(d_mas, mas + (startYIdx * N) - offset,
        sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice); //память на каждый процесс

    cudaMemcpy(d_anew, anew + (startYIdx * N) - offset,
        sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice);


    // Здесь мы получаем размер временного буфера для редукции и выделяем память для этого буфера
    size_t tempStorageSize = 0;

    cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, N * sizeOfAreaForOneProcess);

    cudaMalloc((void**)&tempStorage, tempStorageSize);

    int iter = 0;
    double* error;
    cudaMallocHost(&error, sizeof(double));//память на хосте
    *error = 1.0;


    //вычисление размеров сетки и кол-во потоков 
    unsigned int threads_x = (N < 1024) ? N : 1024;// кол-во потоков, видеокарта боль 1024 не позволяет (св-во видеокарты) 
    unsigned int blocks_y = sizeOfAreaForOneProcess; // кол-во блоков 
    unsigned int blocks_x = N / threads_x;

    dim3 blockDim1(threads_x, 1); // размеры одного блока в потоках
    dim3 gridDim1(blocks_x, blocks_y); //размер сетки в блоках

    cudaStream_t stream, matrixCalculationStream; //создаю поток
    cudaStreamCreate(&stream);
    cudaStreamCreate(&matrixCalculationStream);

    clock_t begin = clock(); //начало отсчёта времени


    while (iter < ITER && (*error) > ACC) {
        iter += 1;

        // Расчитываем границы, которые потом будем отправлять другим процессам
        calculateBoundaries <<<N, 1, 0, stream >>> (d_mas, d_anew, N, sizeOfAreaForOneProcess);

        cudaStreamSynchronize(stream);

        // Расчет матрицы
        calculationMatrix<<<gridDim, blockDim, 0, matrixCalculationStream >>> (d_mas, d_anew, N, sizeOfAreaForOneProcess);

        if (iter% 100 == 0) {
            getErrorMatrix <<<gridDim, blockDim, 0, matrixCalculationStream >>> (d_mas, d_anew, errorMatrix,
                N, sizeOfAreaForOneProcess);

            cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, sizeOfAllocatedMemory, matrixCalculationStream);

            // Находим максимальную ошибку среди всех и передаём её всем процессам
            MPI_Allreduce((void*)deviceError, (void*)deviceError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);//по всем процессам передаётся занч ошибки
            cudaMemcpyAsync(error, deviceError, sizeof(double), cudaMemcpyDeviceToHost, matrixCalculationStream);
        }


        if (rank != 0) //верхние границы не с кем обменивать :(
        {
            	MPI_Sendrecv(d_anew+ N + 1, N - 2, MPI_DOUBLE, rank - 1, 0,
                d_anew+ 1, N - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Обмен с нижней границей
        if (rank != sizeOfTheGroup - 1)
        {
           	MPI_Sendrecv(d_anew+ (sizeOfAreaForOneProcess - 2) * N + 1, N - 2, MPI_DOUBLE, rank + 1, 0, d_anew+ (sizeOfAreaForOneProcess - 1) * N + 1, N - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        cudaStreamSynchronize(matrixCalculationStream);

        //обмен указателям
        double* c = d_mas;
        d_mas= d_anew;
        d_anew = c;


    }
    clock_t end = clock();
    if (rank == 0) //печатаем 1 раз
    {
        std::cout << "Time: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl;
        std::cout << "Iter: " << iter << " Error: " << *error << std::endl;
    }

    cudaFree(d_mas);
    cudaFree(d_anew);
    cudaFree(tempStorage);
    cudaFree(mas);
    cudaFree(anew);
    cudaFree(errorMatrix);
    cudaStreamDestroy(stream);

    //Функция закрывает все MPI-процессы и ликвидирует все области связи
    MPI_Finalize();
    return 0;
}

