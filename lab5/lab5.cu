#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <iomanip>

#include <mpi.h> 

void print(int size, int rank, double* matrixA, int sizeOfAreaForOneProcess){
		if (rank == 0){
		printf("start rank 0\n");
		for (int i = 0; i < sizeOfAreaForOneProcess; i++){
			for (int j = 0; j < size; j++){
				printf("%0.4lf ", matrixA[i * size + j]);
			}
			printf("\n");
		}
		printf("end rank 0\n");
	}
	if (rank == 1){
		printf("start rank 1\n");
		for (int i = 0; i < sizeOfAreaForOneProcess; i++){
			for (int j = 0; j < size; j++){
				printf("%0.4lf ", matrixA[i * size + j]);
			}
			printf("\n");
		}
		printf("end rank 1\n");
	}
	if (rank == 2){
		printf("start rank 2\n");
		for (int i = 0; i < sizeOfAreaForOneProcess; i++){
			for (int j = 0; j < size; j++){
				printf("%0.4lf ", matrixA[i * size + j]);
			}
			printf("\n");
		}
		printf("end rank 2\n");
	}

	if (rank == 3){
		printf("start rank 3\n");
		for (int i = 0; i < sizeOfAreaForOneProcess; i++){
			for (int j = 0; j < size; j++){
				printf("%0.4lf ", matrixA[i * size + j]);
			}
			printf("\n");
		}
		printf("end rank 3\n");
	}

}

//расчёт внутренних значений матрицы 
#define CALCULATE(matrixA, matrixB, size, i, j) \
	matrixB[i * size + j] = 0.25 * (matrixA[i * size + j - 1] + matrixA[(i - 1) * size + j] + \
			matrixA[(i + 1) * size + j] + matrixA[i * size + j + 1]);	

__global__
void calculateBoundaries(double* matrixA, double* matrixB, size_t size, size_t sizePerGpu)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0 || idx > size - 2) return;
	
	if(idx < size)
	{
		CALCULATE(matrixA, matrixB, size, 1, idx);
		CALCULATE(matrixA, matrixB, size, (sizePerGpu - 2), idx);
	}
}

// Р“Р»Р°РІРЅР°СЏ С„СѓРЅРєС†РёСЏ - СЂР°СЃС‡С‘С‚ РїРѕР»СЏ 
__global__
void calculationMatrix(double* matrixA, double* matrixB, size_t size, size_t sizePerGpu)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if((i < sizePerGpu - 2) && (i > 1) && (j > 0) && (j < size - 1))
	{
		CALCULATE(matrixA, matrixB, size, i, j);
	}
}

// Р¤СѓРЅРєС†РёСЏ, РїРѕРґСЃС‡РёС‚С‹РІР°СЋС‰Р°СЏ СЂР°Р·РЅРёС†Сѓ РјР°С‚СЂРёС†
__global__
void getErrorMatrix(double* matrixA, double* matrixB, double* outputMatrix, size_t size, size_t sizePerGpu)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

	size_t idx = i * size + j;
	if(!(j == 0 || i == 0 || j == size - 1 || i == sizePerGpu - 1))
	{
		outputMatrix[idx] = std::abs(matrixB[idx] - matrixA[idx]);
	}
}

void printMatrix(double *arrPrint, int n) {//для вывода сеток
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            std::cout << arrPrint[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {

    double* mas, *anew, * d_mas, * d_anew, * deviceError, * errorMatrix, * tempStorage = NULL;
    int rank, sizeOfTheGroup; //номер текущего процесса, кол-во всех возможных процессов

    MPI_Init(&argc, &argv); //кол-во процессов и 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //инициализация
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfTheGroup); 
 cudaSetDevice(rank);//устанавливает
    int numOfDevices = 0;
    cudaGetDeviceCount(&numOfDevices);
    if (sizeOfTheGroup > numOfDevices || sizeOfTheGroup < 1)
    {
        std::cout << "Invalid number of devices!";
        std::exit(-1);
    }

   

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
size_t sizeOfAreaForOneProcess; 
	size_t startYIdx;

	if (N % sizeOfTheGroup == 0) {
		sizeOfAreaForOneProcess = N / sizeOfTheGroup;

	}
	else {
		if (sizeOfTheGroup == 2) {
			if (N % 2 != 0){
				sizeOfAreaForOneProcess = std::ceil(N / (sizeOfTheGroup)) + 1;

				if (rank == (sizeOfTheGroup - 1)) sizeOfAreaForOneProcess = (N / (sizeOfTheGroup));
				}
			else {
				sizeOfAreaForOneProcess = N / (sizeOfTheGroup);
			}
		}
		if (sizeOfTheGroup == 4) {
			sizeOfAreaForOneProcess = N / (sizeOfTheGroup - 1);

			if (rank == (sizeOfTheGroup - 1)) sizeOfAreaForOneProcess = N % (sizeOfTheGroup - 1);
		}
	}

	
	if (sizeOfTheGroup == 2){ 
		if (N % 2 == 0){
			startYIdx = (N / sizeOfTheGroup) * rank;
		}
		else {
		startYIdx = (N / sizeOfTheGroup) * rank;
		if (rank == 1) startYIdx++;
		}
	}
	if (sizeOfTheGroup == 4) {	
		if (N % sizeOfTheGroup == 0) startYIdx = (N / (sizeOfTheGroup)) * rank;
		else {startYIdx = (N / (sizeOfTheGroup - 1)) * rank;}
		}
	if (sizeOfTheGroup == 1) {startYIdx = 0;}

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

    std::memcpy(anew, mas, totalSize * sizeof(double));//копирует содержимое одной области памяти в другую

      // Расчитываем, сколько памяти требуется процессу
    if(sizeOfTheGroup == 1) sizeOfAreaForOneProcess = sizeOfAreaForOneProcess;

	else {
		if (rank != 0 && rank != sizeOfTheGroup - 1) {
		sizeOfAreaForOneProcess += 2;
		}
		else {
		sizeOfAreaForOneProcess += 1;
		}
	}

    size_t sizeOfAllocatedMemory = N * sizeOfAreaForOneProcess;

    //выделяем память на gpu 
    cudaMalloc((void**)&d_mas, sizeOfAllocatedMemory * sizeof(double));
    cudaMalloc((void**)&d_anew, sizeOfAllocatedMemory * sizeof(double));
    cudaMalloc((void**)&errorMatrix, sizeOfAllocatedMemory * sizeof(double));
    cudaMalloc((void**)&deviceError, sizeof(double));

    // Копируем часть заполненной матрицы в выделенную память, начиная с 1 строки
    size_t offset = (rank != 0) ? N : 0; //тернарный опрератор (если ранк не = 0, то Н)

    cudaMemcpy(d_mas, mas + (startYIdx * N) - offset, sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice); //память на каждый процесс
    cudaMemcpy(d_anew, anew + (startYIdx * N) - offset, sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice);


    // Здесь мы получаем размер временного буфера для редукции и выделяем память для этого буфера
    size_t tempStorageSize = 0;

    cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, N * sizeOfAreaForOneProcess); //выделяем память

    cudaMalloc((void**)&tempStorage, tempStorageSize);

    int iter = 0;
    double* error;
    cudaMallocHost(&error, sizeof(double));//память на хосте
    *error = 1.0;


    //вычисление размеров сетки и кол-во потоков 
    unsigned int threads_x = (N < 1024) ? N : 1024;// кол-во потоков, видеокарта больше 1024 не позволяет (св-во видеокарты) 
    unsigned int blocks_y = sizeOfAreaForOneProcess; // кол-во блоков 
    unsigned int blocks_x = N / threads_x;

    dim3 blockDim(threads_x, 1); // размеры одного блока в потоках
    dim3 gridDim(blocks_x, blocks_y); //размер сетки в блоках

    cudaStream_t stream, matrixCalculationStream; //инициализирую поток
    cudaStreamCreate(&stream);
    cudaStreamCreate(&matrixCalculationStream);

    clock_t begin = clock(); //начало отсчёта времени


    while (iter < ITER && (*error) > ACC) {
        iter += 1;

        // Расчитываем границы, которые потом будем отправлять другим процессам
	if(sizeOfAreaForOneProcess > 2){
        	calculateBoundaries <<<N, 1, 0, stream >>> (d_mas, d_anew, N, sizeOfAreaForOneProcess);

        cudaStreamSynchronize(stream);//ждёт завершения всех операций в потоке

        // Расчет матрицы
        calculationMatrix <<<gridDim, blockDim, 0, matrixCalculationStream >>> (d_mas, d_anew, N, sizeOfAreaForOneProcess);
}
        if (iter % 100 == 0) {
            getErrorMatrix <<<gridDim, blockDim, 0, matrixCalculationStream >>> (d_mas, d_anew, errorMatrix, N, sizeOfAreaForOneProcess); //операция вычисления разницы матриц
            cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, sizeOfAllocatedMemory, matrixCalculationStream); //находим максимальное число
cudaStreamSynchronize(matrixCalculationStream);
            MPI_Allreduce((void*)deviceError, (void*)deviceError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);//по всем процессам передаётся занч ошибки
            cudaMemcpy(&error, deviceError, sizeof(double), cudaMemcpyDeviceToHost);//память на каждый процесс
	   // std::cout<<*error<<std::endl;
        }


        if(sizeOfTheGroup > 1){ 
			// РћР±РјРµРЅ "РіСЂР°РЅРёС‡РЅС‹РјРё" СѓСЃР»РѕРІРёСЏРјРё РєР°Р¶РґРѕР№ РѕР±Р»Р°СЃС‚Рё
			// РћР±РјРµРЅ РІРµСЂС…РЅРµР№ РіСЂР°РЅРёС†РµР№
			if (rank != 0)
			{
				MPI_Sendrecv(d_anew + N + 1, N - 2, MPI_DOUBLE, rank - 1, 0, 
				d_anew + 1, N - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			// РћР±РјРµРЅ РЅРёР¶РЅРµР№ РіСЂР°РЅРёС†РµР№
			if (rank != sizeOfTheGroup - 1)
			{
				MPI_Sendrecv(d_anew + (sizeOfAreaForOneProcess - 2) * N + 1, N - 2, MPI_DOUBLE, rank + 1, 0,
								d_anew + (sizeOfAreaForOneProcess - 1) * N + 1, 
								N - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}

        cudaStreamSynchronize(matrixCalculationStream);//ждёт завершения всех операций в потоке

        //обмен указателям
        double* c = d_mas;
        d_mas = d_anew;
        d_anew = c;
    }
    clock_t end = clock();
    if (rank == 0) //печатаем 1 раз
    {
        std::cout << "Time: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl;
        std::cout << "Iter: " << iter << " Error: " << *error << std::endl;
    }

print(N, rank, mas, sizeOfAreaForOneProcess);

	//очищаем память
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
