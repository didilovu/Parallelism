#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <iomanip>

#include <mpi.h> 

//������ ����������� �������� ������� �� ����� "�����"
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


//������ ���������� �������� ������� 
__global__ void calculationMatrix(double* anew, const double* mas, size_t n, size_t groupSize)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < groupSize - 1 && j > 0 && j < n - 1) //���������� �������� �������� �������
    {
        anew[i * n + j] = 0.25 * (mas[i * n + j - 1] + mas[(i - 1) * n + j] +
            mas[(i + 1) * n + j] + mas[i * n + j + 1]);
    }
}


// �������, �������������� ������� ������
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
    int rank, sizeOfTheGroup; //����� �������� ��������, ���-�� ���� ��������� ���������

    MPI_Init(&argc, &argv); //���-�� ��������� � 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //�������������
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfTheGroup);

    int numOfDevices = 0;
    cudaGetDeviceCount(&numOfDevices);
    if (sizeOfTheGroup > numOfDevices || sizeOfTheGroup < 1)
    {
        std::cout << "Invalid number of devices!";
        std::exit(-1);
    }

    cudaSetDevice(rank);//�������������

    int N = atoi(argv[1]); //��������, ������ ����� SIZE
    int ITER = std::atoi(argv[2]); //��������, ���� ���������� �������� iter max
    float ACC = std::atof(argv[3]); //��������, ��� �������� ������
    int totalSize = N * N;

    //������ �� ������������
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

    //����� SIZE ��� ������� ��������
    size_t sizeOfAreaForOneProcess = N / sizeOfTheGroup;

    //������ ����� 
    size_t startYIdx = sizeOfAreaForOneProcess * rank;

    // ��������� ������ �� ����� 
    cudaMallocHost(&mas, sizeof(double) * totalSize);
    cudaMallocHost(&anew, sizeof(double) * totalSize);

    std::memset(mas, 0, N * N * sizeof(double));//������ ���������


    //�������� �������� �� ����� 

    mas[0] = 10; //����� ������� ���� �������
    mas[N - 1] = 20; //������ ������� ���� �������
    mas[N * (N - 1)] = 20; //������ ����� ���� �������
    mas[N * N - 1] = 30; //������ ������ ���� �������

    clock_t befin = clock(); //������ ������ ������� ������������� 
    for (int i = 1; i < N - 1; i++) //������� ������� ������� �������
        mas[i] = mas[i - 1] + (mas[N - 1] - mas[0]) / (N - 1);

    for (int i = 1; i < N - 1; i++) //�������� �����, ������ � ������� ������� �������
    {
        mas[N * (N - 1) + i] = mas[N * (N - 1) + i - 1] + (mas[N * N - 1] - mas[N * (N - 1)]) / (N - 1);
        mas[N * i] = mas[N * i - N] + (mas[N * N - 1] - mas[N - 1]) / (N - 1);
        mas[(N)*i + (N - 1)] = mas[(N) * (i - 1) + (N - 1)] + (mas[N * N - 1] - mas[N - 1]) / (N - 1);
    }

    for (int i = 0; i < N * N; i++) //�������
        anew[i] = mas[i];

    std::memcpy(anew, mas, totalSize * sizeof(double));//

    //double* deviceMatrixAPtr, *d_anew, *deviceError, *errorMatrix, *tempStorage = NULL;


      // �����������, ������� ������ ��������� ��������
    if (rank != 0 && rank != sizeOfTheGroup - 1)
    {
        sizeOfAreaForOneProcess += 2; // 2 ������� ������ �� ���������� ������
    }
    else
    {
        sizeOfAreaForOneProcess += 1; //otherwise
    }

    size_t sizeOfAllocatedMemory = N * sizeOfAreaForOneProcess;

    //�������� ������ �� gpu 
    cudaMalloc((void**)&d_mas, sizeOfAllocatedMemory * sizeof(double));
    cudaMalloc((void**)&d_anew, sizeOfAllocatedMemory * sizeof(double));
    cudaMalloc((void**)&errorMatrix, sizeOfAllocatedMemory * sizeof(double));
    cudaMalloc((void**)&deviceError, sizeof(double));

    // �������� ����� ����������� ������� � ���������� ������, ������� � 1 ������
    size_t offset = (rank != 0) ? N : 0; //��������� ��������� (���� ���� �� = 0, �� ����)

    cudaMemcpy(d_mas, mas + (startYIdx * N) - offset,
        sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice); //������ �� ������ �������

    cudaMemcpy(d_anew, anew + (startYIdx * N) - offset,
        sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice);


    // ����� �� �������� ������ ���������� ������ ��� �������� � �������� ������ ��� ����� ������
    size_t tempStorageSize = 0;

    cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, N * sizeOfAreaForOneProcess);

    cudaMalloc((void**)&tempStorage, tempStorageSize);

    int iter = 0;
    double* error;
    cudaMallocHost(&error, sizeof(double));//������ �� �����
    *error = 1.0;


    //���������� �������� ����� � ���-�� ������� 
    unsigned int threads_x = (N < 1024) ? N : 1024;// ���-�� �������, ���������� ���� 1024 �� ��������� (��-�� ����������) 
    unsigned int blocks_y = sizeOfAreaForOneProcess; // ���-�� ������ 
    unsigned int blocks_x = N / threads_x;

    dim3 blockDim1(threads_x, 1); // ������� ������ ����� � �������
    dim3 gridDim1(blocks_x, blocks_y); //������ ����� � ������

    cudaStream_t stream, matrixCalculationStream; //������ �����
    cudaStreamCreate(&stream);
    cudaStreamCreate(&matrixCalculationStream);

    clock_t begin = clock(); //������ ������� �������


    while (iter < ITER && (*error) > ACC) {
        iter += 1;

        // ����������� �������, ������� ����� ����� ���������� ������ ���������
        calculateBoundaries <<<N, 1, 0, stream >>> (d_mas, d_anew, N, sizeOfAreaForOneProcess);

        cudaStreamSynchronize(stream);

        // ������ �������
        calculationMatrix<<<gridDim, blockDim, 0, matrixCalculationStream >>> (d_mas, d_anew, N, sizeOfAreaForOneProcess);

        if (iter% 100 == 0) {
            getErrorMatrix <<<gridDim, blockDim, 0, matrixCalculationStream >>> (d_mas, d_anew, errorMatrix,
                N, sizeOfAreaForOneProcess);

            cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, sizeOfAllocatedMemory, matrixCalculationStream);

            // ������� ������������ ������ ����� ���� � ������� � ���� ���������
            MPI_Allreduce((void*)deviceError, (void*)deviceError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);//�� ���� ��������� ��������� ���� ������
            cudaMemcpyAsync(error, deviceError, sizeof(double), cudaMemcpyDeviceToHost, matrixCalculationStream);
        }


        if (rank != 0) //������� ������� �� � ��� ���������� :(
        {
            	MPI_Sendrecv(d_anew+ N + 1, N - 2, MPI_DOUBLE, rank - 1, 0,
                d_anew+ 1, N - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // ����� � ������ ��������
        if (rank != sizeOfTheGroup - 1)
        {
           	MPI_Sendrecv(d_anew+ (sizeOfAreaForOneProcess - 2) * N + 1, N - 2, MPI_DOUBLE, rank + 1, 0, d_anew+ (sizeOfAreaForOneProcess - 1) * N + 1, N - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        cudaStreamSynchronize(matrixCalculationStream);

        //����� ����������
        double* c = d_mas;
        d_mas= d_anew;
        d_anew = c;


    }
    clock_t end = clock();
    if (rank == 0) //�������� 1 ���
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

    //������� ��������� ��� MPI-�������� � ����������� ��� ������� �����
    MPI_Finalize();
    return 0;
}

