#include<iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <openacc.h>
#include <cublas_v2.h>

int main(int arg, char** argv) {
    int N = std::atoi(argv[1]); //параметр, размер сетки
    int ITER = std::atoi(argv[2]); //параметр, макс количество итераций
    float ACC = std::atof(argv[3]); //параметр, мин значение ошибки
    //защита от пользователя
    if (N =< 0){
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


    cublasHandle_t handle; //контекст, позволяет управлять настройками библиотеки
    cublasStatus_t stat; //переменная для проверки работы функций библиотеки

    stat = cublasCreate(&handle); //создаём структуру, содержащую контекст библиотеки
    if (stat != CUBLAS_STATUS_SUCCESS) { //проверка статуса
  	std::cout<<"CUBLAS initialization failed handle"<<std::endl;
 	return EXIT_FAILURE;
    }

    double* mas = new double[N * N]; //создаем массив, который будет содержать старую копию  
    double* anew = new double[N * N]; //массив, который будет содержать новую версию 

    for (int i = 0; i < N * N; i++) //заполняю оба массива нулями, чтобы избавиться от возможного мусора
    {
        mas[i] = 0;
        anew[i] = 0;
    }

    int rep = 0; //инициализация переменной, отвечающей за отслеживания количества итераций 
    double err = 1.0; //инициализация переменной, отвечающей за отслеживания ошибки на определённой итерации

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
#pragma acc enter data copyin(anew[0:N*N], mas[0:N*N]) //копирование матирц из цпу в гпу
    {
        std::cout << "Initialization Time: " << 1.0 * (clock() - befin) / CLOCKS_PER_SEC << std::endl; //вывод потраченного на инициализацию времени
        clock_t befca = clock(); //начало отсчёта для выполнения вычислений
	double a = -1; //альфа 
	int index = 0; //индекс наибольшего элемента

        while (rep < ITER && err >= ACC) //начинаем вычислять матрицу
        {
	    #pragma acc data present(anew, mas)//заявляем, что массивы действительно есть на гпу
            #pragma acc parallel loop independent collapse(2) async () //распараллеливание
            for (int i = 1; i < N - 1; i++)
                for (int j = 1; j < N - 1; j++)
                    anew[i * N + j] = (mas[i * N + j - 1] + mas[(i - 1) * N + j] + mas[(i + 1) * N + j] + mas[i * N + j + 1]) *0.25; //высчитываем среднее для каждого элемента матрицы
	   if (rep % 100 == 0) //каждые 100 итераций обновляем значение ошибки
	    {
		#pragma acc wait () //ожидаем, пока закончат работу все распаралелленные части
		#pragma acc host_data use_device(anew,mas)//перед каждым вызовом функции, тк происходит взаимодействие между опенасс и библиотекой
		{
			stat = cublasDaxpy(handle, N * N, &a, anew, 1, mas, 1); //сумма массивов, но из-за альфа = -1 => разность
			if (stat != CUBLAS_STATUS_SUCCESS) {//проверка статуса
				std::cout<<"CUBLAS initialization failed1"<<std::endl;
				return EXIT_FAILURE;
			}
			stat =cublasIdamax(handle, N*N, mas, 1, &index);//нахождение максимального
			if (stat != CUBLAS_STATUS_SUCCESS) {
			std::cout<<"CUBLAS initialization failed2"<<std::endl;
				return EXIT_FAILURE;
			}
            	}
	   #pragma acc update host(mas[index - 1]) //индекс - 1, тк с нуля
	   err = fabs(mas[index - 1]); //значение ошибки, модуль на случай отрицательного числа

           #pragma acc host_data use_device(mas, anew)//сообщаем о использовании гпу
	   stat = cublasDcopy(handle, N * N, anew, 1, mas, 1); //копируем массив
	   if (stat != CUBLAS_STATUS_SUCCESS) { //проверяем статус функции 
		std::cout<<"CUBLAS initialization failed3"<<std::endl;
		return EXIT_FAILURE;
	   }
	   }
           double* c = mas; //создаём переменную для дальнейшего свопа матриц
           mas = anew;
           anew = c;
           rep++; //итерация 1 пройдена
           std::cout << rep << "  " << err << std::endl; //вывод итерации и значения ошибки  
        }
        std::cout << "Calculation time: " << 1.0 * (clock() - befca) / CLOCKS_PER_SEC << std::endl; //вывод времени, потраченного на вычисления матрицы
    }
    std::cout << "Iteration: " << rep << " " << "Error: " << err << std::endl; //сколько всего итераций потребовалось и достигнутое значение ошибки
    delete[] mas; //очищаем память
    delete[] anew;
    return 0; //программа завершена успешно
} 
