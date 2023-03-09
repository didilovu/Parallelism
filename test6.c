#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define pi 3.1415926535
#define N 10000000
int main()
{
	float* arr = (float*)malloc(sizeof(float) * N);
	float sum = 0;
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);
#pragma acc data create (arr[:N]) copy(sum)
	{
#pragma acc kernels
		for (size_t i = 0; i < N; i++)
		{
			arr[i] = sinf(2 * pi / N * i);
		}
#pragma acc kernels
		for (size_t i = 0; i < N; i++)
		{
			sum += arr[i];
		}
	}
	printf("%.30lf\n", sum);
	//printf("%f\n, timespec");
	free(arr);
	return 0;
}
