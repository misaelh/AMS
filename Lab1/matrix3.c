#include <immintrin.h>
#include <intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*****************************************************
the following function generates a "size"-element vector
and a "size x size" matrix
 ****************************************************/
void matrix_vector_gen(int size, double *matrix, double *matrix2, double *result_sq){
  int i;
  for(i=0; i<size*size; i++){
    matrix[i] = ((double)rand())/5307.0;
    matrix2[i] = ((double)rand())/5307.0;
    result_sq[i] = 0.0;
  }
}

/****************************************************
the following function calculate the below equation
   vector_out = vector_in x matrix_in
 ***************************************************/
void matrix_mult_sq(int size, double *vector_in,
		       double *matrix_in, double *vector_out){
  int rows, cols;
  int j;

	for(rows=0; rows<size; rows++)
	  for(cols=0; cols<size; cols++){
      vector_out[rows*size + cols] = 0.0;
      for(j=0; j<size; j++)
	vector_out[rows*size + cols] += vector_in[rows*size + j] * matrix_in[j*size + cols];
    }
}

void matrix_mult_pl(int size, double *vector_in,
		       double *matrix_in, double *vector_out){
  int rows, cols;
  int j;
# pragma omp parallel				\
  shared(size, vector_in, matrix_in, vector_out)	\
  private(rows, cols, j)
# pragma omp for
  for(cols=0; cols<size; cols++)
    for(rows=0; rows<size; rows++){
      vector_out[rows*size + cols] = 0.0;
      for(j=0; j<size; j++)
	vector_out[rows*size + cols] += vector_in[rows*size + j] * matrix_in[j*size + cols];
    }
}

void matrix_mult_avx2(int size, double *vector_in,
		       double *matrix_in, double *vector_out){
  int rows, cols;
  int j;
  __m256d in1,in2,out;
  double result[4];
  
# pragma omp parallel					\
  shared(size, vector_in, matrix_in, vector_out)	\
  private(rows, cols, j, in1, in2, out, result)
# pragma omp for
  
  for(cols=0; cols<size; cols++)
    for(rows=0; rows<size; rows++){
      vector_out[rows*size + cols] = 0.0;
      out = _mm256_set1_pd(0.0);
      for(j=0; j<size; j+=4){
	in1 = _mm256_loadu_pd(&vector_in[rows*size+j]);
        in2 = _mm256_set_pd(matrix_in[cols+(j+3)*size],matrix_in[cols+(j+2)*size],matrix_in[cols+(j+1)*size],matrix_in[cols+(j)*size]);

	out = _mm256_fmadd_pd(in1, in2, out);

      }
      _mm256_storeu_pd(result, out);
      vector_out[rows*size + cols] = result[0]+result[1]+result[2]+result[3];
    }
}

void mmul_sse(const double * a, const double * b, double * r)
{
  int i,j;
  __m256d a_line, b_line, r_line;
  for (i=0; i<16; i+=4) {
    // unroll the first step of the loop to avoid having to initialize r_line to zero
    a_line = _mm256_loadu_pd(a);
    b_line = _mm256_set1_pd(b[i]);
    r_line = _mm256_mul_pd(a_line, b_line); 
    for (j=1; j<4; j++) {
      a_line = _mm256_loadu_pd(&a[j*4]); 
      b_line = _mm256_set1_pd(b[i+j]);  
                                
      r_line = _mm256_add_pd(_mm256_mul_pd(a_line, b_line), r_line);
    }
    _mm256_storeu_pd(&r[i], r_line);
  }
}

int main(int argc, char *argv[]){
  int rows, cols;
  int j;
  if(argc < 2){
    printf("Usage: %s matrix/vector_size\n", argv[0]);
    return 0;
  }

  int size = atoi(argv[1]);
  double *vector = (double *)malloc(sizeof(double)*size*size);
  double *matrix = (double *)malloc(sizeof(double)*size*size);
  double *result_sq = (double *)malloc(sizeof(double)*size*size);
  double *result_pl = (double *)malloc(sizeof(double)*size*size);
  double *result_avx = (double *)malloc(sizeof(double)*size*size);
  double time_sq = 0, time_pl = 0, time_avx = 0;
  
  matrix_vector_gen(size, matrix, vector, result_sq);
    
  time_sq = omp_get_wtime();
  matrix_mult_sq(size, vector, matrix, result_sq);
  time_sq = omp_get_wtime() - time_sq;
  
  time_pl = omp_get_wtime();
  matrix_mult_pl(size, vector, matrix, result_pl);
  time_pl = omp_get_wtime() - time_pl;
  
  time_avx = omp_get_wtime();
  matrix_mult_avx2(size, vector, matrix, result_avx);
  time_avx = omp_get_wtime() - time_avx;
  
  printf("AVX + MULTI-THREADED EXECUTION: %f (sec)\n", time_avx); // 8 640 - 0.445056
  printf("PARALLEL EXECUTION WITH %d (threads) ON %d (processors): %f (sec)\n",
	 omp_get_max_threads(), omp_get_num_procs(), time_pl); // 8 640 - 0.618194
  printf("SEQUENTIAL EXECUTION : %f (sec)\n", time_sq); // 8 640 - 0.618194

  //check
  int i;
  for(i=0; i<size*size; i++)
    if(result_avx[i] != result_sq[i]){
      printf("SEQ: %f AV: %f\n", result_sq[i], result_avx[i]);
      printf("wrong at position %d\n", i);
      return 0;
      }

  printf("\nDone");
  free(vector);
  free(matrix);
  free(result_sq);
  free(result_pl);
  free(result_avx);
  return 1;
}
