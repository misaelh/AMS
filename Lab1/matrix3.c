#include <immintrin.h>
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
    matrix[i] = i;//((double)rand())/5307.0;
    matrix2[i] = i;//((double)rand())/5307.0;
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

void matrix_mult_avx(int size, double *vector_in,
		       double *matrix_in, double *vector_out){
  int rows, cols;
  int j;
  __m256d in1,in2,out;
  /*
# pragma omp parallel					\
  shared(size, vector_in, matrix_in, vector_out)	\
  private(rows, cols, j)
# pragma omp for
  */
  for(cols=0; cols<size; cols++)
    for(rows=0; rows<size; rows++){
      vector_out[rows*size + cols] = 0.0;
      out = _mm256_set1_pd(0.0);
      for(j=0; j<size; j+=4){
	in1 = _mm256_loadu_pd(&vector_in[rows*size+j]);
        in2 = _mm256_set1_pd(matrix_in[cols+j*size]);

	out = _mm256_add_pd(_mm256_mul_pd(in1, in2), out);

      }
      _mm256_storeu_pd(&vector_out[rows*size + cols], out); 
    }
}

void matrix_mult_avx2(int size, double *vector_in,
		       double *matrix_in, double *vector_out){
  int rows, cols;
  int j;
  __m256d in1,in2,out;
  /*
# pragma omp parallel					\
  shared(size, vector_in, matrix_in, vector_out)	\
  private(rows, cols, j)
# pragma omp for
  */
  for(cols=0; cols<size; cols++)
    for(rows=0; rows<size; rows++){
      vector_out[rows*size + cols] = 0.0;
      out = _mm256_set1_pd(0.0);
      for(j=0; j<size; j+=4){
	in1 = _mm256_loadu_pd(&vector_in[rows*size+j]);
        in2 = _mm256_set_pd(matrix_in[cols+j*size],matrix_in[cols+(j+1)*size],matrix_in[cols+(j+2)*size],matrix_in[cols+(j+3)*size]);

	out = _mm256_add_pd(_mm256_mul_pd(in1, in2), out);

      }
      _mm256_storeu_pd(&vector_out[rows*size + cols], out); 
    }
}

void mmul_sse(const double * a, const double * b, double * r)
{
  int i,j;
  __m256d a_line, b_line, r_line;
  for (i=0; i<16; i+=4) {
    // unroll the first step of the loop to avoid having to initialize r_line to zero
    a_line = _mm256_loadu_pd(a);         // a_line = vec4(column(a, 0))
    b_line = _mm256_set1_pd(b[i]);      // b_line = vec4(b[i][0])
    r_line = _mm256_mul_pd(a_line, b_line); // r_line = a_line * b_line
    for (j=1; j<4; j++) {
      a_line = _mm256_loadu_pd(&a[j*4]); // a_line = vec4(column(a, j))
      b_line = _mm256_set1_pd(b[i+j]);  // b_line = vec4(b[i][j])
                                     // r_line += a_line * b_line
      r_line = _mm256_add_pd(_mm256_mul_pd(a_line, b_line), r_line);
    }
    _mm256_storeu_pd(&r[i], r_line);     // r[i] = r_line
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
  double time_sq = 0;
  double time_pl = 0;
  matrix_vector_gen(size, matrix, vector, result_sq);
  
  matrix_mult_avx(size, vector, matrix, result_sq);

  /*  
  time_sq = omp_get_wtime();
  matrix_mult_sq(size, vector, matrix, result_sq);
  time_sq = omp_get_wtime() - time_sq;
  */
  time_pl = omp_get_wtime();
  matrix_mult_pl(size, vector, matrix, result_pl);
  time_pl = omp_get_wtime() - time_pl;

  //  printf("SEQUENTIAL EXECUTION: %f (sec)\n", time_sq);
  printf("PARALLEL EXECUTION WITH %d (threads) ON %d (processors): %f (sec)\n",
	 omp_get_max_threads(), omp_get_num_procs(), time_pl);

  
  //check
  int i;
  for(i=0; i<size*size; i++)
    if(result_sq[i] != result_pl[i]){
      printf("MP: %f AV: %f\n", result_pl[i], result_sq[i+3]);
      printf("wrong at position %d\n", i);
      return 0;
      }
  
  printf("\nout");
  for(rows=0; rows<1; rows++){
    printf("\n");
    for(cols=0; cols<1; cols++)
      printf("%.1f ", result_sq[rows*size+cols]);
  }


  free(vector);
  free(matrix);
  free(result_sq);
  free(result_pl);
  return 1;
}
