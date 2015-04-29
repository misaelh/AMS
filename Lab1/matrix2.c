#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*****************************************************
the following function generates a "size"-element vector
and a "size x size" matrix
 ****************************************************/
void matrix_vector_gen(int size, double *matrix, double *matrix2){
  int i;
  for(i=0; i<size*size; i++){
    matrix[i] = ((double)rand())/5307.0;
    matrix2[i] = ((double)rand())/5307.0;
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
  matrix_vector_gen(size, matrix, vector);

  double time_sq = 0;
  double time_pl = 0;
  printf("Size: %d\n",size);
  time_sq = omp_get_wtime();
  matrix_mult_sq(size, vector, matrix, result_sq);
  time_sq = omp_get_wtime() - time_sq;

  time_pl = omp_get_wtime();
  matrix_mult_pl(size, vector, matrix, result_pl);
  time_pl = omp_get_wtime() - time_pl;

  printf("SEQUENTIAL EXECUTION: %f (sec)\n", time_sq);
  printf("PARALLEL EXECUTION WITH %d (threads) ON %d (processors): %f (sec)\n",
	 omp_get_max_threads(), omp_get_num_procs(), time_pl);


  //check
  int i;
  for(i=0; i<size*size; i++)
    if(result_sq[i] != result_pl[i]){
      printf("wrong at position %d\n", i);
      return 0;
    }
  /*
  printf("Size: %d\n",size);
  printf("\nIn1");
  for(rows=0; rows<size; rows++){
    printf("\n");
      for(cols=0; cols<size; cols++)
      printf("%.1f ", vector[rows*size+cols]);
  }
  printf("\nIn2");
  for(rows=0; rows<size; rows++){
    printf("\n");
    for(cols=0; cols<size; cols++)
      printf("%.1f ", matrix[rows*size+cols]);
  }
  printf("\nout");
  for(rows=0; rows<size; rows++){
    printf("\n");
    for(cols=0; cols<size; cols++)
      printf("%.1f ", result_sq[rows*size+cols]);
  }
  */
  free(vector);
  free(matrix);
  free(result_sq);
  free(result_pl);
  return 1;
}
