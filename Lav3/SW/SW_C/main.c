#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "scores.h"

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MATCH 2
#define MISMATCH -1

char indexof(char c){
    switch(c){
	case 'A': return 0;
	case 'R': return 1;
	case 'N': return 2;
	case 'D': return 3;
	case 'C': return 4;
	case 'Q': return 5;
	case 'E': return 6;
	case 'G': return 7;
	case 'H': return 8;
	case 'I': return 9;
	case 'L': return 10;
	case 'K': return 11;
	case 'M': return 12;
	case 'F': return 13;
	case 'P': return 14;
	case 'S': return 15;
	case 'T': return 16;
	case 'W': return 17;
	case 'Y': return 18;
	case 'V': return 19;
	case 'B': return 20;
	case 'Z': return 21;
	case 'X': return 22;
	default : return -1;
    }
}

int smith_waterman(char *q, const int q_len, char *db, const int db_len,
        char g_init, char g_ext) {

    char H[q_len+1][db_len+1];
    char E[q_len+1][db_len+1];
    char F[q_len+1][db_len+1];
    char prev[q_len+1][db_len+1][2];
    char q_out[q_len+db_len];
    char db_out[q_len+db_len];

    char max1,max2,idxi,idxj,deW;

    int i, j, k, l, m, n;
    int maxi,maxj;
    int maxval = 0;

    for (i = 0; i <= q_len; i++) {
        H[i][0] = 0;
        E[i][0] = 0;
        F[i][0] = 0;
        prev[i][0][1] = 0;
        if (i==0)prev[i][0][0] = 0;
        else prev[i][0][0] = -1;
    }

    for (j = 0; j <= db_len; j++) {
        H[0][j] = 0;
        E[0][j] = 0;
        F[0][j] = 0;
        prev[0][j][0] = 0;
        if (i==0)prev[0][j][1] = 0;
        else prev[0][j][1] = -1;
    }

    for (i = 1; i <= q_len; i++) {
        for (j = 1; j <= db_len; j++) {
            E[i][j] = MAX((E[i][j-1]-g_ext),(H[i][j-1]-g_init));
            F[i][j] = MAX((F[i-1][j]-g_ext),(H[i-1][j]-g_init));
	    idxi = indexof(*(q+i-1));
	    idxj = indexof(*(db+j-1));
	    if ((idxi == -1) || (idxj == -1)){
		deW = (strncmp(q+i-1, db+j-1, 1) == 0) ? MATCH : MISMATCH;
	    } else {
		deW = W[idxi][idxj];
	    }
	    
            max1 = MAX(0,E[i][j]);
            max2 = MAX(F[i][j],H[i-1][j-1]+deW);
            H[i][j] = MAX(max1,max2);
            if (H[i][j]>maxval) {
                maxval = H[i][j];
                maxi = i; maxj = j;
            } 
            if (H[i][j] == H[i-1][j-1]+deW) {prev[i][j][0] = -1; prev[i][j][1] = -1;}
            else if (H[i][j] == E[i][j])  {prev[i][j][0] = 0; prev[i][j][1] = -1; }
            else if (H[i][j] == F[i][j])  {prev[i][j][0] = -1; prev[i][j][1] = 0; }
            else {prev[i][j][0] = 0; prev[i][j][1] = 0;  }
        }
    }

	
    printf("Alignment Matrix\n");
    for (i = 1; i <= q_len; i++) {
        for (j = 1; j <= db_len; j++) {
            printf("%d\t",H[i][j]);
        }
        printf("\n");
    }
    printf("Max Score:%d\n",H[maxi][maxj]);

    i = maxi;
    j = maxj;
    k = 0;
    while(!( ((i==0) && (j==0)) || (H[i][j] == 0))) {
        if((prev[i][j][0] == -1) && (prev[i][j][1] == -1)){
            q_out[k] = *(q+i-1);
            db_out[k] = *(db+j-1);
        } else if ((prev[i][j][0] == 0) && (prev[i][j][1] == -1)){
            q_out[k] = 45;
            db_out[k] = *(db+j-1);
        } else if ((prev[i][j][0] == -1) && (prev[i][j][1] == 0)){
            q_out[k] = *(q+i-1);
            db_out[k] = 45;
        } else {
            q_out[k] = 45;
            db_out[k] = 45;
        }

        m = i + prev[i][j][0];
        n = j + prev[i][j][1];
        i = m;
        j = n;
        k++;
    }


    for (i=0; i<n-m; i++) printf(" ");
    for (i=0; i<m; i++) printf("%c",q[i]);
    for (i=k-1; i>=0; i--) printf("%c",q_out[i]);
    for (i=maxi; i<=q_len; i++) printf("%c",q[i]);
    printf("\n");
    for (i=0; i<n-m; i++) printf(" ");
    for (i=0; i<m; i++) printf(" ");
    for (i=k-1; i>=0; i--) {if (q_out[i] == db_out[i]) printf(":"); else printf(".");}
    printf("\n");
    for (i=0; i<m-n; i++) printf(" ");
    for (i=0; i<n; i++) printf("%c",db[i]);
    for (i=k-1; i>=0; i--) printf("%c",db_out[i]);
    for (i=maxj; i<=db_len; i++) printf("%c",db[i]);
    printf("\n");
    return 0;
}

int main(int argc, const char **argv) {

    if (argc != 3) {
        printf("usage: swalign SEQ1 SEQ2\n");
        exit(1);
    }

    char q[strlen(argv[1])], db[strlen(argv[2])];
    int q_len,db_len;
    int result;

    strcpy(q, argv[1]);
    strcpy(db, argv[2]);

    q_len = strlen(q);
    db_len = strlen(db);

    result = smith_waterman(q,q_len,db,db_len,12,2);

    //printf("%s\n%s\n", result->a, result->b);

    exit(0);
} 
