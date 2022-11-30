#include <immintrin.h>
void singleThread(int N, int *matA, int *matB, int *output)
{
  assert( N>=4 and N == ( N &~ (N-1)));
  union
  {
    __m256i sum_quad;
    int sum[8];
  };
  for(int rowA = 0; rowA < N; rowA +=2) { 
    for(int iter = 0; iter < N; iter+=8){ 
            int indexA1=rowA * N + (iter+0),indexA2=(rowA+1) * N + (iter+0);
            __m256i a1 =_mm256_set1_epi32 (matA[indexA1+0]+matA[indexA2+0]);
            __m256i a2 =_mm256_set1_epi32 (matA[indexA1+1]+matA[indexA2+1]);
            __m256i a3 =_mm256_set1_epi32 (matA[indexA1+2]+matA[indexA2+2]);
            __m256i a4 =_mm256_set1_epi32 (matA[indexA1+3]+matA[indexA2+3]);
            __m256i a5 =_mm256_set1_epi32 (matA[indexA1+4]+matA[indexA2+4]);
            __m256i a6 =_mm256_set1_epi32 (matA[indexA1+5]+matA[indexA2+5]);
            __m256i a7 =_mm256_set1_epi32 (matA[indexA1+6]+matA[indexA2+6]);
            __m256i a8 =_mm256_set1_epi32 (matA[indexA1+7]+matA[indexA2+7]);
            int rowC = rowA>>1,row_index=rowC * (N>>1);
            for(int colB = 0; colB < N; colB += 8){
                int indexB=(iter+0)*N+colB;
                __m256i b1= _mm256_loadu_si256((__m256i*)&matB[indexB]);
                __m256i b2= _mm256_loadu_si256((__m256i*)&matB[indexB+N]);
                __m256i b3= _mm256_loadu_si256((__m256i*)&matB[indexB+2*N]);
                __m256i b4= _mm256_loadu_si256((__m256i*)&matB[indexB+3*N]);
                __m256i b5= _mm256_loadu_si256((__m256i*)&matB[indexB+4*N]);
                __m256i b6= _mm256_loadu_si256((__m256i*)&matB[indexB+5*N]);
                __m256i b7= _mm256_loadu_si256((__m256i*)&matB[indexB+6*N]);
                __m256i b8= _mm256_loadu_si256((__m256i*)&matB[indexB+7*N]);
        //sum_quad=_mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(a1,b1),_mm256_mullo_epi32(a2,b1)),_mm256_add_epi32(_mm256_mullo_epi32(a3,b2),_mm256_mullo_epi32(a4,b2)));
                sum_quad=_mm256_add_epi32(
        	        _mm256_add_epi32(
        		        _mm256_add_epi32(
        			        _mm256_mullo_epi32(a1,b1),
        			        _mm256_mullo_epi32(a2,b2)),
        		        _mm256_add_epi32(
        			        _mm256_mullo_epi32(a3,b3),
        			        _mm256_mullo_epi32(a4,b4))
        		        ),
        	        _mm256_add_epi32(
        		        _mm256_add_epi32(
        			        _mm256_mullo_epi32(a5,b5),
        			        _mm256_mullo_epi32(a6,b6)),
        		        _mm256_add_epi32(
        			        _mm256_mullo_epi32(a7,b7),
        			        _mm256_mullo_epi32(a8,b8))
        		        )
        	        );
                int colC = colB>>1;
                int indexC = row_index + colC;
                output[indexC]+=sum[0]+sum[1];
                output[indexC+1]+=sum[2]+sum[3];
                output[indexC+2]+=sum[4]+sum[5];
                output[indexC+3]+=sum[6]+sum[7];
            }
        }
  }
}
