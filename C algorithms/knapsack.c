/* @file: knapsack.c
 * @details:
 * This an implementation of the 0-1 knapsack problem
 * using dynamic programming which operates in 
 * O(nW) time and O(W) space, where n is the number
 * of items, and W is the weight of the knapsack.
 * 
 * @author: Charles Irick
 */
#include<stdio.h>
#include<stdlib.h>

// A utility function that returns maximum of two integers
int max(int a, int b) { return (a > b)? a : b; }
 
// Returns the maximum value that can be put in a knapsack of capacity W
int knapSack(int W, int wt[], int val[], int n)
{
   int i, w, cur, left;
   int res;
   int **K;

   /* Dynamically allocate K to hold tabulation of results
      as we build up knapsack results. When iterating across
      items, we only ever need the previous items results, so
      we only need 2 rows. This reduces O(nW) space down to
      O(W) space. This optimization removes the ability to reconstruct
      which items we chose, but will still provide the maximum value. */
   K = (int**)malloc((2)*sizeof(int*));
   for(i=0;i<n+1;i++) 
   {
     /* Use calloc to initialize data to 0 */
     K[i] = (int*)calloc((W+1),sizeof(int));
   }
 
   /* Initialize item indexing values */
   cur = left = 0;

   /* Start from first item */
   for (i = 1; i <= n; i++)
   {
       left = cur;
       cur = i%2;
       for (w = 1; w <= W; w++)
       {
           if (wt[i-1] <= w)
                 K[cur][w] = max(val[i-1] + K[left][w-wt[i-1]],  K[left][w]);
           else
                 K[cur][w] = K[left][w];
       }
   }
 
   res = K[cur][W];

   /* Free K */
   for(i=0;i<n+1;i++)
     free(K[i]);
   free(K);

   return res;
}
 
int main(int argc, char *argv[])
{
  FILE *ifp;
  int W, n;
  int i;

  ifp = fopen(argv[1],"r");
  if(ifp == NULL)
  {
    printf("ERROR\n");
    return -1;
  }

  fscanf(ifp,"%d %d",&W, &n);

  int *val = (int*)malloc(n*(sizeof(int)));
  int *wt = (int*)malloc(n*(sizeof(int)));

  printf("Reading input\n");
  for(i=0;i<n;i++)
  {
    fscanf(ifp,"%d %d",&val[i],&wt[i]);
  }
  fclose(ifp);
    
  printf("Running knapsack\n");
  printf("%d\n", knapSack(W, wt, val, n));
    
  free(val);
  free(wt);

  return 0;
}
