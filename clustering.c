/* @file: clustering.c
 * @details:
 * This algorithm reads from an input file a number of nodes
 * that each have a value represented by a given number of
 * bits. It will then use the union find concept to cluster 
 * items that have a hamming distance of less than 3. It 
 * will then present the number of clusters that are created
 * which guarantee the minimum hamming distance between any two 
 * clusters is at least 3.
 * 
 * @author: Charles Irick
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <list>

using namespace std;

// A structure to represent a subset for union-find
typedef struct
{
    int parent;
    int rank;
}cluster_t;
 
// A utility function to find set of an element i
// (uses path compression technique)
int find(cluster_t clusters[], int i)
{
    // find root and make root as parent of i (path compression)
    if (clusters[i].parent != i)
        clusters[i].parent = find(clusters, clusters[i].parent);
 
    return clusters[i].parent;
}
 
// A function that does union of two sets of x and y
// (uses union by rank)
void Union(cluster_t clusters[], int x, int y)
{
    int xroot = find(clusters, x);
    int yroot = find(clusters, y);
 
    // Attach smaller rank tree under root of high rank tree
    // (Union by Rank)
    if (clusters[xroot].rank < clusters[yroot].rank)
        clusters[xroot].parent = yroot;
    else if (clusters[xroot].rank > clusters[yroot].rank)
        clusters[yroot].parent = xroot;
 
    // If ranks are same, then make one as root and increment
    // its rank by one
    else
    {
        clusters[yroot].parent = xroot;
        clusters[xroot].rank++;
    }
}

void cluster_item(list<int>* hash_table, cluster_t *clusters, int cur_word, int cur_pos)
{
  list<int>::iterator it;

    if(!hash_table[cur_word].empty())
    {
      for(it=hash_table[cur_word].begin();it!=hash_table[cur_word].end();it++)
      {
        // find (i)
        int x = find(clusters,cur_pos);
        // find (*it)
        int y = find(clusters,*it);
        
        if(x!=y)
        {
          // union(i,*it)
          Union(clusters,cur_pos,*it);
        }
      }
    }
}
// Driver program to test above functions
int main(int argc, char*argv[])
{
  FILE *ifp;
  int i,j;
  int num_nodes = 0;
  int num_bits = 0;
  int word,bit;
  list<int> *hash_table;
  int * words;

  ifp = fopen(argv[1],"r");
  if(ifp == NULL)
  {
    printf("ERROR\n");
    return -1;
  }

  fscanf(ifp,"%d %d",&num_nodes,&num_bits);

  hash_table = new list<int>[(1<<num_bits)];
  cluster_t *clusters = new cluster_t[num_nodes];
  words = new int[num_nodes];

  /* Initialize clusters
     Each node is it's own cluster  */
  for (int v = 0; v < num_nodes; ++v)
  {
      clusters[v].parent = v;
      clusters[v].rank = 0;
  }
 
  /* Retrieve each word */
  for(i=0;i<num_nodes;i++)
  {
    word = 0;
    for(j=0;j<num_bits;j++)
    {
      fscanf(ifp,"%d",&bit);
      word = (word << 1) + bit;
    }
    /* Save words for iterative find/union */
    words[i] = word;
    /* Hash locations for union */
    hash_table[word].push_front(i);
  }

  // Iterate through all words, finding all possible
  // permutations with hamming distance 1 or 2 (less than 3)
  // then find if those permutations exist from the input, 
  // and if they do add them to your cluster
  for(i=0;i<num_nodes;i++)
  {
    int cur_word = words[i];
    int shift;
 
    // must union unpermuted word in case of duplicates
    cluster_item(hash_table,clusters,cur_word,i);

    // try all permutations of hamming distance < 3
    for(shift=0;shift<num_bits;shift++)
    {
      // permute combinations of cur_word
      int bitmask = 1<<shift | 1;
      int shifts = num_bits - shift;
      for(j=0;j<shifts;j++)
      {
        int temp = cur_word ^ bitmask;
   
        cluster_item(hash_table,clusters,temp,i); 
        
        bitmask <<= 1;
      }
    }
  }

  int num_clusters = 0;
  for(i=0;i<num_nodes;i++)
  {
    if(clusters[i].parent == i)
      num_clusters++;
  }

  printf("Num clusters %d\n",num_clusters);

  delete[] hash_table;
  delete[] clusters;
  delete[] words;

  return 0;
}

