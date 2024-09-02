# Dynamic Programming
## Single Data Dynamic Programming
* [Unique-Binary-Search-Trees](https://leetcode.cn/problems/unique-binary-search-trees/description/)
  * Combine Previous Results based on the root. A little like Convolution
## Multiple Data Dynamic Programming
* [Interleaving String](https://leetcode.cn/problems/interleaving-string/description/)
  * construct a matrix s, where $s[i][j]$ means whether **s1[0..i] and s2[0..j] can form s3[0..i+j]** and the condition holds
  * when $s[i][j]$ is determined by $s[i-1][j]$ and $s[i][j-1]$, then the matrix can be compressed to a list. Just need to carefully update the list.

