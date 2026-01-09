# Dynamic Programming Cheat Sheet - Java

## Table of Contents
1. [Core DP Concepts](#core-dp-concepts)
2. [Pattern 1: Fibonacci/Climbing Stairs](#pattern-1-fibonacciclimbing-stairs)
3. [Pattern 2: 0/1 Knapsack](#pattern-2-01-knapsack)
4. [Pattern 3: Unbounded Knapsack](#pattern-3-unbounded-knapsack)
5. [Pattern 4: Longest Common Subsequence (LCS)](#pattern-4-longest-common-subsequence)
6. [Pattern 5: Longest Increasing Subsequence (LIS)](#pattern-5-longest-increasing-subsequence)
7. [Pattern 6: Palindromic Subsequence](#pattern-6-palindromic-subsequence)
8. [Pattern 7: Matrix Chain Multiplication](#pattern-7-matrix-chain-multiplication)
9. [Pattern 8: DP on Strings](#pattern-8-dp-on-strings)
10. [Pattern 9: DP on Trees](#pattern-9-dp-on-trees)
11. [Pattern 10: DP on Grids](#pattern-10-dp-on-grids)
12. [Backtracking Patterns](#backtracking-patterns)
13. [Optimization Techniques](#optimization-techniques)

---

## Core DP Concepts

### When to Use DP?
1. **Optimal Substructure**: Problem can be broken into smaller subproblems
2. **Overlapping Subproblems**: Same subproblems are solved multiple times
3. **Look for**: "Maximum/Minimum", "Count ways", "Yes/No decision"

### DP Approaches
```java
// 1. Memoization (Top-Down)
class Solution {
    Map<String, Integer> memo = new HashMap<>();
    
    public int solve(int n) {
        if (memo.containsKey(key)) return memo.get(key);
        // base case
        // recursive calls
        memo.put(key, result);
        return result;
    }
}

// 2. Tabulation (Bottom-Up)
class Solution {
    public int solve(int n) {
        int[] dp = new int[n + 1];
        dp[0] = baseCase;
        for (int i = 1; i <= n; i++) {
            dp[i] = // computation based on previous values
        }
        return dp[n];
    }
}

// 3. Space Optimized
class Solution {
    public int solve(int n) {
        int prev = baseCase, curr = 0;
        for (int i = 1; i <= n; i++) {
            curr = // computation using prev
            prev = curr;
        }
        return curr;
    }
}
```

---

## Pattern 1: Fibonacci/Climbing Stairs

### Template
```java
// Classic Fibonacci
public int fib(int n) {
    if (n <= 1) return n;
    int prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}

// Climbing Stairs (Can take 1 or 2 steps)
public int climbStairs(int n) {
    if (n <= 2) return n;
    int prev2 = 1, prev1 = 2;
    for (int i = 3; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}

// Min Cost Climbing Stairs
public int minCostClimbingStairs(int[] cost) {
    int n = cost.length;
    int prev2 = 0, prev1 = 0;
    for (int i = 2; i <= n; i++) {
        int curr = Math.min(prev1 + cost[i-1], prev2 + cost[i-2]);
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}

// House Robber
public int rob(int[] nums) {
    int n = nums.length;
    if (n == 1) return nums[0];
    int prev2 = nums[0], prev1 = Math.max(nums[0], nums[1]);
    for (int i = 2; i < n; i++) {
        int curr = Math.max(prev1, prev2 + nums[i]);
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

---

## Pattern 2: 0/1 Knapsack

### Core Template
```java
// Recursive with Memoization
public int knapsack(int[] weights, int[] values, int capacity) {
    int n = weights.length;
    Integer[][] dp = new Integer[n][capacity + 1];
    return knapsackHelper(weights, values, capacity, 0, dp);
}

private int knapsackHelper(int[] wt, int[] val, int cap, int idx, Integer[][] dp) {
    if (idx >= wt.length || cap == 0) return 0;
    if (dp[idx][cap] != null) return dp[idx][cap];
    
    // Don't include current item
    int exclude = knapsackHelper(wt, val, cap, idx + 1, dp);
    
    // Include current item (if possible)
    int include = 0;
    if (wt[idx] <= cap) {
        include = val[idx] + knapsackHelper(wt, val, cap - wt[idx], idx + 1, dp);
    }
    
    dp[idx][cap] = Math.max(include, exclude);
    return dp[idx][cap];
}

// Tabulation (Bottom-Up)
public int knapsackTabulation(int[] weights, int[] values, int capacity) {
    int n = weights.length;
    int[][] dp = new int[n + 1][capacity + 1];
    
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            if (weights[i-1] <= w) {
                dp[i][w] = Math.max(
                    values[i-1] + dp[i-1][w - weights[i-1]], // include
                    dp[i-1][w]                                // exclude
                );
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    return dp[n][capacity];
}

// Space Optimized (1D array)
public int knapsackOptimized(int[] weights, int[] values, int capacity) {
    int[] dp = new int[capacity + 1];
    
    for (int i = 0; i < weights.length; i++) {
        for (int w = capacity; w >= weights[i]; w--) {
            dp[w] = Math.max(dp[w], values[i] + dp[w - weights[i]]);
        }
    }
    return dp[capacity];
}
```

### Variations

#### Subset Sum
```java
public boolean canPartition(int[] nums, int target) {
    boolean[] dp = new boolean[target + 1];
    dp[0] = true;
    
    for (int num : nums) {
        for (int j = target; j >= num; j--) {
            dp[j] = dp[j] || dp[j - num];
        }
    }
    return dp[target];
}
```

#### Equal Sum Partition
```java
public boolean canPartition(int[] nums) {
    int sum = 0;
    for (int num : nums) sum += num;
    if (sum % 2 != 0) return false;
    
    return canPartition(nums, sum / 2);
}
```

#### Count of Subset Sum
```java
public int countSubsets(int[] nums, int target) {
    int[] dp = new int[target + 1];
    dp[0] = 1; // Empty subset
    
    for (int num : nums) {
        for (int j = target; j >= num; j--) {
            dp[j] += dp[j - num];
        }
    }
    return dp[target];
}
```

#### Minimum Subset Sum Difference
```java
public int minimumDifference(int[] nums) {
    int sum = 0;
    for (int num : nums) sum += num;
    
    boolean[] dp = new boolean[sum + 1];
    dp[0] = true;
    
    for (int num : nums) {
        for (int j = sum; j >= num; j--) {
            dp[j] = dp[j] || dp[j - num];
        }
    }
    
    int minDiff = Integer.MAX_VALUE;
    for (int i = 0; i <= sum / 2; i++) {
        if (dp[i]) {
            minDiff = Math.min(minDiff, sum - 2 * i);
        }
    }
    return minDiff;
}
```

#### Target Sum (+ and -)
```java
public int findTargetSumWays(int[] nums, int target) {
    int sum = 0;
    for (int num : nums) sum += num;
    
    if ((target + sum) % 2 != 0 || Math.abs(target) > sum) return 0;
    
    int subsetSum = (target + sum) / 2;
    return countSubsets(nums, subsetSum);
}
```

---

## Pattern 3: Unbounded Knapsack

### Core Template
```java
// Unbounded Knapsack (Items can be used unlimited times)
public int unboundedKnapsack(int[] weights, int[] values, int capacity) {
    int[] dp = new int[capacity + 1];
    
    for (int i = 0; i < weights.length; i++) {
        for (int w = weights[i]; w <= capacity; w++) {
            dp[w] = Math.max(dp[w], values[i] + dp[w - weights[i]]);
        }
    }
    return dp[capacity];
}
```

### Variations

#### Coin Change (Minimum coins)
```java
public int coinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, amount + 1);
    dp[0] = 0;
    
    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] = Math.min(dp[i], 1 + dp[i - coin]);
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}
```

#### Coin Change 2 (Count combinations)
```java
public int change(int amount, int[] coins) {
    int[] dp = new int[amount + 1];
    dp[0] = 1;
    
    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] += dp[i - coin];
        }
    }
    return dp[amount];
}
```

#### Rod Cutting Problem
```java
public int cutRod(int[] prices, int n) {
    int[] dp = new int[n + 1];
    
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= i; j++) {
            dp[i] = Math.max(dp[i], prices[j-1] + dp[i - j]);
        }
    }
    return dp[n];
}
```

---

## Pattern 4: Longest Common Subsequence

### Core Template
```java
// LCS - Tabulation
public int longestCommonSubsequence(String text1, String text2) {
    int m = text1.length(), n = text2.length();
    int[][] dp = new int[m + 1][n + 1];
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1.charAt(i-1) == text2.charAt(j-1)) {
                dp[i][j] = 1 + dp[i-1][j-1];
            } else {
                dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}

// Space Optimized
public int longestCommonSubsequenceOptimized(String text1, String text2) {
    int m = text1.length(), n = text2.length();
    int[] prev = new int[n + 1];
    int[] curr = new int[n + 1];
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1.charAt(i-1) == text2.charAt(j-1)) {
                curr[j] = 1 + prev[j-1];
            } else {
                curr[j] = Math.max(prev[j], curr[j-1]);
            }
        }
        int[] temp = prev;
        prev = curr;
        curr = temp;
    }
    return prev[n];
}
```

### Variations

#### Longest Common Substring
```java
public int longestCommonSubstring(String text1, String text2) {
    int m = text1.length(), n = text2.length();
    int[][] dp = new int[m + 1][n + 1];
    int maxLen = 0;
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1.charAt(i-1) == text2.charAt(j-1)) {
                dp[i][j] = 1 + dp[i-1][j-1];
                maxLen = Math.max(maxLen, dp[i][j]);
            }
        }
    }
    return maxLen;
}
```

#### Print LCS
```java
public String printLCS(String text1, String text2) {
    int m = text1.length(), n = text2.length();
    int[][] dp = new int[m + 1][n + 1];
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1.charAt(i-1) == text2.charAt(j-1)) {
                dp[i][j] = 1 + dp[i-1][j-1];
            } else {
                dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    
    StringBuilder lcs = new StringBuilder();
    int i = m, j = n;
    while (i > 0 && j > 0) {
        if (text1.charAt(i-1) == text2.charAt(j-1)) {
            lcs.append(text1.charAt(i-1));
            i--; j--;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            i--;
        } else {
            j--;
        }
    }
    return lcs.reverse().toString();
}
```

#### Shortest Common Supersequence
```java
public String shortestCommonSupersequence(String str1, String str2) {
    int m = str1.length(), n = str2.length();
    int[][] dp = new int[m + 1][n + 1];
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (str1.charAt(i-1) == str2.charAt(j-1)) {
                dp[i][j] = 1 + dp[i-1][j-1];
            } else {
                dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    
    StringBuilder result = new StringBuilder();
    int i = m, j = n;
    while (i > 0 && j > 0) {
        if (str1.charAt(i-1) == str2.charAt(j-1)) {
            result.append(str1.charAt(i-1));
            i--; j--;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            result.append(str1.charAt(i-1));
            i--;
        } else {
            result.append(str2.charAt(j-1));
            j--;
        }
    }
    
    while (i > 0) result.append(str1.charAt(--i));
    while (j > 0) result.append(str2.charAt(--j));
    
    return result.reverse().toString();
}
```

#### Minimum Insertions/Deletions to Convert String
```java
public int minDistance(String word1, String word2) {
    int lcs = longestCommonSubsequence(word1, word2);
    int deletions = word1.length() - lcs;
    int insertions = word2.length() - lcs;
    return deletions + insertions;
}
```

#### Edit Distance (Levenshtein Distance)
```java
public int minDistance(String word1, String word2) {
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m + 1][n + 1];
    
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1.charAt(i-1) == word2.charAt(j-1)) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + Math.min(dp[i-1][j-1], // replace
                              Math.min(dp[i-1][j],    // delete
                                       dp[i][j-1]));  // insert
            }
        }
    }
    return dp[m][n];
}
```

---

## Pattern 5: Longest Increasing Subsequence

### Core Templates
```java
// O(n²) DP Solution
public int lengthOfLIS(int[] nums) {
    int n = nums.length;
    int[] dp = new int[n];
    Arrays.fill(dp, 1);
    int maxLen = 1;
    
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = Math.max(dp[i], dp[j] + 1);
            }
        }
        maxLen = Math.max(maxLen, dp[i]);
    }
    return maxLen;
}

// O(n log n) Binary Search Solution
public int lengthOfLISOptimized(int[] nums) {
    List<Integer> tails = new ArrayList<>();
    
    for (int num : nums) {
        int pos = Collections.binarySearch(tails, num);
        if (pos < 0) pos = -(pos + 1);
        
        if (pos == tails.size()) {
            tails.add(num);
        } else {
            tails.set(pos, num);
        }
    }
    return tails.size();
}

// Print LIS
public List<Integer> printLIS(int[] nums) {
    int n = nums.length;
    int[] dp = new int[n];
    int[] parent = new int[n];
    Arrays.fill(dp, 1);
    Arrays.fill(parent, -1);
    
    int maxLen = 1, maxIdx = 0;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j] && dp[i] < dp[j] + 1) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
        if (dp[i] > maxLen) {
            maxLen = dp[i];
            maxIdx = i;
        }
    }
    
    List<Integer> lis = new ArrayList<>();
    while (maxIdx != -1) {
        lis.add(nums[maxIdx]);
        maxIdx = parent[maxIdx];
    }
    Collections.reverse(lis);
    return lis;
}
```

### Variations

#### Longest Bitonic Subsequence
```java
public int longestBitonicSequence(int[] nums) {
    int n = nums.length;
    int[] lis = new int[n]; // LIS ending at i
    int[] lds = new int[n]; // LDS starting at i
    Arrays.fill(lis, 1);
    Arrays.fill(lds, 1);
    
    // Calculate LIS for each position
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                lis[i] = Math.max(lis[i], lis[j] + 1);
            }
        }
    }
    
    // Calculate LDS for each position
    for (int i = n - 2; i >= 0; i--) {
        for (int j = n - 1; j > i; j--) {
            if (nums[i] > nums[j]) {
                lds[i] = Math.max(lds[i], lds[j] + 1);
            }
        }
    }
    
    int maxLen = 0;
    for (int i = 0; i < n; i++) {
        maxLen = Math.max(maxLen, lis[i] + lds[i] - 1);
    }
    return maxLen;
}
```

#### Maximum Sum Increasing Subsequence
```java
public int maxSumIS(int[] nums) {
    int n = nums.length;
    int[] dp = new int[n];
    for (int i = 0; i < n; i++) dp[i] = nums[i];
    
    int maxSum = nums[0];
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = Math.max(dp[i], dp[j] + nums[i]);
            }
        }
        maxSum = Math.max(maxSum, dp[i]);
    }
    return maxSum;
}
```

#### Number of LIS
```java
public int findNumberOfLIS(int[] nums) {
    int n = nums.length;
    int[] length = new int[n];
    int[] count = new int[n];
    Arrays.fill(length, 1);
    Arrays.fill(count, 1);
    
    int maxLen = 1;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                if (length[j] + 1 > length[i]) {
                    length[i] = length[j] + 1;
                    count[i] = count[j];
                } else if (length[j] + 1 == length[i]) {
                    count[i] += count[j];
                }
            }
        }
        maxLen = Math.max(maxLen, length[i]);
    }
    
    int result = 0;
    for (int i = 0; i < n; i++) {
        if (length[i] == maxLen) {
            result += count[i];
        }
    }
    return result;
}
```

---

## Pattern 6: Palindromic Subsequence

### Core Templates
```java
// Longest Palindromic Subsequence
public int longestPalindromeSubseq(String s) {
    int n = s.length();
    int[][] dp = new int[n][n];
    
    // Every single character is a palindrome of length 1
    for (int i = 0; i < n; i++) dp[i][i] = 1;
    
    // Check for length 2 and more
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i < n - len + 1; i++) {
            int j = i + len - 1;
            if (s.charAt(i) == s.charAt(j)) {
                dp[i][j] = 2 + (len == 2 ? 0 : dp[i+1][j-1]);
            } else {
                dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);
            }
        }
    }
    return dp[0][n-1];
}

// Longest Palindromic Substring
public String longestPalindrome(String s) {
    int n = s.length();
    boolean[][] dp = new boolean[n][n];
    int maxLen = 1, start = 0;
    
    // All substrings of length 1 are palindromes
    for (int i = 0; i < n; i++) dp[i][i] = true;
    
    // Check for length 2
    for (int i = 0; i < n - 1; i++) {
        if (s.charAt(i) == s.charAt(i + 1)) {
            dp[i][i + 1] = true;
            start = i;
            maxLen = 2;
        }
    }
    
    // Check for lengths > 2
    for (int len = 3; len <= n; len++) {
        for (int i = 0; i < n - len + 1; i++) {
            int j = i + len - 1;
            if (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]) {
                dp[i][j] = true;
                start = i;
                maxLen = len;
            }
        }
    }
    return s.substring(start, start + maxLen);
}

// Expand Around Center (Space Optimized)
public String longestPalindromeOptimized(String s) {
    if (s == null || s.length() < 1) return "";
    int start = 0, end = 0;
    
    for (int i = 0; i < s.length(); i++) {
        int len1 = expandAroundCenter(s, i, i);     // odd length
        int len2 = expandAroundCenter(s, i, i + 1); // even length
        int len = Math.max(len1, len2);
        
        if (len > end - start) {
            start = i - (len - 1) / 2;
            end = i + len / 2;
        }
    }
    return s.substring(start, end + 1);
}

private int expandAroundCenter(String s, int left, int right) {
    while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
        left--;
        right++;
    }
    return right - left - 1;
}
```

### Variations

#### Count Palindromic Substrings
```java
public int countSubstrings(String s) {
    int n = s.length();
    boolean[][] dp = new boolean[n][n];
    int count = 0;
    
    for (int i = 0; i < n; i++) {
        dp[i][i] = true;
        count++;
    }
    
    for (int i = 0; i < n - 1; i++) {
        if (s.charAt(i) == s.charAt(i + 1)) {
            dp[i][i + 1] = true;
            count++;
        }
    }
    
    for (int len = 3; len <= n; len++) {
        for (int i = 0; i < n - len + 1; i++) {
            int j = i + len - 1;
            if (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]) {
                dp[i][j] = true;
                count++;
            }
        }
    }
    return count;
}
```

#### Minimum Insertions to Make Palindrome
```java
public int minInsertions(String s) {
    int n = s.length();
    int lps = longestPalindromeSubseq(s);
    return n - lps;
}
```

#### Palindrome Partitioning II (Min Cuts)
```java
public int minCut(String s) {
    int n = s.length();
    boolean[][] isPalin = new boolean[n][n];
    int[] cuts = new int[n];
    
    // Build palindrome table
    for (int len = 1; len <= n; len++) {
        for (int i = 0; i < n - len + 1; i++) {
            int j = i + len - 1;
            if (s.charAt(i) == s.charAt(j)) {
                isPalin[i][j] = (len <= 2) || isPalin[i + 1][j - 1];
            }
        }
    }
    
    // Calculate minimum cuts
    for (int i = 0; i < n; i++) {
        if (isPalin[0][i]) {
            cuts[i] = 0;
        } else {
            cuts[i] = Integer.MAX_VALUE;
            for (int j = 0; j < i; j++) {
                if (isPalin[j + 1][i]) {
                    cuts[i] = Math.min(cuts[i], cuts[j] + 1);
                }
            }
        }
    }
    return cuts[n - 1];
}
```

---

## Pattern 7: Matrix Chain Multiplication

### Core Template
```java
// Matrix Chain Multiplication
public int matrixChainOrder(int[] dims) {
    int n = dims.length - 1;
    int[][] dp = new int[n][n];
    
    // Cost is 0 for single matrix
    for (int i = 0; i < n; i++) dp[i][i] = 0;
    
    // len is chain length
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i < n - len + 1; i++) {
            int j = i + len - 1;
            dp[i][j] = Integer.MAX_VALUE;
            
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] + 
                          dims[i] * dims[k+1] * dims[j+1];
                dp[i][j] = Math.min(dp[i][j], cost);
            }
        }
    }
    return dp[0][n-1];
}

// Generic MCM Template (Memoization)
public int solve(int[] arr, int i, int j, Integer[][] dp) {
    // Base case
    if (i >= j) return 0;
    
    if (dp[i][j] != null) return dp[i][j];
    
    int min = Integer.MAX_VALUE;
    for (int k = i; k < j; k++) {
        int temp = solve(arr, i, k, dp) + 
                   solve(arr, k+1, j, dp) + 
                   cost(arr, i, k, j);
        min = Math.min(min, temp);
    }
    
    dp[i][j] = min;
    return min;
}
```

### Variations

#### Burst Balloons
```java
public int maxCoins(int[] nums) {
    int n = nums.length;
    int[] arr = new int[n + 2];
    arr[0] = arr[n + 1] = 1;
    for (int i = 0; i < n; i++) arr[i + 1] = nums[i];
    
    int[][] dp = new int[n + 2][n + 2];
    
    for (int len = 1; len <= n; len++) {
        for (int left = 1; left <= n - len + 1; left++) {
            int right = left + len - 1;
            for (int i = left; i <= right; i++) {
                int coins = arr[left-1] * arr[i] * arr[right+1];
                coins += dp[left][i-1] + dp[i+1][right];
                dp[left][right] = Math.max(dp[left][right], coins);
            }
        }
    }
    return dp[1][n];
}
```

#### Boolean Parenthesization
```java
public int countWays(String s) {
    int n = s.length();
    int[][][] dp = new int[n][n][2]; // [i][j][0=false, 1=true]
    
    for (int i = 0; i < n; i += 2) {
        dp[i][i][s.charAt(i) == 'T' ? 1 : 0] = 1;
    }
    
    for (int len = 3; len <= n; len += 2) {
        for (int i = 0; i < n - len + 1; i += 2) {
            int j = i + len - 1;
            for (int k = i + 1; k < j; k += 2) {
                char op = s.charAt(k);
                int lt = dp[i][k-1][1], lf = dp[i][k-1][0];
                int rt = dp[k+1][j][1], rf = dp[k+1][j][0];
                
                if (op == '&') {
                    dp[i][j][1] += lt * rt;
                    dp[i][j][0] += lt * rf + lf * rt + lf * rf;
                } else if (op == '|') {
                    dp[i][j][1] += lt * rt + lt * rf + lf * rt;
                    dp[i][j][0] += lf * rf;
                } else { // XOR
                    dp[i][j][1] += lt * rf + lf * rt;
                    dp[i][j][0] += lt * rt + lf * rf;
                }
            }
        }
    }
    return dp[0][n-1][1];
}
```

#### Scramble String
```java
public boolean isScramble(String s1, String s2) {
    if (s1.equals(s2)) return true;
    if (s1.length() != s2.length()) return false;
    
    int n = s1.length();
    boolean[][][] dp = new boolean[n][n][n + 1];
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dp[i][j][1] = s1.charAt(i) == s2.charAt(j);
        }
    }
    
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i <= n - len; i++) {
            for (int j = 0; j <= n - len; j++) {
                for (int k = 1; k < len; k++) {
                    if ((dp[i][j][k] && dp[i+k][j+k][len-k]) ||
                        (dp[i][j+len-k][k] && dp[i+k][j][len-k])) {
                        dp[i][j][len] = true;
                        break;
                    }
                }
            }
        }
    }
    return dp[0][0][n];
}
```

---

## Pattern 8: DP on Strings

### String Matching Problems
```java
// Wildcard Pattern Matching
public boolean isMatch(String s, String p) {
    int m = s.length(), n = p.length();
    boolean[][] dp = new boolean[m + 1][n + 1];
    dp[0][0] = true;
    
    // Handle patterns like *, **, ***
    for (int j = 1; j <= n; j++) {
        if (p.charAt(j - 1) == '*') {
            dp[0][j] = dp[0][j - 1];
        }
    }
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (p.charAt(j - 1) == '*') {
                dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
            } else if (p.charAt(j - 1) == '?' || 
                       s.charAt(i - 1) == p.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }
    return dp[m][n];
}

// Regular Expression Matching
public boolean isMatchRegex(String s, String p) {
    int m = s.length(), n = p.length();
    boolean[][] dp = new boolean[m + 1][n + 1];
    dp[0][0] = true;
    
    // Handle a*, a*b*, etc
    for (int j = 2; j <= n; j++) {
        if (p.charAt(j - 1) == '*') {
            dp[0][j] = dp[0][j - 2];
        }
    }
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (p.charAt(j - 1) == '.' || 
                s.charAt(i - 1) == p.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1];
            } else if (p.charAt(j - 1) == '*') {
                dp[i][j] = dp[i][j - 2]; // 0 occurrence
                if (p.charAt(j - 2) == '.' || 
                    p.charAt(j - 2) == s.charAt(i - 1)) {
                    dp[i][j] = dp[i][j] || dp[i - 1][j];
                }
            }
        }
    }
    return dp[m][n];
}

// Distinct Subsequences
public int numDistinct(String s, String t) {
    int m = s.length(), n = t.length();
    int[][] dp = new int[m + 1][n + 1];
    
    for (int i = 0; i <= m; i++) dp[i][0] = 1;
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            dp[i][j] = dp[i - 1][j];
            if (s.charAt(i - 1) == t.charAt(j - 1)) {
                dp[i][j] += dp[i - 1][j - 1];
            }
        }
    }
    return dp[m][n];
}

// Interleaving String
public boolean isInterleave(String s1, String s2, String s3) {
    int m = s1.length(), n = s2.length();
    if (m + n != s3.length()) return false;
    
    boolean[][] dp = new boolean[m + 1][n + 1];
    dp[0][0] = true;
    
    for (int i = 1; i <= m; i++) {
        dp[i][0] = dp[i-1][0] && s1.charAt(i-1) == s3.charAt(i-1);
    }
    for (int j = 1; j <= n; j++) {
        dp[0][j] = dp[0][j-1] && s2.charAt(j-1) == s3.charAt(j-1);
    }
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            dp[i][j] = (dp[i-1][j] && s1.charAt(i-1) == s3.charAt(i+j-1)) ||
                       (dp[i][j-1] && s2.charAt(j-1) == s3.charAt(i+j-1));
        }
    }
    return dp[m][n];
}
```

---

## Pattern 9: DP on Trees

### Core Templates
```java
class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int val) { this.val = val; }
}

// House Robber III
public int rob(TreeNode root) {
    int[] result = robHelper(root);
    return Math.max(result[0], result[1]);
}

private int[] robHelper(TreeNode node) {
    if (node == null) return new int[]{0, 0};
    
    int[] left = robHelper(node.left);
    int[] right = robHelper(node.right);
    
    // [0] = max when not robbing current, [1] = max when robbing current
    int[] dp = new int[2];
    dp[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
    dp[1] = node.val + left[0] + right[0];
    
    return dp;
}

// Binary Tree Maximum Path Sum
int maxSum = Integer.MIN_VALUE;

public int maxPathSum(TreeNode root) {
    maxPathSumHelper(root);
    return maxSum;
}

private int maxPathSumHelper(TreeNode node) {
    if (node == null) return 0;
    
    int left = Math.max(0, maxPathSumHelper(node.left));
    int right = Math.max(0, maxPathSumHelper(node.right));
    
    maxSum = Math.max(maxSum, left + right + node.val);
    
    return Math.max(left, right) + node.val;
}

// Longest Path in Tree (Diameter)
int diameter = 0;

public int diameterOfBinaryTree(TreeNode root) {
    height(root);
    return diameter;
}

private int height(TreeNode node) {
    if (node == null) return 0;
    
    int left = height(node.left);
    int right = height(node.right);
    
    diameter = Math.max(diameter, left + right);
    
    return 1 + Math.max(left, right);
}

// Count Unique Binary Search Trees (Catalan Number)
public int numTrees(int n) {
    int[] dp = new int[n + 1];
    dp[0] = dp[1] = 1;
    
    for (int nodes = 2; nodes <= n; nodes++) {
        for (int root = 1; root <= nodes; root++) {
            int left = root - 1;
            int right = nodes - root;
            dp[nodes] += dp[left] * dp[right];
        }
    }
    return dp[n];
}
```

---

## Pattern 10: DP on Grids

### Core Templates
```java
// Unique Paths
public int uniquePaths(int m, int n) {
    int[][] dp = new int[m][n];
    
    for (int i = 0; i < m; i++) dp[i][0] = 1;
    for (int j = 0; j < n; j++) dp[0][j] = 1;
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    }
    return dp[m-1][n-1];
}

// Unique Paths II (With Obstacles)
public int uniquePathsWithObstacles(int[][] grid) {
    int m = grid.length, n = grid[0].length;
    if (grid[0][0] == 1) return 0;
    
    int[][] dp = new int[m][n];
    dp[0][0] = 1;
    
    for (int i = 1; i < m; i++) {
        dp[i][0] = (grid[i][0] == 0 && dp[i-1][0] == 1) ? 1 : 0;
    }
    for (int j = 1; j < n; j++) {
        dp[0][j] = (grid[0][j] == 0 && dp[0][j-1] == 1) ? 1 : 0;
    }
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (grid[i][j] == 0) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
    }
    return dp[m-1][n-1];
}

// Minimum Path Sum
public int minPathSum(int[][] grid) {
    int m = grid.length, n = grid[0].length;
    int[][] dp = new int[m][n];
    dp[0][0] = grid[0][0];
    
    for (int i = 1; i < m; i++) dp[i][0] = dp[i-1][0] + grid[i][0];
    for (int j = 1; j < n; j++) dp[0][j] = dp[0][j-1] + grid[0][j];
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = grid[i][j] + Math.min(dp[i-1][j], dp[i][j-1]);
        }
    }
    return dp[m-1][n-1];
}

// Maximal Square
public int maximalSquare(char[][] matrix) {
    int m = matrix.length, n = matrix[0].length;
    int[][] dp = new int[m + 1][n + 1];
    int maxSide = 0;
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (matrix[i-1][j-1] == '1') {
                dp[i][j] = Math.min(dp[i-1][j], 
                           Math.min(dp[i][j-1], dp[i-1][j-1])) + 1;
                maxSide = Math.max(maxSide, dp[i][j]);
            }
        }
    }
    return maxSide * maxSide;
}

// Dungeon Game
public int calculateMinimumHP(int[][] dungeon) {
    int m = dungeon.length, n = dungeon[0].length;
    int[][] dp = new int[m][n];
    
    dp[m-1][n-1] = Math.max(1, 1 - dungeon[m-1][n-1]);
    
    for (int i = m - 2; i >= 0; i--) {
        dp[i][n-1] = Math.max(1, dp[i+1][n-1] - dungeon[i][n-1]);
    }
    for (int j = n - 2; j >= 0; j--) {
        dp[m-1][j] = Math.max(1, dp[m-1][j+1] - dungeon[m-1][j]);
    }
    
    for (int i = m - 2; i >= 0; i--) {
        for (int j = n - 2; j >= 0; j--) {
            int minHealth = Math.min(dp[i+1][j], dp[i][j+1]);
            dp[i][j] = Math.max(1, minHealth - dungeon[i][j]);
        }
    }
    return dp[0][0];
}
```

---

## Backtracking Patterns

### 1. Subsets (All Combinations)
```java
// Find all subsets of an array
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    backtrackSubsets(nums, 0, new ArrayList<>(), result);
    return result;
}

private void backtrackSubsets(int[] nums, int start, 
                              List<Integer> current, List<List<Integer>> result) {
    result.add(new ArrayList<>(current)); // Add current subset
    
    for (int i = start; i < nums.length; i++) {
        current.add(nums[i]);                    // Choose
        backtrackSubsets(nums, i + 1, current, result); // Explore
        current.remove(current.size() - 1);      // Un-choose (backtrack)
    }
}

// Subsets with Duplicates
public List<List<Integer>> subsetsWithDup(int[] nums) {
    Arrays.sort(nums); // IMPORTANT: Sort to handle duplicates
    List<List<Integer>> result = new ArrayList<>();
    backtrackSubsetsDup(nums, 0, new ArrayList<>(), result);
    return result;
}

private void backtrackSubsetsDup(int[] nums, int start,
                                 List<Integer> current, List<List<Integer>> result) {
    result.add(new ArrayList<>(current));
    
    for (int i = start; i < nums.length; i++) {
        if (i > start && nums[i] == nums[i-1]) continue; // Skip duplicates
        current.add(nums[i]);
        backtrackSubsetsDup(nums, i + 1, current, result);
        current.remove(current.size() - 1);
    }
}
```

### 2. Subsequences (All combinations of String)
```java
// All subsequences of a string
public List<String> subsequences(String s) {
    List<String> result = new ArrayList<>();
    backtrackSubseq(s, 0, new StringBuilder(), result);
    return result;
}

private void backtrackSubseq(String s, int index, 
                             StringBuilder current, List<String> result) {
    if (index == s.length()) {
        result.add(current.toString());
        return;
    }
    
    // Don't include current character
    backtrackSubseq(s, index + 1, current, result);
    
    // Include current character
    current.append(s.charAt(index));
    backtrackSubseq(s, index + 1, current, result);
    current.deleteCharAt(current.length() - 1);
}

// Generate Parentheses (Valid subsequences)
public List<String> generateParenthesis(int n) {
    List<String> result = new ArrayList<>();
    backtrackParens(result, new StringBuilder(), 0, 0, n);
    return result;
}

private void backtrackParens(List<String> result, StringBuilder current,
                             int open, int close, int max) {
    if (current.length() == max * 2) {
        result.add(current.toString());
        return;
    }
    
    if (open < max) {
        current.append('(');
        backtrackParens(result, current, open + 1, close, max);
        current.deleteCharAt(current.length() - 1);
    }
    if (close < open) {
        current.append(')');
        backtrackParens(result, current, open, close + 1, max);
        current.deleteCharAt(current.length() - 1);
    }
}
```

### 3. Permutations (CRITICAL PATTERN)
```java
// Array Permutations - Approach 1 (Using visited array)
public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    boolean[] used = new boolean[nums.length];
    backtrackPermute(nums, new ArrayList<>(), used, result);
    return result;
}

private void backtrackPermute(int[] nums, List<Integer> current,
                              boolean[] used, List<List<Integer>> result) {
    if (current.size() == nums.length) {
        result.add(new ArrayList<>(current));
        return;
    }
    
    for (int i = 0; i < nums.length; i++) {
        if (used[i]) continue; // Skip already used elements
        
        used[i] = true;                  // Mark as used
        current.add(nums[i]);            // Choose
        backtrackPermute(nums, current, used, result); // Explore
        current.remove(current.size() - 1); // Un-choose
        used[i] = false;                 // Unmark (backtrack)
    }
}

// Array Permutations - Approach 2 (Swapping - in-place)
public List<List<Integer>> permuteSwap(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    backtrackSwap(nums, 0, result);
    return result;
}

private void backtrackSwap(int[] nums, int start, List<List<Integer>> result) {
    if (start == nums.length) {
        List<Integer> perm = new ArrayList<>();
        for (int num : nums) perm.add(num);
        result.add(perm);
        return;
    }
    
    for (int i = start; i < nums.length; i++) {
        swap(nums, start, i);              // Choose
        backtrackSwap(nums, start + 1, result); // Explore
        swap(nums, start, i);              // Un-choose (backtrack)
    }
}

private void swap(int[] nums, int i, int j) {
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
}

// Permutations with Duplicates
public List<List<Integer>> permuteUnique(int[] nums) {
    Arrays.sort(nums); // Sort to handle duplicates
    List<List<Integer>> result = new ArrayList<>();
    boolean[] used = new boolean[nums.length];
    backtrackPermuteDup(nums, new ArrayList<>(), used, result);
    return result;
}

private void backtrackPermuteDup(int[] nums, List<Integer> current,
                                 boolean[] used, List<List<Integer>> result) {
    if (current.size() == nums.length) {
        result.add(new ArrayList<>(current));
        return;
    }
    
    for (int i = 0; i < nums.length; i++) {
        if (used[i]) continue;
        // Skip duplicates: if current is same as previous AND previous not used
        if (i > 0 && nums[i] == nums[i-1] && !used[i-1]) continue;
        
        used[i] = true;
        current.add(nums[i]);
        backtrackPermuteDup(nums, current, used, result);
        current.remove(current.size() - 1);
        used[i] = false;
    }
}

// String Permutations
public List<String> stringPermutations(String s) {
    List<String> result = new ArrayList<>();
    char[] chars = s.toCharArray();
    backtrackStringPerm(chars, 0, result);
    return result;
}

private void backtrackStringPerm(char[] chars, int start, List<String> result) {
    if (start == chars.length) {
        result.add(new String(chars));
        return;
    }
    
    for (int i = start; i < chars.length; i++) {
        swapChars(chars, start, i);
        backtrackStringPerm(chars, start + 1, result);
        swapChars(chars, start, i);
    }
}

private void swapChars(char[] chars, int i, int j) {
    char temp = chars[i];
    chars[i] = chars[j];
    chars[j] = temp;
}
```

### BACKTRACKING KEY CONCEPTS
```java
/**
 * CRITICAL UNDERSTANDING OF BACKTRACKING:
 * 
 * Backtracking is about exploring ALL possible solutions by:
 * 1. CHOOSE: Make a choice
 * 2. EXPLORE: Recursively explore with that choice
 * 3. UN-CHOOSE: Undo the choice (backtrack) to try other options
 * 
 * Why do we need backtracking?
 * - To generate all possible combinations/permutations
 * - To find solutions that satisfy constraints
 * - To avoid exploring invalid paths early (pruning)
 * 
 * When to use visited/used array vs swapping?
 * - VISITED ARRAY: When order matters and we need to track what's used
 *   Example: [1,2,3] → [1,2,3], [1,3,2], [2,1,3], etc.
 * 
 * - SWAPPING: When we want in-place permutations
 *   More memory efficient but modifies the array
 * 
 * Key Pattern Recognition:
 * - Subsets: Start index moves forward (i+1)
 * - Permutations: Check all positions, use visited array
 * - Combinations: Similar to subsets but with size constraint
 */

// Combination Sum (Can reuse elements)
public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> result = new ArrayList<>();
    Arrays.sort(candidates);
    backtrackCombSum(candidates, target, 0, new ArrayList<>(), result);
    return result;
}

private void backtrackCombSum(int[] candidates, int remain, int start,
                              List<Integer> current, List<List<Integer>> result) {
    if (remain < 0) return;
    if (remain == 0) {
        result.add(new ArrayList<>(current));
        return;
    }
    
    for (int i = start; i < candidates.length; i++) {
        current.add(candidates[i]);
        // NOTE: Pass 'i' (not i+1) because we can reuse same element
        backtrackCombSum(candidates, remain - candidates[i], i, current, result);
        current.remove(current.size() - 1);
    }
}

// Combination Sum II (Each element used once)
public List<List<Integer>> combinationSum2(int[] candidates, int target) {
    Arrays.sort(candidates);
    List<List<Integer>> result = new ArrayList<>();
    backtrackCombSum2(candidates, target, 0, new ArrayList<>(), result);
    return result;
}

private void backtrackCombSum2(int[] candidates, int remain, int start,
                               List<Integer> current, List<List<Integer>> result) {
    if (remain < 0) return;
    if (remain == 0) {
        result.add(new ArrayList<>(current));
        return;
    }
    
    for (int i = start; i < candidates.length; i++) {
        if (i > start && candidates[i] == candidates[i-1]) continue;
        current.add(candidates[i]);
        backtrackCombSum2(candidates, remain - candidates[i], i + 1, current, result);
        current.remove(current.size() - 1);
    }
}
```

### 4. N-Queens (Grid Backtracking)
```java
public List<List<String>> solveNQueens(int n) {
    List<List<String>> result = new ArrayList<>();
    char[][] board = new char[n][n];
    for (int i = 0; i < n; i++) {
        Arrays.fill(board[i], '.');
    }
    backtrackQueens(board, 0, result);
    return result;
}

private void backtrackQueens(char[][] board, int row, List<List<String>> result) {
    if (row == board.length) {
        result.add(constructBoard(board));
        return;
    }
    
    for (int col = 0; col < board.length; col++) {
        if (isValidQueen(board, row, col)) {
            board[row][col] = 'Q';           // Choose
            backtrackQueens(board, row + 1, result); // Explore
            board[row][col] = '.';           // Un-choose (backtrack)
        }
    }
}

private boolean isValidQueen(char[][] board, int row, int col) {
    int n = board.length;
    
    // Check column
    for (int i = 0; i < row; i++) {
        if (board[i][col] == 'Q') return false;
    }
    
    // Check diagonal (top-left)
    for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == 'Q') return false;
    }
    
    // Check diagonal (top-right)
    for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
        if (board[i][j] == 'Q') return false;
    }
    
    return true;
}

private List<String> constructBoard(char[][] board) {
    List<String> result = new ArrayList<>();
    for (char[] row : board) {
        result.add(new String(row));
    }
    return result;
}
```

### 5. Sudoku Solver
```java
public void solveSudoku(char[][] board) {
    backtrackSudoku(board);
}

private boolean backtrackSudoku(char[][] board) {
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            if (board[row][col] == '.') {
                for (char num = '1'; num <= '9'; num++) {
                    if (isValidSudoku(board, row, col, num)) {
                        board[row][col] = num;           // Choose
                        if (backtrackSudoku(board)) return true; // Explore
                        board[row][col] = '.';           // Un-choose
                    }
                }
                return false; // No valid number found
            }
        }
    }
    return true; // All cells filled
}

private boolean isValidSudoku(char[][] board, int row, int col, char num) {
    // Check row
    for (int j = 0; j < 9; j++) {
        if (board[row][j] == num) return false;
    }
    
    // Check column
    for (int i = 0; i < 9; i++) {
        if (board[i][col] == num) return false;
    }
    
    // Check 3x3 box
    int startRow = (row / 3) * 3;
    int startCol = (col / 3) * 3;
    for (int i = startRow; i < startRow + 3; i++) {
        for (int j = startCol; j < startCol + 3; j++) {
            if (board[i][j] == num) return false;
        }
    }
    
    return true;
}
```

### 6. Word Search
```java
public boolean exist(char[][] board, String word) {
    int m = board.length, n = board[0].length;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (backtrackWordSearch(board, word, i, j, 0)) {
                return true;
            }
        }
    }
    return false;
}

private boolean backtrackWordSearch(char[][] board, String word, 
                                    int row, int col, int index) {
    if (index == word.length()) return true;
    
    if (row < 0 || row >= board.length || col < 0 || col >= board[0].length ||
        board[row][col] != word.charAt(index)) {
        return false;
    }
    
    char temp = board[row][col];
    board[row][col] = '#'; // Mark as visited
    
    boolean found = backtrackWordSearch(board, word, row + 1, col, index + 1) ||
                    backtrackWordSearch(board, word, row - 1, col, index + 1) ||
                    backtrackWordSearch(board, word, row, col + 1, index + 1) ||
                    backtrackWordSearch(board, word, row, col - 1, index + 1);
    
    board[row][col] = temp; // Backtrack
    return found;
}
```

---

## Efficient Enumeration Patterns

### All Subarrays (Contiguous)
```java
/**
 * MOST EFFICIENT WAY TO FIND ALL SUBARRAYS
 * Time Complexity: O(n²) - unavoidable as there are n(n+1)/2 subarrays
 * Space Complexity: O(1) for iteration, O(n²) if storing all
 */
public List<List<Integer>> allSubarrays(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    int n = nums.length;
    
    // For each starting point
    for (int start = 0; start < n; start++) {
        // For each ending point
        for (int end = start; end < n; end++) {
            List<Integer> subarray = new ArrayList<>();
            for (int k = start; k <= end; k++) {
                subarray.add(nums[k]);
            }
            result.add(subarray);
        }
    }
    return result;
}

// Optimized version (if you only need to process, not store)
public void processAllSubarrays(int[] nums) {
    int n = nums.length;
    
    for (int start = 0; start < n; start++) {
        int sum = 0; // Example: maintaining running sum
        for (int end = start; end < n; end++) {
            sum += nums[end];
            // Process subarray from start to end
            System.out.println("Subarray [" + start + ", " + end + "] sum: " + sum);
        }
    }
}

// Maximum Subarray Sum (Kadane's Algorithm)
public int maxSubArray(int[] nums) {
    int maxSum = nums[0];
    int currentSum = nums[0];
    
    for (int i = 1; i < nums.length; i++) {
        currentSum = Math.max(nums[i], currentSum + nums[i]);
        maxSum = Math.max(maxSum, currentSum);
    }
    return maxSum;
}

// Subarray with given sum
public boolean subarraySum(int[] nums, int target) {
    Map<Integer, Integer> prefixSum = new HashMap<>();
    prefixSum.put(0, -1);
    int sum = 0;
    
    for (int i = 0; i < nums.length; i++) {
        sum += nums[i];
        if (prefixSum.containsKey(sum - target)) {
            return true;
        }
        prefixSum.put(sum, i);
    }
    return false;
}
```

### All Substrings (Contiguous)
```java
/**
 * MOST EFFICIENT WAY TO FIND ALL SUBSTRINGS
 * Time Complexity: O(n²) - unavoidable as there are n(n+1)/2 substrings
 */
public List<String> allSubstrings(String s) {
    List<String> result = new ArrayList<>();
    int n = s.length();
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j <= n; j++) {
            result.add(s.substring(i, j));
        }
    }
    return result;
}

// Optimized: Using StringBuilder for large strings
public List<String> allSubstringsOptimized(String s) {
    List<String> result = new ArrayList<>();
    int n = s.length();
    
    for (int i = 0; i < n; i++) {
        StringBuilder sb = new StringBuilder();
        for (int j = i; j < n; j++) {
            sb.append(s.charAt(j));
            result.add(sb.toString());
        }
    }
    return result;
}

// Process all substrings without storing (most memory efficient)
public void processAllSubstrings(String s) {
    int n = s.length();
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j <= n; j++) {
            String substring = s.substring(i, j);
            // Process substring
            System.out.println(substring);
        }
    }
}

// Longest Substring Without Repeating Characters (Sliding Window)
public int lengthOfLongestSubstring(String s) {
    Set<Character> set = new HashSet<>();
    int left = 0, maxLen = 0;
    
    for (int right = 0; right < s.length(); right++) {
        while (set.contains(s.charAt(right))) {
            set.remove(s.charAt(left));
            left++;
        }
        set.add(s.charAt(right));
        maxLen = Math.max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

---

## Optimization Techniques

### 1. Space Optimization
```java
// 2D to 1D DP Array
// Before: dp[i][j] = dp[i-1][j] + dp[i][j-1]
// After: Use two arrays or one array with careful iteration

// Example: Unique Paths
public int uniquePathsSpaceOptimized(int m, int n) {
    int[] dp = new int[n];
    Arrays.fill(dp, 1);
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[j] += dp[j-1];
        }
    }
    return dp[n-1];
}

// Rolling Array Technique
public int longestCommonSubsequenceOptimized(String text1, String text2) {
    int n = text2.length();
    int[] prev = new int[n + 1];
    int[] curr = new int[n + 1];
    
    for (int i = 1; i <= text1.length(); i++) {
        for (int j = 1; j <= n; j++) {
            if (text1.charAt(i-1) == text2.charAt(j-1)) {
                curr[j] = 1 + prev[j-1];
            } else {
                curr[j] = Math.max(prev[j], curr[j-1]);
            }
        }
        int[] temp = prev;
        prev = curr;
        curr = temp;
    }
    return prev[n];
}
```

### 2. State Compression
```java
// Bitmask DP for subset problems
public int shortestPathAllKeys(String[] grid) {
    int m = grid.length, n = grid[0].length;
    int allKeys = 0;
    int startX = 0, startY = 0;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            char c = grid[i].charAt(j);
            if (c == '@') {
                startX = i; startY = j;
            } else if (c >= 'a' && c <= 'f') {
                allKeys |= (1 << (c - 'a'));
            }
        }
    }
    
    Queue<int[]> queue = new LinkedList<>();
    Set<String> visited = new HashSet<>();
    queue.offer(new int[]{startX, startY, 0, 0}); // x, y, keys, steps
    visited.add(startX + "," + startY + "," + 0);
    
    int[][] dirs = {{0,1}, {1,0}, {0,-1}, {-1,0}};
    
    while (!queue.isEmpty()) {
        int[] curr = queue.poll();
        int x = curr[0], y = curr[1], keys = curr[2], steps = curr[3];
        
        if (keys == allKeys) return steps;
        
        for (int[] dir : dirs) {
            int nx = x + dir[0], ny = y + dir[1];
            int newKeys = keys;
            
            if (nx < 0 || nx >= m || ny < 0 || ny >= n) continue;
            
            char c = grid[nx].charAt(ny);
            if (c == '#') continue;
            
            if (c >= 'a' && c <= 'f') {
                newKeys |= (1 << (c - 'a'));
            }
            if (c >= 'A' && c <= 'F') {
                if ((keys & (1 << (c - 'A'))) == 0) continue;
            }
            
            String state = nx + "," + ny + "," + newKeys;
            if (visited.contains(state)) continue;
            
            visited.add(state);
            queue.offer(new int[]{nx, ny, newKeys, steps + 1});
        }
    }
    return -1;
}
```

### 3. Memoization Patterns
```java
// String as key (for array/string problems)
Map<String, Integer> memo = new HashMap<>();

private int helper(int[] nums, int index, int target) {
    String key = index + "," + target;
    if (memo.containsKey(key)) return memo.get(key);
    
    // base case and logic
    int result = 0; // computation
    
    memo.put(key, result);
    return result;
}

// Array as key (using Arrays.hashCode)
Map<Integer, Map<Integer, Integer>> memo2D = new HashMap<>();

private int helper2D(int i, int j) {
    if (!memo2D.containsKey(i)) {
        memo2D.put(i, new HashMap<>());
    }
    if (memo2D.get(i).containsKey(j)) {
        return memo2D.get(i).get(j);
    }
    
    int result = 0; // computation
    
    memo2D.get(i).put(j, result);
    return result;
}
```

---

## Common DP Problems by Pattern

### Buy and Sell Stock Problems
```java
// Best Time to Buy and Sell Stock (One transaction)
public int maxProfit(int[] prices) {
    int minPrice = Integer.MAX_VALUE;
    int maxProfit = 0;
    
    for (int price : prices) {
        minPrice = Math.min(minPrice, price);
        maxProfit = Math.max(maxProfit, price - minPrice);
    }
    return maxProfit;
}

// Best Time to Buy and Sell Stock II (Unlimited transactions)
public int maxProfitII(int[] prices) {
    int profit = 0;
    for (int i = 1; i < prices.length; i++) {
        if (prices[i] > prices[i-1]) {
            profit += prices[i] - prices[i-1];
        }
    }
    return profit;
}

// Best Time to Buy and Sell Stock III (At most 2 transactions)
public int maxProfitIII(int[] prices) {
    int buy1 = Integer.MAX_VALUE, buy2 = Integer.MAX_VALUE;
    int profit1 = 0, profit2 = 0;
    
    for (int price : prices) {
        buy1 = Math.min(buy1, price);
        profit1 = Math.max(profit1, price - buy1);
        buy2 = Math.min(buy2, price - profit1);
        profit2 = Math.max(profit2, price - buy2);
    }
    return profit2;
}

// Best Time to Buy and Sell Stock with Cooldown
public int maxProfitCooldown(int[] prices) {
    if (prices.length <= 1) return 0;
    
    int sell = 0, prevSell = 0, buy = Integer.MIN_VALUE;
    
    for (int price : prices) {
        int prevBuy = buy;
        buy = Math.max(buy, prevSell - price);
        prevSell = sell;
        sell = Math.max(sell, prevBuy + price);
    }
    return sell;
}
```

---

## Quick Reference Table

| Pattern | Time | Space | Key Insight |
|---------|------|-------|-------------|
| Fibonacci | O(n) | O(1) | Two variables |
| 0/1 Knapsack | O(n*W) | O(W) | Can't reuse items |
| Unbounded Knapsack | O(n*W) | O(W) | Can reuse items |
| LCS | O(m*n) | O(n) | Match or skip |
| LIS | O(n²) or O(n log n) | O(n) | Binary search optimization |
| Palindrome | O(n²) | O(n²) | Expand from center |
| MCM | O(n³) | O(n²) | Try all partitions |w
| Grid DP | O(m*n) | O(n) | Direction matters |

---

## Interview Tips

1. **Identify DP**: Look for overlapping subproblems and optimal substructure
2. **Start with recursion**: Write recursive solution first
3. **Add memoization**: Cache results of subproblems
4. **Convert to tabulation**: Build bottom-up if needed
5. **Optimize space**: Reduce dimensions when possible

### Common Mistakes to Avoid
- Forgetting base cases
- Wrong loop direction in tabulation
- Not handling edge cases (empty array, n=0)
- Incorrect space optimization (reading values before updating)
- Mixing up i and i+1 in rolling arrays

---

**END OF CHEAT SHEET**