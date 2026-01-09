# old solution

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int DFS(TreeNode root, int max, PriorityQueue<Integer> heap) {
        if (root == null) return 0;

        int curr = root.val;

        heap.add(root.val);

        int left = DFS(root.left, max, heap);
        heap.add(left + curr);
        int right = DFS(root.right, max, heap);
        heap.add(right + curr);

        // if (left < 0 || right < 0) return 0;

        int total = left + right + curr;
        heap.add(total);

        return Math.max(left + curr, right + curr);
        // return Math.max(Math.max(left, curr), Math.max(right, curr));
    }

    public int maxPathSum(TreeNode root) {
        PriorityQueue<Integer> heap = new PriorityQueue<>(Collections.reverseOrder());
        int max = DFS(root, 0, heap);

        System.out.println("heap: " + heap);
        
        return heap.peek();
    }
}
```