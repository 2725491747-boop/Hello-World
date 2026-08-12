from typing import List

class Solution:
    nums = [4, 8, 6, 0, 9]
    target = 8

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] + nums[j] == target:
                    return [i, j]
        
        return []
print(Solution().twoSum(Solution().nums, Solution().target))