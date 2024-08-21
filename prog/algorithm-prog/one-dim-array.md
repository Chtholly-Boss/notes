# 一维数组
## rotate array
[Problem-Site](https://leetcode.cn/problems/rotate-array/description/)

* 直观上来看，开一个新数组，然后遍历原数组将元素放至对应位置即可
* In-place 解法有两种：
    * 数组反转
    * Step-by-Step
        * 计算得到元素的目标位置
        * 用临时变量存储将要被替换的元素
        * 将元素移动至目标位置
        * 同样的步骤应用至被替换的元素
        * 需要注意元素的全覆盖问题，进行一定的数学推导

[Solution](https://leetcode.cn/problems/rotate-array/solutions/551039/xuan-zhuan-shu-zu-by-leetcode-solution-nipk/)

## Plus One
[Problem-Site](https://leetcode.cn/problems/plus-one/)
维护一个进位标志，从末位开始即可

## Pivot Index
[Problem-Site](https://leetcode.cn/problems/find-pivot-index/description/)
前缀和的直接计算。
