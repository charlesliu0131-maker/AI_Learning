'''
D03 矩阵乘法 = 变换（attention 的基石）

目标：① 用 PyTorch 验证形状规则 (2,3)@(3,4) → (2,4)
      ② 用点积视角复现 C[i,j] = A 的第 i 行 · B 的第 j 列
      ③ 写 2×2 矩阵作用在点 (x,y) 上的函数，观察「变换」
      ④ 复合变换：顺序很重要（呼应 3B1B EP4）

手算参照：C[0][0] = 1×7 + 2×11 + 3×15 = 74

跑出的结论：C 的每个格子 = A 的一行 与 B 的一列 做点积；
行数来自 A、列数来自 B，中间那个维度必须对齐。
并且：矩阵的「列」就是基向量 î、ĵ 变换后落到的位置。
'''

import torch

print("===== 1. 形状规则：为什么 (2,3)@(3,4) 得到 (2,4)？ =====")
A = torch.tensor([[1., 2., 3.],      # 2 行 3 列
                  [4., 5., 6.]])
B = torch.tensor([[7.,  8.,  9.,  10.],   # 3 行 4 列
                  [11., 12., 13., 14.],
                  [15., 16., 17., 18.]])
C = A @ B          # @ 才是矩阵乘法（上一课说过：* 是对位相乘）
print("A.shape =", A.shape, " B.shape =", B.shape, " → C.shape =", C.shape)

# 规则拆开看：
#   A 的"列数 3" 必须 == B 的"行数 3"（内部维度对齐，否则无法相乘）
#   结果的形状 = (A 的行数 2, B 的列数 4)
assert A.shape[1] == B.shape[0], "内部维度必须相等"
assert C.shape == (2, 4)
print("✅ 形状规则验证：C 的第 (i,j) 个格子，来自 A 第 i 行 与 B 第 j 列的「碰撞」")

print("\n===== 2. 点积视角：C[i,j] = A[i,:] · B[:,j] =====")
print("C[0,0] 手算 = 1×7 + 2×11 + 3×15 =", C[0, 0].item())
# 用昨天学的 torch.dot 复现
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        manual = torch.dot(A[i], B[:, j])
        print("C[{i},{j}] 用点积复现 = torch.dot(A[{i}], B[:,{j}]) =", manual.item())
        assert abs(manual.item() - C[i, j].item()) < 1e-6
print("✅ 点积视角验证通过 —— 矩阵乘法 = 一行一列对齐做点积，排成新格子")

print("\n===== 3. 2×2 矩阵作用在点上：观察「变换」 =====")
def transform(M, pts):
    """M 是 2×2 矩阵，pts 是多个点(每行一个点)组成的张量。"""
    # 每个点 (x,y) → M @ [x,y]ᵀ，一次算完所有点。
    # 为什么写 pts @ M.T 而不是 M @ pts？因为 (M·p)ᵀ = pᵀ·Mᵀ ——
    # 把「左乘 M」改写成「右乘 M 的转置」，就能一次性处理整批点。
    return pts @ M.T

# 3B1B EP3 的核心：矩阵的「每一列」= 基向量 î、ĵ 变换后落到的位置
# 先看标准基向量：î = (1,0)，ĵ = (0,1)
pts = torch.tensor([[1., 0.],     # î（x 方向单位向量）
                    [0., 1.],     # ĵ（y 方向单位向量）
                    [1., 1.]])    # 顺便带一个普通点 (1,1)

# 例 1：错切 shear —— 把 y 的分量"借"给 x
shear = torch.tensor([[1., 1.],
                      [0., 1.]])
print("shear = [[1,1],[0,1]]，作用在三个点上：")
print(transform(shear, pts))
# 观察：第 1 列 [1,0]ᵀ → î 原地不动；第 2 列 [1,1]ᵀ → ĵ 倒向 (1,1)
#       所以原来在 (0,1) 的 ĵ 现在躺到 (1,1)，网格被"推斜"了

# 例 2：旋转 90°（逆时针）
rot90 = torch.tensor([[0., -1.],
                      [1.,  0.]])
print("\nrot90 = [[0,-1],[1,0]]，作用在三个点上：")
print(transform(rot90, pts))
# 观察：第 1 列 [0,1]ᵀ → î 从 (1,0) 转到 (0,1)（竖起来了！）
#       第 2 列 [-1,0]ᵀ → ĵ 从 (0,1) 转到 (-1,0)

# assert 关卡：验证「矩阵列 = 基向量去向」
assert torch.equal(rot90 @ torch.tensor([1., 0.]), torch.tensor([0., 1.]))
assert torch.equal(shear @ torch.tensor([0., 1.]), torch.tensor([1., 1.]))
print("✅ 变换检查通过：M 的第 1 列就是 î 的去向，第 2 列就是 ĵ 的去向")

print("\n===== 4. 复合变换：顺序很重要（EP4 核心） =====")
# 对同一个点 (1,0)，先"旋转 90°"再"错切"，和先"错切"再"旋转 90°"，
# 得到的结果不一样 → 这就是"矩阵乘法不满足交换律"
x = torch.tensor([1., 0.])
first_rotate_then_shear = shear @ rot90 @ x    # 先 rot90，再 shear（从右往左读）
first_shear_then_rotate = rot90 @ shear @ x    # 先 shear，再 rot90

print("先旋转后错切：", first_rotate_then_shear)
print("先错切后旋转：", first_shear_then_rotate)
assert not torch.equal(first_rotate_then_shear, first_shear_then_rotate)
print("✅ 两种顺序结果不同 —— 所以写代码时「谁先作用」要看乘法顺序")

# 但有一个性质总是成立：结合律
# shear @ (rot90 @ x) == (shear @ rot90) @ x  —— 都表示"先 rot90 后 shear"
combo = shear @ rot90
assert torch.equal(combo @ x, shear @ (rot90 @ x))
print("✅ 结合律验证：先算哪个括号结果一样，只要顺序没变")

print("\n🎉 D2 动手全部通过！")
print("复盘一句话：为什么 (2,3)@(3,4) 结果是 (2,4)？")
print("→ 因为 C 的每个格子 = A 的一行(3个数) 与 B 的一列(3个数) 做点积，")
print("  行数来自 A(2)、列数来自 B(4)，内部那 3 必须对齐。")
