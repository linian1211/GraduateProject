import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定義旋轉軸（單位向量）
a_world = np.array([0, 0, 1])  # 例如繞 Z 軸旋轉
a_world = a_world / np.linalg.norm(a_world)  # 確保是單位向量

# 旋轉角度（弧度）
joint_angle = np.pi / 4  # 45 度

# 計算 K 矩陣
K = np.array([
    [0, -a_world[2], a_world[1]],
    [a_world[2], 0, -a_world[0]],
    [-a_world[1], a_world[0], 0]
])

# 計算 R_joint（旋轉矩陣）
R_joint = np.eye(3) + np.sin(joint_angle) * K + (1 - np.cos(joint_angle)) * (K @ K)

# 定義要旋轉的向量（例如 X 軸上的向量）
v_initial = np.array([1, 0, 0])

# 旋轉後的向量
v_rotated = R_joint @ v_initial

# ======= 視覺化 =======
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 原始向量
ax.quiver(0, 0, 0, v_initial[0], v_initial[1], v_initial[2], color='r', label='Initial Vector')

# 旋轉後的向量
ax.quiver(0, 0, 0, v_rotated[0], v_rotated[1], v_rotated[2], color='b', label='Rotated Vector')

# 旋轉軸
ax.quiver(0, 0, 0, a_world[0], a_world[1], a_world[2], color='g', label='Rotation Axis', linestyle='dashed')

# 設定軸範圍
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# 標籤與圖例
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
