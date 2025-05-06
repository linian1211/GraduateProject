import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#R_parent 繞 Y 軸 90° 旋轉的坐標系
#a_local 在R_parent坐標系中的一個向量(1 0 0)
#a_world a_local在世界坐標系中的實際向量

# 定義繞 Y 軸 90° 旋轉的矩陣
theta = np.radians(90)  # 轉 90 度
R_parent = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])

# 定義 node.a，在局部座標系中是 X 軸方向
a_local = np.array([1, 0, 0])

# 計算 a 在世界座標系的方向
a_world = R_parent @ a_local

# 建立 3D 圖表
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 繪製 X、Y、Z 軸
ax.quiver(0, 0, 0, 1, 0, 0, color='r', linestyle='dashed', alpha=0.3, label="X axis")
ax.quiver(0, 0, 0, 0, 1, 0, color='g', linestyle='dashed', alpha=0.3, label="Y axis")
ax.quiver(0, 0, 0, 0, 0, 1, color='b', linestyle='dashed', alpha=0.3, label="Z axis")

# 畫出原始向量 (a_local)
ax.quiver(0, 0, 0, *a_local, color='orange', linewidth=2, label="Original a_local")

# 畫出旋轉後的向量 (a_world)
ax.quiver(0, 0, 0, *a_world, color='purple', linewidth=2, label="Rotated a_world")

# 設定標籤
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Vector Rotation (90° around Y-axis)")
ax.legend()

# 顯示圖表
plt.show()
