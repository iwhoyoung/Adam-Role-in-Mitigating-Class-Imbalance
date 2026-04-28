import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 生成示例数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# 这里我们使用一个带有噪声的正弦函数作为示例
# Z = X**3 - 3*X*Y**2 + Y**3 - 3*X**2*Y + 0.02 * np.random.randn(X.shape[0], X.shape[1])
Z = np.abs(1-np.abs((np.sin(np.sqrt((X)**2 + (Y)**2)) * np.sin(0.5*np.sqrt((X-2.32)**2 + (Y-3.73)**2)) + 0.005 * np.random.randn(X.shape[0], X.shape[1]))))
# random_add = np.random.rand(100,100) * 0.1
# Z[Z<0.01] += random_add[Z < 0.01]
# Z = np.sin(np.sqrt(X**2 + Y**2))  # 示例函数

# # 生成3D网格数据
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# x, y = np.meshgrid(x, y)
# z = np.sin(np.sqrt(x**2 + y**2))  # 示例函数，你可以根据需要更改

plt.rcParams.update({'font.size': 14})
# 创建图形和3D轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制3D曲面图（可选）
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=1)#cm.coolwarm

# 绘制等势线（等高线投影到xy、xz和yz平面）
levels = np.arange(-1, 1.1, 0.2)  # 设置等势线的水平

# xy平面上的等高线
contour_xy = ax.contourf(X, Y, Z, levels=levels, cmap=cm.coolwarm, alpha=0.3, zdir='z', offset=-6)

# xz平面上的等高线
# contour_xz = ax.contourf(X, Z, Y.mean(axis=0), levels=levels, cmap=cm.coolwarm, alpha=0.3, zdir='y', offset=-6)

# yz平面上的等高线
# contour_yz = ax.contourf(Y, Z, X.mean(axis=0), levels=levels, cmap=cm.coolwarm, alpha=0.3, zdir='x', offset=-6)

# 去掉坐标轴
# ax.set_axis_off()

# 添加颜色条
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, fraction=0.1,pad=0.1)
ax.set_zlim(-0.2, 1)

cbar.ax.tick_params(labelsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.rcParams.update({'font.size': 14})
# 设置轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

fig.set_size_inches(6, 4)
# 显示图表
plt.tight_layout()

plt.savefig('/lichenghao/huY/ada_optimizer/submit/contour_map3d.png', dpi=400, bbox_inches = 'tight')