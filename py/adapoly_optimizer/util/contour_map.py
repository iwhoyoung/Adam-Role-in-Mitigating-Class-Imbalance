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

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制等势线
contour = ax.contour(X, Y, Z, levels=10, cmap='viridis')

# 添加颜色条
cbar = fig.colorbar(contour)
cbar.set_label('Loss')

# 添加标题和标签
# ax.set_title('Contour Plot')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')

fig.set_size_inches(6, 2.5)
# 显示图表
plt.tight_layout()

plt.savefig('/lichenghao/huY/ada_optimizer/submit/contour_mapratio3.png', dpi=400, bbox_inches = 'tight')
