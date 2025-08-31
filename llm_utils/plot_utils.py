import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(data:list[float], bin_width:float=None, x_range:tuple=None, y_range:tuple=None, 
                   show_values:bool=True, figsize=(10, 6), dpi=100):
    """
        绘制直方图
        data: 输入数据列表
        bin_width: 直方图每个区间的长度（默认自适应）
        x_range: x轴范围 (x_min, x_max)（默认自适应）
        y_range: y轴范围 (y_min, y_max)（默认自适应）
        show_values: 是否在柱上方显示数值
        figsize: 图的尺寸 (宽度, 高度)
        dpi: 图像分辨率
    """
    # 创建图形和坐标轴，指定大小和分辨率
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 计算直方图
    if bin_width is not None:
        # 根据指定的bin_width计算bins
        if x_range is not None:
            bins = np.arange(x_range[0], x_range[1] + bin_width, bin_width)
        else:
            bins = np.arange(min(data), max(data) + bin_width, bin_width)
    else:
        bins = 'auto'  # 使用matplotlib的自动bins计算
    
    # 绘制直方图
    n, bins, patches = ax.hist(data, bins=bins, edgecolor='black')
    
    # 设置x轴范围
    if x_range is not None:
        ax.set_xlim(x_range)
    
    # 设置y轴范围
    if y_range is not None:
        ax.set_ylim(y_range)
    
    # 在柱上方显示数值
    if show_values:
        for i in range(len(patches)):
            height = n[i]
            # if height > 0:  # 只显示高度大于0的柱
            ax.text(
                patches[i].get_x() + patches[i].get_width() / 2,
                height + 0.01 * max(n),    # 在柱顶稍上方位置
                f'{int(height)}',          # 显示整数值
                ha='center',               # 水平居中
                va='bottom',               # 垂直底部对齐
                fontsize=9
            )
    
    # 添加标签和标题
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Histogram', fontsize=14)
    
    # 网格线和布局调整
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


