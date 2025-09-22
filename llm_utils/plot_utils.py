import matplotlib.pyplot as plt
import plotly.graph_objects as go
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


class Sunburst_Handle:

    @staticmethod
    def align_nested_dict(nested_dict):
        """ 将嵌套字典按照最大深度对齐，不足的复制最后一个叶子节点
        """
        def get_max_depth(d, current_depth=0):
            if not isinstance(d, dict):
                return current_depth
            
            max_depth = current_depth
            for value in d.values():
                child_depth = get_max_depth(value, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        def align_dict(d, current_depth, max_depth):
            if current_depth >= max_depth:
                return d
            
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    aligned_value = align_dict(value, current_depth + 1, max_depth)
                    result[key] = aligned_value
                else:
                    if current_depth == max_depth - 1:
                        result[key] = value
                    else:
                        result[key] = align_dict({key: value}, current_depth + 1, max_depth)
            
            return result
        
        max_depth = get_max_depth(nested_dict)
        return align_dict(nested_dict, 0, max_depth)
    
    @staticmethod
    def nested_dict_to_sunburst(nested_dict, parent='', node_counter=None):
        """ 将嵌套字典转换为旭日图所需的 labels, parents, values
            nested_dict: dict[str, dict|float]
            parent: 当前层级的父节点名称（内部递归使用）
            node_counter: 用于生成唯一节点ID的计数器
        """
        if node_counter is None:
            node_counter = {'count': 0}
        
        labels = []
        parents = []
        values = []
        current_total = 0  # 当前字典的总值（所有直接子节点值之和）

        for key, value in nested_dict.items():
            # 断言检查：确保键不包含分隔符
            assert '~' not in str(key), f"键 '{key}' 包含保留分隔符 '~'，请修改键名"
            
            # 生成唯一节点ID
            node_counter['count'] += 1
            unique_key = f"{key}~{node_counter['count']}"
            
            if isinstance(value, (int, float)):
                # 叶子节点：直接添加
                labels.append(unique_key)
                parents.append(parent)
                values.append(value)
                current_total += value
            
            elif isinstance(value, dict):
                # 递归处理子字典
                child_labels, child_parents, child_values, child_total = Sunburst_Handle.nested_dict_to_sunburst(value, parent=unique_key, node_counter=node_counter)
                
                # 添加当前分支节点（值为子字典的总值）
                labels.append(unique_key)
                parents.append(parent)
                values.append(child_total)
                current_total += child_total
                
                # 添加子字典的节点
                labels.extend(child_labels)
                parents.extend(child_parents)
                values.extend(child_values)
        
        return labels, parents, values, current_total

    @staticmethod
    def create_display_mapping(labels):
        """ 创建显示名称映射，将唯一ID映射回原始名称 """
        display_map = {}
        for label in labels:
            # 使用安全分隔符分割，恢复原始名称
            if '~' in label:
                original_name = label.split('~')[0]
            else:
                original_name = label
            display_map[label] = original_name
        return display_map

    @staticmethod
    def plot_sunburst(nested_dict, title="", size=(600, 600), dpi=300, uniform_depth=True):
        """ 绘制旭日图
        """
        if uniform_depth:
            nested_dict = Sunburst_Handle.align_nested_dict(nested_dict)
        labels, parents, values, _ = Sunburst_Handle.nested_dict_to_sunburst(nested_dict)
        
        # 创建显示名称映射
        display_map = Sunburst_Handle.create_display_mapping(labels)
        
        fig = go.Figure(go.Sunburst(
            labels=labels,  # 使用唯一ID确保图表正确渲染
            parents=parents,
            values=values,
            branchvalues="total", # 分支值=子节点值之和
            insidetextorientation='radial', # 内部文本沿径向排列
            textinfo="label+value", # 显示标签和数值
            text=[display_map[label] for label in labels],  # 显示原始名称
            texttemplate='%{text}<br>%{value}',
            hovertemplate='<b>%{text}</b><br>值: %{value}<br>占比: %{percentParent:.1%}<extra></extra>',
            maxdepth=-1 # 显示所有层级
        ))
        
        fig.update_layout(
            margin=dict(t=50, l=0, r=0, b=0), 
            title={
                'text': title,
                'y':0.95,  # 标题垂直位置
                'x':0.5,   # 标题水平居中
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=16)  # 标题字体
            },
            height=size[1], # 图像高度
            width=size[0], # 图像宽度
            uniformtext=dict(minsize=10, mode='hide') # 文本最小字号，过小则隐藏
        )
        
        fig.update_traces(
            textfont_size=12,
            marker=dict(line=dict(width=1, color='white'))  # 白色分隔线
        )

        # 计算DPI缩放因子 (72是默认DPI)
        scale_factor = dpi / 72
        
        # 设置DPI（通过Plotly配置）
        fig.show(config={'toImageButtonOptions': {
            'format': 'png',
            'scale': scale_factor
        }})

