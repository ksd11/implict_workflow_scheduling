import  matplotlib.pyplot as plt
import networkx as nx

def visualize_dag(G, title="DAG Visualization", layout_type="hierarchical"):
    """
    可视化DAG图
    
    参数:
        G: networkx.DiGraph 对象
        title: 图的标题
        layout_type: 布局类型，可选 "hierarchical" 或 "spring"
    """
    plt.figure(figsize=(12, 8))
    plt.title(title)
    
    # 选择布局方式
    if layout_type == "hierarchical":
        # 分层布局，适合展示DAG的层次结构
        pos = nx.spring_layout(G, k=2, iterations=50)
    else:
        # 弹簧布局，适合展示图的整体结构
        pos = nx.spring_layout(G, k=1)
    
    # 绘制边（带箭头）
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=50,
                          arrowstyle='-|>',
                          connectionstyle='arc3,rad=0.2')
    
    # 设置节点颜色
    node_colors = []
    for node in G.nodes():
        if node.endswith('_source'):
            node_colors.append('lightgreen')
        elif node.endswith('_sink'):
            node_colors.append('lightpink')
        else:
            node_colors.append('lightblue')
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=2000)
    
    # 添加节点标签
    nx.draw_networkx_labels(G, pos, font_size=8)

    # 添加边的标签（同时显示数据大小和概率）
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        data_size = data.get('data_size', 0)
        probability = data.get('probability', 1.0)
        edge_labels[(u, v)] = f'd={data_size:.1f}\np={probability:.2f}'
    
    # 添加边的权重标签（如果有data_size属性）
    # edge_labels = nx.get_edge_attributes(G, 'data_size')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    plt.tight_layout()
    plt.show()
    plt.close()

