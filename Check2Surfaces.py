import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import Normalize

def Get_Surface_Nodes(target_folder, Surface, Cordinate, FileType):
    # 初始化存储列表
    dframe = []    
    for cord in Cordinate:
        # 构造文件名和路径
        name = f"{Surface}{cord}{FileType}"  # 使用f-string
        target = os.path.join(target_folder, name)        
        # 读取文件，跳过第一行，并删除第一列
        df = pd.read_csv(target, skiprows=1, header=0)
        df = df.iloc[:, 1:]  # 保留除第一列外的所有列
        df = df.dropna(axis=1, how='all')  # axis=1 表示列，how='all' 表示整列为空才删除

        # 修改列名：提取数字并转换为整数
        df.columns = df.columns.str.extract('(\d+)').squeeze().astype(int)  # 提取列名中的数字，并转换为一维数组

        dframe.append(df)
    return dframe

def Get_Shell_elem(target_folder,k_file):
    target = os.path.join(target_folder,k_file)
    data = []
    with open(target,'r') as file:
        for line in file:
            # line = line.strip()
            if line:
                values = [
                int(line[0:8].strip()),  # 第 1 列
                int(line[8:16].strip()), # 第 2 列
                int(line[16:24].strip()), # 第 3 列
                int(line[24:32].strip()), # 第 4 列
                int(line[32:40].strip()), # 第 5 列
                int(line[40:48].strip()), # 第 6 列
            ]
            # 添加到数据列表
            data.append(values)
            data_array = np.array(data)
    return data_array

def AssembeShell(surf,Elem_surf,timestep):
    data = []
    for Elem in Elem_surf:
        n = [Elem[2],Elem[3],Elem[4],Elem[5]]
        value = [
            (surf[0][n[0]][timestep],surf[1][n[0]][timestep],surf[2][n[0]][timestep]),
            (surf[0][n[1]][timestep],surf[1][n[1]][timestep],surf[2][n[1]][timestep]),
            (surf[0][n[2]][timestep],surf[1][n[2]][timestep],surf[2][n[2]][timestep]),
            (surf[0][n[3]][timestep],surf[1][n[3]][timestep],surf[2][n[3]][timestep])
            ]
        data.append(value)
        data_array = np.array(data)
    return data_array

def compute_node_normals(surf,Elem_surf,timestep):
    """
    计算主表面节点法向量。
    
    参数：
        
        
    返回：
        
    """
    element_normals = []
    node_normals_dict = {}

    # 遍历每个单元，计算法向量
    for element in Elem_surf:
        # 获取单元节点的坐标
        n = [element[2],element[3],element[4],element[5]]
        points = [
            np.array([surf[0][n[0]][timestep], surf[1][n[0]][timestep], surf[2][n[0]][timestep]]),
            np.array([surf[0][n[1]][timestep], surf[1][n[1]][timestep], surf[2][n[1]][timestep]]),
            np.array([surf[0][n[2]][timestep], surf[1][n[2]][timestep], surf[2][n[2]][timestep]]),
            np.array([surf[0][n[3]][timestep], surf[1][n[3]][timestep], surf[2][n[3]][timestep]])
        ]

        vec1 = points[1] - points[0]
        vec2 = points[3] - points[0]
        normal1 = np.cross(vec1, vec2)
        
        vec3 = points[2] - points[1]
        vec4 = points[0] - points[1]
        normal2 = np.cross(vec3, vec4)
        
        normal = (normal1 + normal2) / 2.0

        # 单位化法向量
        normal = normal / np.linalg.norm(normal)
        element_normals.append(normal)

        # 将单元法向量累加到节点
        for node in n:
            if node not in node_normals_dict:
                node_normals_dict[node] = np.array([0.0, 0.0, 0.0])
            node_normals_dict[node] += normal

    # 对节点法向量进行单位化
    for node, normal in node_normals_dict.items():
        norm = np.linalg.norm(normal)
        if norm > 0:
            node_normals_dict[node] /= norm

    return node_normals_dict, np.array(element_normals)

def calculate_gap_or_penetration(surfA, surfB, timestep, normal_B, gap=0.0):
    """
    计算两个表面之间的间隙或穿透，保证 normal_B 与 surfB 的节点一致。
    
    参数：
        surfA: 从表面节点的坐标数据，包含 X、Y、Z 坐标
        surfB: 主表面节点的坐标数据，包含 X、Y、Z 坐标，列名为节点编号
        timestep: 当前的时间步
        normal_B: 字典，主表面每个节点的法向量，键为节点编号，值为法向量 (3,)
        gap: float, 接触间隙，默认值为 0.0
        
    返回：
        result: numpy array, 每个从表面节点的间隙或穿透值（间隙为正，穿透为负）
    """
    # 重组 surfA 为二维坐标数组
    nodes_A = np.column_stack([
        surfA[0].iloc[timestep, :],  # X 坐标
        surfA[1].iloc[timestep, :],  # Y 坐标
        surfA[2].iloc[timestep, :]   # Z 坐标
    ])
    
    # 获取主表面节点编号
    node_ids_B = surfB[0].columns
    
    # 重组 surfB 为二维坐标数组
    nodes_B = np.column_stack([
        surfB[0].iloc[timestep, :],  # X 坐标
        surfB[1].iloc[timestep, :],  # Y 坐标
        surfB[2].iloc[timestep, :]   # Z 坐标
    ])
    
    # 构建 normal_B 的数组，顺序与 nodes_B 对应
    normals_B = np.array([normal_B[node_id] for node_id in node_ids_B])
    
    result = []
    
    # 遍历从表面每个节点
    for node_A in nodes_A:
        # 计算与主表面节点的距离
        distances = np.linalg.norm(nodes_B - node_A, axis=1)
        closest_idx = np.argmin(distances)
        
        # 获取最近主表面节点的法向量
        normal = normals_B[closest_idx]
        
        # 计算间隙或穿透
        vector = node_A - nodes_B[closest_idx]
        penetration = np.dot(vector, normal) - gap
        result.append(penetration)
    
    return np.array(result)

def PotSurface(quadrilaterals):
    # 创建一个新的三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 初始化坐标的最大最小值
    x_min, y_min, z_min = 0, 0, 0
    x_max, y_max, z_max = 0, 0, 0

    # 遍历每个四边形并绘制
    for quad in quadrilaterals:

        quad = np.array(quad)
        x, y, z = quad[:, 0], quad[:, 1], quad[:, 2]
        # 更新最大最小值
        x_min, x_max = min(x_min, np.min(x)), max(x_max, np.max(x))
        y_min, y_max = min(y_min, np.min(y)), max(y_max, np.max(y))
        z_min, z_max = min(z_min, np.min(z)), max(z_max, np.max(z))
        
        # 保证闭合四边形
        verts = [[(x[i], y[i], z[i]) for i in range(4)]]        
        # 添加四边形到图中
        ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='w', alpha=0.25))

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    # 显示图形
    plt.show()

def Pot2Surface(quadrilaterals1,quadrilaterals2):
    # 创建一个新的三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 初始化坐标的最大最小值
    x_min, y_min, z_min = 0, 0, 0
    x_max, y_max, z_max = 0, 0, 0

    # 遍历每个四边形并绘制
    for quad in quadrilaterals1:

        quad = np.array(quad)
        x, y, z = quad[:, 0], quad[:, 1], quad[:, 2]
        # 更新最大最小值
        x_min, x_max = min(x_min, np.min(x)), max(x_max, np.max(x))
        y_min, y_max = min(y_min, np.min(y)), max(y_max, np.max(y))
        z_min, z_max = min(z_min, np.min(z)), max(z_max, np.max(z))
        
        # 保证闭合四边形
        verts = [[(x[i], y[i], z[i]) for i in range(4)]]        
        # 添加四边形到图中
        ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='w', alpha=0.25))

    for quad in quadrilaterals2:

        quad = np.array(quad)
        x, y, z = quad[:, 0], quad[:, 1], quad[:, 2]
        # 更新最大最小值
        x_min, x_max = min(x_min, np.min(x)), max(x_max, np.max(x))
        y_min, y_max = min(y_min, np.min(y)), max(y_max, np.max(y))
        z_min, z_max = min(z_min, np.min(z)), max(z_max, np.max(z))
        
        # 保证闭合四边形
        verts = [[(x[i], y[i], z[i]) for i in range(4)]]        
        # 添加四边形到图中
        ax.add_collection3d(Poly3DCollection(verts, facecolors='red', linewidths=1, edgecolors='b', alpha=0.25))   

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    # 显示图形
    plt.show()

def PenetrationAddID(ID, Penetration):
    """
    将穿透值与节点ID关联。
    
    参数：
        ID: 节点ID列表
        Penetration: 对应的穿透值
        
    返回：
        Penetration_ID: 字典，节点ID与穿透值的映射
    """
    Penetration_ID = {}
    for i, node_id in enumerate(ID):
        Penetration_ID[node_id] = Penetration[i]
    return Penetration_ID

def PlotSurface_Penetration(quadrilaterals, Elem_surf, Penetration_ID, penetration_values):
    """
    根据穿透值绘制表面四边形，并根据穿透值为每个点着色。
    
    参数：
        quadrilaterals: List[List[np.array]], 每个四边形的顶点坐标，形状 (n, 4, 3)
        penetration_values: List[float], 每个四边形的穿透值，与 quadrilaterals 顺序对应
    """
    # 创建一个新的三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 初始化坐标的最大最小值
    x_min, y_min, z_min = 0, 0, 0
    x_max, y_max, z_max = 0, 0, 0
    
    # 标准化穿透值用于颜色映射
    norm = Normalize(vmin=min(penetration_values), vmax=max(penetration_values))
    cmap = plt.colormaps['coolwarm']  # 使用 coolwarm 颜色映射（蓝-白-红）

    # 遍历每个四边形并绘制
    for quad, element in zip(quadrilaterals, Elem_surf):
        quad = np.array(quad)
        x, y, z = quad[:, 0], quad[:, 1], quad[:, 2]
        n = [element[2], element[3], element[4], element[5]]
        # 更新最大最小值
        x_min, x_max = min(x_min, np.min(x)), max(x_max, np.max(x))
        y_min, y_max = min(y_min, np.min(y)), max(y_max, np.max(y))
        z_min, z_max = min(z_min, np.min(z)), max(z_max, np.max(z))
        
        # 计算穿透值
        penetration = np.mean([Penetration_ID[node_id] for node_id in n])
        
        # 定义颜色
        color = cmap(norm(penetration))
        
        # 保证闭合四边形
        verts = [[(x[i], y[i], z[i]) for i in range(4)]]
        # 添加四边形到图中
        ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=1, edgecolors='k', alpha=0.8))

    # 设置坐标范围
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # 添加颜色条
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='vertical', label='Penetration Value')

    # 显示图形
    plt.show()

target_folder = r".\SFSA1"
SurfA_Name = "Pia"
SurfB_Name = "CSF"
Cordinate = ["X", "Y", "Z"]
FileType_csv = ".csv"
FileType_k = ".k"
surfA = Get_Surface_Nodes(target_folder, SurfA_Name, Cordinate, FileType_csv)
surfB = Get_Surface_Nodes(target_folder, SurfB_Name, Cordinate, FileType_csv)
Elem_surfA = Get_Shell_elem(target_folder,SurfA_Name+FileType_k)
Elem_surfB = Get_Shell_elem(target_folder,SurfB_Name+FileType_k)
Time_step = 200
surfA_ToPlot = AssembeShell(surfA,Elem_surfA,Time_step)
surfB_ToPlot = AssembeShell(surfB,Elem_surfB,Time_step)
surfB_nodes_normal, surfB_elems_normal = compute_node_normals(surfB,Elem_surfB,Time_step)
surfA_Penetration = calculate_gap_or_penetration(surfA, surfB, Time_step, surfB_nodes_normal, gap=0.0)

# 调用修改后的函数
surfA_Penetration_id = PenetrationAddID(surfA[0].columns, surfA_Penetration)
PlotSurface_Penetration(surfA_ToPlot, Elem_surfA, surfA_Penetration_id, surfA_Penetration)

