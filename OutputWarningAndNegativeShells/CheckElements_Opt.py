import os
import re
import numpy as np

# 定义文件前缀和正则模式
file_prefix = "mes"
warpage_angle_pattern = r"warpage angle of element ID= (\d+) is computed as ([\d\.E\+\-]+) degrees"
shell_failure_pattern = r"^\s*shell element (\d+)\s+failed at time\s+([\d\.E\+\-]+)"
node_deletion_pattern = r"^\s*node\s+number (\d+)\s+deleted at time\s+([\d\.E\+\-]+)"

# 存储提取结果
warpage_angle = []
shell_failures = []
node_deletions = []

# 获取当前文件夹中的所有文件
current_dir = os.getcwd()
files = [f for f in os.listdir(current_dir) if f.startswith(file_prefix)]

# 遍历所有符合条件的文件
for file_name in files:
    with open(file_name, 'r') as file:
        for line in file:
            # 匹配警告信息
            warpage_angle_match = re.search(warpage_angle_pattern, line)
            if warpage_angle_match:
                element_id = warpage_angle_match.group(1)
                angle = warpage_angle_match.group(2)
                warpage_angle.append((element_id, angle))
            
            # 匹配 shell element failure
            shell_failures_match = re.search(shell_failure_pattern, line)
            if shell_failures_match:
                element_id = shell_failures_match.group(1)
                time_failed = shell_failures_match.group(2)
                shell_failures.append((element_id, time_failed))
            
            # 匹配 node deletion
            node_deletions_match = re.search(node_deletion_pattern, line)
            if node_deletions_match:
                node_number = node_deletions_match.group(1)
                time_deleted = node_deletions_match.group(2)
                node_deletions.append((node_number, time_deleted))

# 将提取的结果转换为 NumPy 数组
warpage_angle_np = np.array(warpage_angle)
shell_failures_np = np.array(shell_failures)
node_deletions_np = np.array(node_deletions)

# 将字符串数组的第一列转换为 float 类型，并四舍五入
warpage_angle_float = warpage_angle_np[:, 0].astype(float)
shell_failures_float = shell_failures_np[:, 0].astype(float)
warpage_angle_int = np.round(warpage_angle_float).astype(int)
shell_failures_int = np.round(shell_failures_float).astype(int)

def process_and_count_Failure_Elements(file1_path, element_list, Part_FailureElements):
    with open(file1_path, 'r') as file1:
        inside_node_block = False
        for line in file1:
            line = line.strip()
            if line.startswith("$"):  # 跳过以 $ 开头的行
                continue
            if line.startswith("*"):  # 跳过以 * 开头的行
                if line.startswith("*ELEMENT"):
                    inside_node_block = True
                else:
                    inside_node_block = False
                continue          
            if inside_node_block:
                try:
                    eid = int(line[:8].strip())  # 提取前 8 位的 eid
                    pid = int(line[8:16].strip())  # 提取 8-16 位的 pid                    
                except ValueError:
                    continue  # 跳过无法转换为整数的行

                if eid in element_list:  # 如果 eid 在给定的 element_list 中
                    # 如果 pid 不在字典中，初始化一个空列表
                    if pid not in Part_FailureElements:
                        Part_FailureElements[pid] = []
                    # 向 pid 对应的列表中添加 eid
                    Part_FailureElements[pid].append(eid)

    return Part_FailureElements

# 指定文件夹路径
folder_path = "00_INCLUDE"
Part_shell_failures = {}
Part_warpage_angle = {}

# 遍历文件夹及其子文件夹，获取所有 .k 文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".k"):  # 过滤所有 .k 文件
            file_path = os.path.join(root, file)
            Part_warpage_angle = process_and_count_Failure_Elements(file_path, warpage_angle_int, Part_warpage_angle)            
            Part_shell_failures = process_and_count_Failure_Elements(file_path, shell_failures_int, Part_shell_failures)            
            print(f"Check .k file: {file_path}")

# 输出提取结果
print("Warpage Angle Element :")
print(list(Part_warpage_angle.keys()))

print("Shell Element Failures:")
print(list(Part_shell_failures.keys()))
