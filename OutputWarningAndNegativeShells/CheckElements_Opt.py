import os
import re
import numpy as np

def process_and_count_failure_elements(file_path, element_list, part_failure_elements):
    """
    从 .k 文件中提取与给定 element_list 匹配的 pid 和 eid 信息。
    
    参数:
    - file_path: .k 文件路径。
    - element_list: 要匹配的 element_id 列表。
    - part_failure_elements: 记录 pid 与其对应 eid 列表的字典。
    
    返回:
    - part_failure_elements: 更新后的字典。
    """
    with open(file_path, 'r') as file:
        inside_element_block = False
        for line in file:
            line = line.strip()
            if line.startswith("$"):  # 跳过注释行
                continue
            if line.startswith("*"):  # 检查块的开始和结束
                inside_element_block = line.startswith("*ELEMENT")
                continue
            if inside_element_block:
                try:
                    eid = int(line[:8].strip())
                    pid = int(line[8:16].strip())
                except ValueError:
                    continue  # 忽略无效行
                if eid in element_list:  # 如果 eid 在给定列表中
                    part_failure_elements.setdefault(pid, []).append(eid)
    return part_failure_elements

def search_message_files(file_prefix, patterns):
    """
    搜索符合特定正则模式的日志文件并提取数据。
    
    参数:
    - file_prefix: 文件名前缀，用于筛选文件。
    - patterns: 字典，键为模式名称，值为正则表达式字符串。
    
    返回:
    - results: 包含每个模式匹配结果的字典，键为模式名称，值为 NumPy 数组。
    """
    # 编译正则表达式
    compiled_patterns = {key: re.compile(pattern) for key, pattern in patterns.items()}
    results = {key: [] for key in compiled_patterns}

    # 遍历当前目录中的所有符合前缀的文件
    current_dir = os.getcwd()
    for file_name in os.listdir(current_dir):
        if file_name.startswith(file_prefix):
            try:
                with open(file_name, 'r') as file:
                    for line in file:
                        for key, pattern in compiled_patterns.items():
                            match = pattern.search(line)
                            if match:
                                results[key].append(match.groups())
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
    
    # 将匹配结果转换为 NumPy 数组
    for key in results:
        results[key] = np.array(results[key])
    return results

def convert_numpy_chart_int(input_numpy):
    """
    将每个正则模式匹配结果的第一个列转换为整型。
    
    参数:
    - input_numpy: 包含 NumPy 数组的字典。
    
    返回:
    - results: 转换为整型的字典。
    """
    results = {}
    for key, array in input_numpy.items():
        if array.size > 0:  # 确保数组非空
            results[key] = np.round(array[:, 0].astype(float)).astype(int)
    return results

def check_failure_pid(folder_path, filetype, input_numpy):
    """
    检查文件夹中的所有 .k 文件，根据输入的 element_id 列表统计 pid 对应的失败元素。
    
    参数:
    - folder_path: 包含 .k 文件的文件夹路径。
    - filetype: 要筛选的文件类型（例如 ".k"）。
    - input_numpy: 包含 element_id 列表的字典。
    
    返回:
    - results: 包含每个 pid 及其失败元素的字典。
    """
    results = {key: {} for key in input_numpy.keys()}  # 初始化结果字典

    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(filetype):  # 筛选特定类型的文件
                file_path = os.path.join(root, file)
                for key, element_list in input_numpy.items():
                    results[key] = process_and_count_failure_elements(file_path, element_list, results[key])
                print(f"Checked .k file: {file_path}")
    return results

# 定义日志文件正则模式
patterns = {
    "warpage_angle": r"^\s*warpage angle of element ID= (\d+) is computed as ([\d\.E\+\-]+) degrees",
    "shell_failures": r"^\s*shell element (\d+)\s+failed at time\s+([\d\.E\+\-]+)",
    "node_deletions": r"^\s*node\s+number (\d+)\s+deleted at time\s+([\d\.E\+\-]+)",
    "solid_negative": r"^\s*negative volume in solid element # (\d+) cycle (\d+)"
}

# 1. 提取日志文件信息
file_prefix = "mes"
results = search_message_files(file_prefix, patterns)

# 2. 转换提取结果为整型
results_int = convert_numpy_chart_int(results)

# 3. 检查 .k 文件中 pid 和 eid 的对应关系
folder_path = "00_INCLUDE"
filetype_k = ".k"
Part_failure = check_failure_pid(folder_path, filetype_k, results_int)

# 4. 输出结果
print("Part Failure Elements:")
for key, failures in Part_failure.items():
    print(f"Pattern: {key}, Failures: {failures.keys()}")
