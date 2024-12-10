def process_and_count_nodes(file1_path, file2_path, output_path):
    # 读取第一个文件的数据（格式：nid, x, y, z）
    node_data = {}
    with open(file1_path, 'r') as file1:
        inside_node_block = False
        for line in file1:
            line = line.strip()
            if line.startswith("$"):  # 跳过以 $ 开头的行
                continue
            if line.startswith("*NODE"):
                inside_node_block = True
                continue
            if line.startswith("*END"):
                inside_node_block = False
                break
            if inside_node_block:
                parts = line.split()
                nid = int(parts[0])
                x, y, z = map(float, parts[1:4])
                node_data[nid] = (x, y, z)

    # 统计和替换操作
    total_nids = 0
    replaced_count = 0
    not_replaced_nids = set()
    processed_nids = set()

    with open(file2_path, 'r') as file2, open(output_path, 'w') as output_file:
        inside_node_block = False
        for line in file2:
            if line.startswith("$"):  # 跳过以 $ 开头的行
                continue
            if line.startswith("*NODE"):
                inside_node_block = True
                output_file.write(line)
                continue
            if line.startswith("*END") and inside_node_block:
                inside_node_block = False
                output_file.write(line)
                continue

            if inside_node_block:
                nid = int(line[:8].strip())  # 提取前 8 位的 nid
                total_nids += 1
                if nid in node_data:
                    # 替换 x, y, z，保持格式
                    x, y, z = node_data[nid]
                    updated_line = (
                        f"{line[:8]}"  # nid 和前面的空格
                        f"{x:16.6f}"  # x 占 16 位
                        f"{y:16.6f}"  # y 占 16 位
                        f"{z:16.6f}"  # z 占 16 位
                        f"{line[56:]}"  # 保留行尾 (0.0 和 0.0)
                    )
                    output_file.write(updated_line)
                    replaced_count += 1
                else:
                    not_replaced_nids.add(nid)
                    output_file.write(line)
            else:
                output_file.write(line)  # 写入非节点部分

    # 返回统计信息
    return {
        "total_nids": total_nids,
        "replaced_count": replaced_count,
        "not_replaced_nids": not_replaced_nids
    }

def process_and_count_nodes_message(file1,file2,output):
    statistics = process_and_count_nodes(file1,file2,output)
    # 输出统计信息
    print(f"{file2}中的总 nid 数量: {statistics['total_nids']}")
    print(f"被替换的 nid 数量: {statistics['replaced_count']}")
    print(f"未被替换的 nid 数量: {len(statistics['not_replaced_nids'])}")
    print(f"未被替换的 nid 列表: {sorted(statistics['not_replaced_nids'])}")

# 示例使用
Morphed_nodes_file_path = 'output_nodes'  # lsdyna输出文件
Original_nodes_file_path = 'FINAL_Muscle_Nodes_MORPHED.k'  # 需要替换的文件
Updated_nodes_file_path = 'FINAL_Muscle_Nodes_MORPHED_output.k'  # 输出文件路径
statistics = process_and_count_nodes_message(Morphed_nodes_file_path, Original_nodes_file_path, Updated_nodes_file_path)
Original_nodes_file_path = 'FINAL_HBM_Nodes_MORPHED.k'  # 需要替换的文件
Updated_nodes_file_path = 'FINAL_HBM_Nodes_MORPHED_output.k'  # 输出文件路径
statistics = process_and_count_nodes_message(Morphed_nodes_file_path, Original_nodes_file_path, Updated_nodes_file_path)