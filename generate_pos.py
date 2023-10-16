# 打开pos.txt文件并读取数据
with open("pos.txt", "r") as file:
    data = file.readlines()

# 创建一个输出文件
output_file = open("AMPpos.fasta", "w")

# 初始化计数器
counter = 1

# 遍历数据并将其写入文件
for line in data:
    sequence = line.strip()  # 移除行尾的换行符
    parts = sequence.split()
    sequence = parts[0]
    output_file.write(f">NO_{counter:06}\n{sequence}\n")
    counter += 1

# 关闭文件
output_file.close()
