import pandas as pd
import argparse

def select_attributes(input_csv, output_csv, attribute_columns, attribute_values):
    # 读取输入CSV文件
    data = pd.read_csv(input_csv)

    # 创建一个布尔索引，用于筛选数据
    mask = data[attribute_columns].eq(attribute_values).all(axis=1)

    # 根据布尔索引筛选数据
    selected_data = data[mask]

    # 将选择的数据保存到输出CSV文件
    selected_data.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='属性选择工具')

    parser.add_argument('-i', '--input_csv', required=True, help='输入CSV文件名')
    parser.add_argument('-o', '--output_csv', required=True, help='输出CSV文件名')
    parser.add_argument('-c', '--columns', nargs='+', required=True, help='属性列的名称')
    parser.add_argument('-v', '--values', nargs='+', required=True, help='要筛选的属性值')

    args = parser.parse_args()

    # 确保属性列和属性值的数量一致
    if len(args.columns) != len(args.values):
        raise ValueError("属性列和属性值的数量不一致")

    select_attributes(args.input_csv, args.output_csv, args.columns, args.values)
