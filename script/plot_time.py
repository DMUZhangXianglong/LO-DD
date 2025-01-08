'''
Author: DMUZhangXianglong 347913076@qq.com
Date: 2024-11-22 11:39:13
LastEditors: DMUZhangXianglong 347913076@qq.com
LastEditTime: 2024-11-22 11:46:31
FilePath: /LO-DD/script/plot_time.py
Description: 
'''
import matplotlib.pyplot as plt

# 读取文件中的点
def read_points_from_file(file_path):
    points = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    y = float(line.strip())  # 将每行转为浮点数
                    points.append(y)
                except ValueError:
                    print(f"Invalid value in line: {line.strip()}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return points

# 绘制两个点集
def plot_two_sets_of_points(points1, points2):
    plt.figure(figsize=(10, 6))
    
    # 绘制第一个点集
    plt.plot(points1, marker='o', linestyle='-', label='common', color='blue')
    
    # 绘制第二个点集
    plt.plot(points2, marker='s', linestyle='--', label='effect', color='red')
    
    # 设置图表信息
    plt.title("Comparison of Two method")
    plt.xlabel("Index")
    plt.ylabel("cost time (milliseconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

# 主程序
if __name__ == "__main__":
    file_path1 = "common.txt"  # 替换为你的第一个文件路径
    file_path2 = "effect.txt"  # 替换为你的第二个文件路径
    
    points1 = read_points_from_file(file_path1)
    points2 = read_points_from_file(file_path2)
    
    if points1 and points2:
        print(f"Read {len(points1)} points from {file_path1}.")
        print(f"Read {len(points2)} points from {file_path2}.")
        plot_two_sets_of_points(points1, points2)
    else:
        print("One or both files have no valid points to plot.")
