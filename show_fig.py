import matplotlib.pyplot as plt

# 数据
values_pantry = [0.0207, 0.0196, 0.0192, 0.0215]
values_instruments = [0.0524, 0.0529, 0.0559, 0.0522]
values_arts = [0.0342, 0.0348, 0.0341, 0.0385]
categories = ['wo_np', 'wo_mp', 'wo_pp', 'IDA-SR']

# 设置颜色
colors_pantry = ['skyblue', 'lightblue', 'deepskyblue', 'dodgerblue']
colors_instruments = ['lightcoral', 'indianred', 'darkred', 'firebrick']
colors_arts = ['lightgreen', 'mediumseagreen', 'forestgreen', 'darkgreen']

# 绘图
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# 绘制柱形图 - Pantry
axs[0].bar(categories, values_pantry, color=colors_pantry)
axs[0].set_ylabel('NDCG@10')
axs[0].set_title('Pantry')

# 绘制柱形图 - Instruments
axs[1].bar(categories, values_instruments, color=colors_instruments)
axs[1].set_ylabel('NDCG@10')
axs[1].set_title('Instruments')

# 绘制柱形图 - Arts
axs[2].bar(categories, values_arts, color=colors_arts)
axs[2].set_ylabel('NDCG@10')
axs[2].set_title('Arts')

# 调整布局
plt.tight_layout()

# 显示图表
# plt.show()
plt.savefig('result.png',dpi=500)