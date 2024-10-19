import json


with open('experiments/output/gpt_annotation/agreement.json', encoding='utf-8') as f:
    results = json.load(f)


for task in [
    'instruction_following',
    'correctness',
    'informativeness',
    'aesthetics',
    'harmlessness',
]:
    task_name = 'cot_' + task
    for model_name in ['gpt-4o-mini', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06']:
        for frame in [1, 2, 4, 8]:
            print(f'& {results[task_name][model_name][str(frame)]["Agreement ratio"]:.2f} ', end='')
    print('\\\\')


# for task in ['instruction_following', 'correctness', 'informativeness', 'aesthetics', 'harmlessness']:
#     plt.figure(figsize=(10, 6))
#     x = [1, 2, 4, 8]
#     line_list = []

#     for model_name in ['gpt-4o-mini', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06']:
#         task_name = 'simple_'+task
#         y = [
#             results[task_name][model_name][str(max_frames)]['Agreement ratio']
#             for max_frames in [1, 2, 4, 8]
#         ]

#         label = {
#             'gpt-4o-mini': 'GPT-4o-Mini',
#             'gpt-4o-2024-05-13': 'GPT-4o-0513',
#             'gpt-4o-2024-08-06': 'GPT-4o-0806',
#         }[model_name]
#         plt.plot(x, y, label=label, linewidth=2, linestyle='-', marker='o')

#         task_name = 'cot_'+task
#         y = [
#             results[task_name][model_name][str(max_frames)]['Agreement ratio']
#             for max_frames in [1, 2, 4, 8]
#         ]

#         label = {
#             'gpt-4o-mini': 'GPT-4o-Mini(CoT)',
#             'gpt-4o-2024-05-13': 'GPT-4o-0513(CoT)',
#             'gpt-4o-2024-08-06': 'GPT-4o-0806(CoT)',
#         }[model_name]
#         plt.plot(x, y, label=label, linewidth=2, linestyle='--', marker='x')

#     # 添加标题和标签
#     plt.title('Top-Tier AI Journal Style Line Plot', fontsize=16, weight='bold')
#     plt.xlabel('X-axis (Time)', fontsize=14)
#     plt.ylabel('Y-axis (Amplitude)', fontsize=14)

#     # 设置图例
#     plt.legend(loc='upper right', fontsize=12)

#     # 网格线
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)

#     # 设置坐标轴的刻度字体大小
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)

#     # 显示图表
#     plt.savefig(f'experiments/output/gpt_annotation/{task}.pdf')
