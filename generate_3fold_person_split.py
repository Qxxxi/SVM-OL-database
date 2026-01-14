"""
3-Fold按人划分数据集
生成包含 'mode' 列的CSV文件，可直接用于 visual_processed.ipynb
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# ============ 配置 ============
INPUT_CSV = 'data2025.csv'  # 原始CSV文件
OUTPUT_PREFIX = 'fold'  # 输出文件前缀
NUM_FOLDS = 3

# ============ 读取数据 ============
print("=" * 60)
print("读取原始数据...")
df = pd.read_csv(INPUT_CSV, encoding='GBK')
print(f"总样本数: {len(df)}")
print(f"列名: {df.columns.tolist()}")

# ============ 按人分组 ============
print("\n" + "=" * 60)
print("按人分组...")

# 从视频路径中提取人名
# 支持多种路径格式:
# Windows: D:\QIAO\shipin\pz_gxq1.mp4
# Linux:   /home/user/shipin/pz_gxq1.mp4
# 人名规则: 文件名 pz_gxq1.mp4 -> 人名是 pz
import os

def extract_person_name(path):
    """从路径中提取人名"""
    # 使用os.path.basename获取文件名，兼容Windows和Linux
    filename = os.path.basename(path)  # 获取文件名：pz_gxq1.mp4
    # 去除扩展名
    name_without_ext = os.path.splitext(filename)[0]  # pz_gxq1
    # 提取人名（下划线之前的部分）
    person_name = name_without_ext.split('_')[0]  # pz
    return person_name

# 添加人名列
df['person'] = df['path'].apply(extract_person_name)

# 统计每个人的样本数
person_counts = df['person'].value_counts()
print(f"\n共有 {len(person_counts)} 个人")
print(f"每人样本数统计:\n{person_counts}")

# ============ 按人划分为3个Fold ============
print("\n" + "=" * 60)
print("按人划分为3个Fold...")

# 获取所有人名
all_persons = list(person_counts.index)
np.random.seed(42)  # 设置随机种子保证可复现
np.random.shuffle(all_persons)  # 随机打乱

# 将人划分为3个fold
n_persons = len(all_persons)
fold_size = n_persons // NUM_FOLDS
remainder = n_persons % NUM_FOLDS

person_to_fold = {}
start_idx = 0

for fold_idx in range(NUM_FOLDS):
    # 平均分配，余数分配给前几个fold
    end_idx = start_idx + fold_size + (1 if fold_idx < remainder else 0)
    persons_in_fold = all_persons[start_idx:end_idx]

    for person in persons_in_fold:
        person_to_fold[person] = fold_idx

    print(f"Fold {fold_idx}: {len(persons_in_fold)} 人 - {persons_in_fold}")
    start_idx = end_idx

# 给每个样本分配fold
df['fold'] = df['person'].map(person_to_fold)

# 验证划分
print("\n" + "=" * 60)
print("验证Fold划分...")
for fold_idx in range(NUM_FOLDS):
    fold_df = df[df['fold'] == fold_idx]
    persons_in_fold = fold_df['person'].unique()
    samples_in_fold = len(fold_df)
    print(f"Fold {fold_idx}: {len(persons_in_fold)} 人, {samples_in_fold} 个样本")

    # 检查每个类别的分布
    class_dist = fold_df['classes'].value_counts()
    print(f"  类别分布: {dict(class_dist)}")

# ============ 生成3个训练CSV文件 ============
print("\n" + "=" * 60)
print("生成3个训练CSV文件...")
print("每个文件对应一次训练，使用不同的test fold")
print("=" * 60)

for test_fold in range(NUM_FOLDS):
    print(f"\n【配置 {test_fold + 1}/{NUM_FOLDS}】")
    print(f"  Test Fold:  {test_fold}")
    print(f"  Train Folds: 其余{NUM_FOLDS-1}个fold")

    # 创建mode列
    df_copy = df.copy()

    def assign_mode(fold):
        if fold == test_fold:
            return 'test'
        else:
            return 'train'  # 其余所有fold都作为训练集

    df_copy['mode'] = df_copy['fold'].apply(assign_mode)

    # 统计
    mode_counts = df_copy['mode'].value_counts()
    print(f"  样本分布: Train={mode_counts.get('train', 0)}, "
          f"Test={mode_counts.get('test', 0)}")

    # 验证没有人同时出现在train和test中
    train_persons = set(df_copy[df_copy['mode'] == 'train']['person'].unique())
    test_persons = set(df_copy[df_copy['mode'] == 'test']['person'].unique())

    # 检查是否有重叠
    assert len(train_persons & test_persons) == 0, "训练集和测试集有人重叠！"

    print(f"  ✅ 验证通过：训练/测试集人员无重叠")
    print(f"     Train人数: {len(train_persons)}")
    print(f"     Test人数:  {len(test_persons)}")

    # 保存CSV（移除辅助列）
    output_file = f'{OUTPUT_PREFIX}{test_fold}_train.csv'
    df_save = df_copy.drop(columns=['person', 'fold'])  # 移除辅助列
    df_save.to_csv(output_file, index=False, encoding='GBK')
    print(f"  💾 保存到: {output_file}")

# ============ 生成一个包含所有fold信息的CSV ============
print("\n" + "=" * 60)
print("生成完整的fold信息CSV（用于记录）...")

output_all = 'dataset_3fold_person_split_all.csv'
df.to_csv(output_all, index=False, encoding='GBK')
print(f"💾 保存到: {output_all}")
print("   此文件包含所有样本的fold分配信息")

# ============ 完成 ============
print("\n" + "=" * 60)
print("🎉 3-Fold划分完成！")
print("=" * 60)
print("\n生成的文件:")
for i in range(NUM_FOLDS):
    print(f"  {i+1}. fold{i}_train.csv  - 第{i+1}次训练配置 (test_fold={i})")
print(f"  {NUM_FOLDS+1}. dataset_3fold_person_split_all.csv - 完整fold信息")

print("\n使用方法:")
print("1. 在 visual_processed.ipynb 中修改:")
print("   ids,paths,nlps,classes,modes = load_data('fold0_train.csv')")
print("2. 运行 visual_processed.ipynb 提取特征")
print("3. 运行 visual_emotion_recognition.ipynb 训练模型")
print("4. 重复上述步骤，依次使用 fold1_train.csv 和 fold2_train.csv")
print("5. 汇总3次结果，计算平均性能")

print("\n注意:")
print("- fold0_train.csv: 使用fold0作为测试集")
print("- fold1_train.csv: 使用fold1作为测试集")
print("- fold2_train.csv: 使用fold2作为测试集")
print("- 每个CSV都包含 'mode' 列，可直接用于原代码！")
