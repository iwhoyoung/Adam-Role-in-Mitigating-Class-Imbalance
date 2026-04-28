import pandas as pd
import os
import glob

def process_cifar_csvs_in_directory(directory_path):
    # 构造匹配指定目录下所有包含 'cifar' 的 .csv 文件路径
    csv_files = glob.glob(os.path.join(directory_path, "*cifar*.csv"))

    # 构造新的列名顺序（共200列）
    train_names = [f"class_{i}_r-10_train" for i in range(100)]
    test_names = [f"class_{i}_r-10_test" for i in range(100)]
    new_names = train_names + test_names

    for file in csv_files:
        if 'renew' in file:
            continue
        try:
            df = pd.read_csv(file)
            original_columns = list(df.columns)

            if len(original_columns) < 200:
                print(f"⚠️ 跳过文件（列数不足200）：{file}")
                continue

            # 替换前200列列名
            df.columns = [original_columns[0]] + new_names + original_columns[201:]


            # 构造新文件名（与原文件同目录）
            base, ext = os.path.splitext(file)
            new_filename = f"{base}_renew.csv"

            # 保存修改后的文件
            df.to_csv(new_filename, index=False)
            print(f"✅ 成功处理：{file} → {new_filename}")

        except Exception as e:
            print(f"❌ 错误处理文件 {file}：{e}")

# === 用法：替换下面路径为你要处理的目录路径 ===
process_cifar_csvs_in_directory("log")
