import os
import tarfile
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_archive(archive_path, pattern_str, key_to_class, output_dir):
    """
    子进程函数：处理一个压缩包，匹配并提取目标图片
    """
    pattern = re.compile(pattern_str)
    matched_files = {}

    try:
        with tarfile.open(archive_path, 'r:*') as tar:
            for member in tar.getmembers():
                filename = os.path.basename(member.name)
                # print(filename)
                # import sys
                # sys.exit()
                basekey = os.path.splitext(filename)[0]

                match = pattern.search(basekey)
                if match:
                    matched_key = match.group(0)
                    class_id = key_to_class.get(matched_key)
                    if class_id is not None:
                        dest_path = os.path.join(output_dir, filename)
                        extracted_file = tar.extractfile(member)
                        if extracted_file:
                            with open(dest_path, 'wb') as out:
                                out.write(extracted_file.read())
                            matched_files[filename] = class_id
    except Exception as e:
        print(f"❌ 处理 {os.path.basename(archive_path)} 出错: {str(e)}")
    
    return matched_files

def extract_specific_images_multiprocess(txt_path, archive_dir, output_dir, max_workers=10):
    """
    使用多进程并行解压图片（充分利用多核CPU）
    """
    # Step 1: 加载目标图片信息
    target_keys = set()
    key_to_class = {}
    with open(txt_path, 'r') as f:
        for line in f:
            image_path, class_id = line.strip().split()
            key = os.path.splitext(os.path.basename(image_path))[0]
            print(key)
            #key=key.split('_')[1]+'_'+key.split('_')[0]
            key=key.split('_')[2]
            target_keys.add(key)
            key_to_class[key] = int(class_id)
            #break

    print(f"🎯 待提取图片数: {len(target_keys)}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: 构造正则表达式，传递给子进程（字符串形式）
    pattern_str = '|'.join(re.escape(k) for k in target_keys)
    print(pattern_str)
    # import sys
    # sys.exit()
    # print(pattern_str)
    # Step 3: 获取所有压缩包路径
    archives = [os.path.join(archive_dir, f)
                for f in os.listdir(archive_dir)
                if f.startswith('val') and (f.endswith('.tar') or f.endswith('.tar.gz'))]
    print(f"📦 压缩包数: {len(archives)}")

    # Step 4: 多进程执行
    matched_files = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_archive, path, pattern_str, key_to_class, output_dir)
                   for path in archives]
        for future in tqdm(as_completed(futures), total=len(futures), desc="多进程解压"):
            result = future.result()
            matched_files.update(result)

    # Step 5: 保存类别映射
    class_map_path = os.path.join(output_dir, 'class_map.txt')
    with open(class_map_path, 'w') as f:
        for filename, class_id in matched_files.items():
            f.write(f"{filename} {class_id}\n")

    print(f"\n✅ 总共提取: {len(matched_files)} 张图片")
    print(f"❌ 未匹配的图片数: {len(target_keys) - len(set([k for k in key_to_class if k in ''.join(matched_files.keys())]))}")
    print(f"📝 类别映射写入: {class_map_path}")
    return output_dir


# 使用示例
if __name__ == "__main__":
    # txt_path = "ImageNet_LT_val.txt"       # 替换为你的txt文件路径
    # archive_dir = "data"             # 压缩包所在目录
    # output_dir = "imagenet_val"      # 输出目录
    # extract_specific_images_multiprocess(txt_path, archive_dir, output_dir)
    txt_path = "ImageNet_LT_test.txt"       # 替换为你的txt文件路径
    archive_dir = "data"             # 压缩包所在目录
    output_dir = "imagenet_test"      # 输出目录
    extract_specific_images_multiprocess(txt_path, archive_dir, output_dir)