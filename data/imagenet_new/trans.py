with open('label/ImageNet_LT_test.txt', 'r', encoding='utf-8') as source_file:
    lines = source_file.readlines()    
    # 写入目标文件
    with open('label/ImageNet_LT_test_new.txt', 'w', encoding='utf-8') as dest_file:
        for line in lines:
            dest_file.write(line.split('/')[-1])