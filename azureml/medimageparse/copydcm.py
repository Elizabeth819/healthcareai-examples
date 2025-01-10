import os
import shutil

def copy_dcm_file(original_file, destination_folder, num_copies):
    base_name = os.path.basename(original_file).replace('.dcm', '')
    for i in range(1, num_copies + 1):
        new_file_name = f"{base_name}-{i:03d}.dcm"
        new_file_path = os.path.join(destination_folder, new_file_name)
        shutil.copyfile(original_file, new_file_path)
        print(f"Copied to {new_file_path}")

# 原始DICOM文件路径
original_file = "/Users/wanmeng/repository/healthcareai-examples/images_client/CHEST PA＿I00220866332.dcm"
# 目标文件夹
destination_folder = "/Users/wanmeng/repository/healthcareai-examples/images_client"
# 复制次数
num_copies = 100

# 执行复制操作
copy_dcm_file(original_file, destination_folder, num_copies)