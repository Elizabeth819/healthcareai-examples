import os
import time
import concurrent.futures
from healthcareai_toolkit.clients import MedImageParseClient
from healthcareai_toolkit import settings
import matplotlib.pyplot as plt
import numpy as np
from healthcareai_toolkit.data.manip import extract_instance_masks

endpoint = settings.MIP_MODEL_ENDPOINT
print(endpoint)
client = MedImageParseClient(endpoint)

def save_image_with_mask(image: np.ndarray, mask: np.ndarray, image_path: str, start_time: float, colormap: str = "Set1", alpha: float = 0.5):
    if image.shape[:2] != mask.shape:
        raise ValueError("The dimensions of the mask do not match the dimensions of the image.")

    labels = np.unique(mask)
    labels = labels[labels > 0]  # Exclude background
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    for label in labels:
        color = cmap(norm(label))[:3]  # Get RGB values
        overlay[mask == label] = [*color, alpha]

    # 生成唯一的文件名
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(start_time))
    base_name = os.path.basename(image_path).replace('.dcm', '')
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}_{timestamp}_segmentation.png")

    # 保存可视化结果
    plt.imsave(output_path, overlay)
    print(f"Saved visualization to {output_path} (ID: {id(image)})")

def process_batch(image_paths, text_prompt):
    start_time = time.time()
    print(f"Processing {len(image_paths)} images with prompt '{text_prompt}' at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"Processing images: {', '.join([os.path.basename(path) for path in image_paths])}")

    # 读取和标准化图像
    images = [client.read_and_normalize_image(image_path) for image_path in image_paths]
    
    # 提交图像和文本提示
    results = client.submit(image_list=images, prompts=[text_prompt] * len(images))

    # 可视化分割结果
    for image, result, image_path in zip(images, results, image_paths):
        print(f"Calling visualize_segmentation_results_contour for {image_path} (ID: {id(image)})")
        visualize_segmentation_results_contour(image, [result], image_path, start_time, extract_instance=True)

    end_time = time.time()
    print(f"\nFinished processing {len(image_paths)} images at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}, duration: {end_time - start_time:.2f} seconds")

def visualize_segmentation_results_contour(image, results, image_path, start_time, extract_instance=False, colormap: str = "Set1"):
    text_features = results[0]["text_features"]
    masks = results[0]["image_features"]
    masks = (masks > 0).astype("uint8")

    print("Image shape:", image.shape)
    print("Segmentation Mask shape: ", masks.shape)
    print("Text features: ", text_features)

    if extract_instance:
        masks = [extract_instance_masks(mask) for mask in masks]

    for mask, mask_name in zip(masks, text_features):
        print(f"Calling save_image_with_mask for {image_path} (ID: {id(image)})")
        save_image_with_mask(image, mask, image_path, start_time, colormap=colormap)

def chunk_list(lst, chunk_size):
    """将列表分割成指定大小的块"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

if __name__ == '__main__':
    # 读取文件夹中的所有 .dcm 文件
    folder_path = "/Users/wanmeng/repository/healthcareai-examples/images_client"
    image_paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith('.dcm')]

    # 只处理前一百张图
    total_images = 5  # 确保 total_images 是 batch_size 的整数倍
    image_paths = image_paths[:total_images]
    
    text_prompt = "locate the abnormal part"

    # 设置批量大小和并发数量
    batch_size = 1
    max_workers = int(total_images/batch_size)  

    # 分批处理图像
    batches = list(chunk_list(image_paths, batch_size))

    # 记录开始时间
    total_start_time = time.time()

    # 使用线程池并发处理批次
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, batch, text_prompt) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing batch: {e}")

    # 记录结束时间并计算总耗时
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nTotal time taken to process {total_images} images: {total_duration:.2f} seconds: max_workers={max_workers}, batch_size={batch_size}")