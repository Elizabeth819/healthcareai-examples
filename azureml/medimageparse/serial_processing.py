import os
import time
import matplotlib.pyplot as plt
from healthcareai_toolkit.clients import MedImageParseClient
from healthcareai_toolkit import settings

endpoint = settings.MIP_MODEL_ENDPOINT
print(endpoint)
client = MedImageParseClient(endpoint)

import numpy as np
from healthcareai_toolkit.data.manip import extract_instance_masks


def show_image_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    title=None,
    ax: plt.Axes = None,
    colormap: str = "Set1",
):
    if image.shape[:2] != mask.shape:
        raise ValueError(
            "The dimensions of the mask do not match the dimensions of the image."
        )

    labels = np.unique(mask)
    labels = labels[labels > 0]  # Exclude background
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    for label in labels:
        color = cmap(norm(label))[:3]  # Get RGB values
        overlay[mask == label] = [*color, alpha]

    show = False
    if ax is None:
        ax = plt.gca()
        show = True

    ax.imshow(image)
    ax.imshow(overlay)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()


def visualize_segmentation_results(
    image, results, prompt=None, extract_instance=False, colormap: str = "Set1"
):
    text_features = results[0]["text_features"]
    masks = results[0]["image_features"]
    masks = (masks > 0).astype("uint8")

    print("Image shape:", image.shape)
    print("Segmentation Mask shape: ", masks.shape)
    print("Text features: ", text_features)

    if extract_instance:
        masks = [extract_instance_masks(mask) for mask in masks]

    fig, axes = plt.subplots(1, len(masks), figsize=(40, 10 * len(masks)))
    if len(masks) == 1:
        axes = [axes]
    for ax, mask, mask_name in zip(axes, masks, text_features):
        show_image_with_mask(image, mask, title=mask_name, ax=ax, colormap=colormap)

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

def pixels_at_distance(mask: np.ndarray, d: int) -> np.ndarray:
    """
    Returns a new segmentation mask where pixels are exactly distance `d` from the edges of a binary mask.

    Parameters:
        mask (np.ndarray): Binary mask as a 2D NumPy array (0s and 1s).
        d (int): Distance from the edge.

    Returns:
        np.ndarray: New binary mask with pixels at distance `d` set to 1.
    """
    if d <= 0:
        raise ValueError("Distance 'd' must be greater than 0")
    if mask.dtype != bool:
        mask = mask.astype(bool)
    # Invert the mask to calculate distance to the edges
    inverted_mask = ~mask
    distance_to_edge = distance_transform_edt(inverted_mask)

    # Threshold the distance transform to get a binary mask so values are greater than 0 and less than d
    new_mask = (distance_to_edge > 0) & (distance_to_edge <= d)
    return new_mask.astype(np.uint8)


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
    print(f"Saved visualization to {output_path}")

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
        save_image_with_mask(image, mask, image_path, start_time, colormap=colormap)

def convert_dcm_to_png_path(image_path):
    base, _ = os.path.splitext(image_path)
    return base + ".png"

def process_images(image_paths, text_prompt):
    start_time = time.time()
    print(f"Processing {len(image_paths)} images with prompt '{text_prompt}' at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # 读取和标准化图像
    images = [client.read_and_normalize_image(image_path) for image_path in image_paths]
    
    # 提交图像和文本提示
    results = client.submit(image_list=images, prompts=[text_prompt] * len(images))

    # 可视化分割结果
    for image, result, image_path in zip(images, results, image_paths):
        visualize_segmentation_results_contour(image, [result], image_path, start_time, extract_instance=True)

    end_time = time.time()
    print(f"\nFinished processing {len(image_paths)} images at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}, duration: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    # 读取文件夹中的所有 .dcm 文件
    folder_path = "/Users/wanmeng/repository/healthcareai-examples/images_client"
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]

    # 测试不同批量大小的推理时间
    batch_sizes = [1, 2, 3, 4, 5]
    text_prompt = "locate the abnormal part"
    results = []

    for batch_size in batch_sizes:
        print(f"\nTesting with batch size: {batch_size}")
        
        # 指定总共处理图片数量为当前批次大小
        total_images = batch_size
        image_paths_batch = image_paths[:total_images]

        # 循环5次并取平均值
        total_durations = []
        for _ in range(3):
            # 记录开始时间
            total_start_time = time.time()

            # 处理图像
            try:
                process_images(image_paths_batch, text_prompt)
            except Exception as e:
                print(f"Error processing images: {e}")

            # 记录结束时间并计算总耗时
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time
            total_durations.append(total_duration)

        # 计算平均耗时和吞吐量
        avg_duration = sum(total_durations) / len(total_durations)
        throughput =  avg_duration / total_images
        results.append({
            "batch_size": batch_size,
            "time": avg_duration,
            "throughput": throughput
        })
        print(f"Total time taken to process {total_images} images with batch size {batch_size}: {avg_duration:.2f} seconds, Throughput: {throughput:.2f} seconds/image")

    # 可视化结果
    batch_sizes = [result["batch_size"] for result in results]
    times = [result["time"] for result in results]
    throughputs = [result["throughput"] for result in results]

    # 绘制推理时间
    plt.figure()
    plt.plot(batch_sizes, times, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Inference Time (s)")
    plt.title("Batch Size vs Inference Time")
    plt.grid()

    # 绘制吞吐量
    plt.figure()
    plt.plot(batch_sizes, throughputs, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (seconds/image)")
    plt.title("Batch Size vs Throughput")
    plt.grid()

    plt.show()