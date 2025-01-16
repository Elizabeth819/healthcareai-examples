"""
DICOM to PNG Converter

This module provides functionality to convert DICOM medical images to PNG format.

Main Features:
    - Convert single DICOM file to PNG
    - Batch convert multiple DICOM files
    - Automatic output path generation
    - Error handling and logging

Dependencies:
    - healthcareai_toolkit
    - opencv-python (cv2)
    - os

Usage:
    Single file conversion:
        converter = DicomConverter()
        converter.convert_dicom_to_png("input.dcm", "output.png")

    Batch conversion:
        converter = DicomConverter()
        for dicom_file in dicom_files:
            converter.convert_dicom_to_png(dicom_file)

Author: [Elizabeth Wan]
Date: [2025-01-15]
Version: 1.0.0
License: MIT
"""

import os
import cv2
from healthcareai_toolkit.clients import MedImageParseClient

class DicomConverter:
    def __init__(self, endpoint=None):
        """
        初始化转换器
        Args:
            endpoint: MedImageParse模型的端点URL。如果为None，将从settings中获取
        """
        self.endpoint = "/subscriptions/0d3f39ba-7349-4bd7-8122-649ff18f0a4a/resourceGroups/llama3-405B-rg/providers/Microsoft.MachineLearningServices/workspaces/llama3-h100/onlineEndpoints/medimageparse-1v100"
        self.client = MedImageParseClient(self.endpoint)

    def convert_dicom_to_png(self, input_path: str, output_path: str = None) -> None:
        """
        将DICOM文件转换为PNG格式
        Args:
            input_path: DICOM文件的路径
            output_path: 输出PNG文件的路径。如果为None，将使用输入文件名修改扩展名
        """
        try:
            # 读取和标准化DICOM图像
            image = self.client.read_and_normalize_image(input_path)

            # 如果没有指定输出路径，创建默认输出路径
            if output_path is None:
                base_name = os.path.splitext(input_path)[0]
                output_path = f"{base_name}.png"

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

            # 保存为PNG
            cv2.imwrite(output_path, image)
            print(f"成功将 {input_path} 转换为 {output_path}")

        except Exception as e:
            print(f"转换失败: {str(e)}")

def main():
    """使用示例"""
    # 初始化转换器
    converter = DicomConverter()

    # 单个文件转换示例
    dicom_file = "/Users/wanmeng/repository/healthcareai-examples/images_client/CHEST PA＿I00220866332-001.dcm"
    converter.convert_dicom_to_png(dicom_file)

    # # 批量转换示例
    # dicom_folder = "/Users/wanmeng/repository/healthcareai-examples/images_client"
    # output_folder = "/Users/wanmeng/repository/healthcareai-examples/images_client_png"
    
    # for filename in os.listdir(dicom_folder):
    #     if filename.endswith('.dcm'):
    #         input_path = os.path.join(dicom_folder, filename)
    #         output_path = os.path.join(output_folder, filename.replace('.dcm', '.png'))
    #         converter.convert_dicom_to_png(input_path, output_path)

if __name__ == "__main__":
    main()