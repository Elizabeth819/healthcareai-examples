# healthcareai-examples

此repo指导如何将DICOM文件转为padded 1024*1024 png图片, 在Azure Machine Learning (AML)的model catalog部署图像分割模型和诊断报告生成模型medImageParse and CXRReportGen, 并创建realtime endpoint,生成CXR诊断报告和胸部等CT的异常病理图像识别和分割.

上述创建的endpoint可以在本repo的notebook里代码方式使用, AML没有playground.

如果直接在Azure Foundry Portal(AI Studio) model catalog里创建endpoint, 则可以直接在AI Foundry Healthcare Playground里UI方式使用. AML和AI Foundry不能互相引用.

三个主要功能:
1.将DICOM文件转为符合微软healthcareai图片分割模型medimagaparse所必须的padded 1024*1024 png图片格式.
![image](https://github.com/user-attachments/assets/3d7333ad-d259-4356-bf4e-8b166788e698)

2.使用medimageparse模型对胸部CT进行异常病理图像识别分割, 用户输入病理位置以及异常定位的text Prompt, 结果以mask方式展示.
![image](https://github.com/user-attachments/assets/131e0fc8-fa56-4372-a476-462ae6e15d1a)

3.可以进一步对异常区域的边缘可视化
![image](https://github.com/user-attachments/assets/893bf5ab-c26a-4a24-b804-8d6dff59fd14)

代码执行:

0. 安装az命令, 用于登录azure账号, 下载windows/ubuntu/mac版本的Azure CLI, 选择带有Powershell的windows安装程序MSI  
https://learn.microsoft.com/zh-cn/cli/azure/install-azure-cli  
通过Azure CLI登陆到Azure账号
```
az login
```

1. 代码包安装  
   创建一个新的conda环境,  python>=3.9,<3.12, 推荐3.10
   执行package包的安装:   pip install -e package
   
3. 配置环境  
  在AML model catelog里deploy medImageParse, CXRReportGen.部署所需时间40分钟左右. 
4. 设置自己的env  
.env: MIP_MODEL_ENDPOINT格式如下: 
Example format: "/subscriptions/{subscription-id}/resourceGroups/{resource-group}/providers/Microsoft.MachineLearningServices/workspaces/{workspace-name}/onlineEndpoints/{endpoint-name}"
MIP_MODEL_ENDPOINT = "/subscriptions/0d3f39ba-7349-4bd7-8122-649ff18f0a4a/resourceGroups/wanmeng-healthcare/providers/Microsoft.MachineLearningServices/workspaces/wanmeng-7491/onlineEndpoints/wanmeng-7491-yyfba"  
resource-group和workspace-name可以在打开模型的URL里找到.

5. 下载config.json, 从AML workspace首页或AI Studio的project overview首页都可以找到config的下载  
   config放在项目根目录下面, 会在建立ml_client的时候从本地读取, 否则会与.env认证不符.  
   <img width="609" alt="image" src="https://github.com/user-attachments/assets/5b70e0fe-fae5-4a2d-8f5d-669afa778775" />

7. 执行图像分割notebook: 
   /Users/wanmeng/repository/healthcareai-examples/azureml/medimageparse/medimageparse_segmentation_demo.ipynb  
9. 执行CXR报告稿生成: 
   /Users/wanmeng/repository/healthcareai-examples/azureml/cxrreportgen/cxr-deploy-aml.ipynb  
![image](https://github.com/user-attachments/assets/21f7cf8e-257a-43fd-bade-aed9d817a08b)
