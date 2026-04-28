import torch
import os
import torchvision.models as models

def load_model(model_path):
    """加载指定路径的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载模型
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 处理可能的模型包装情况
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # 如果是多GPU训练的模型，需要获取实际模型
    
    return model

def get_layer_weight_means(model, model_name):
    """计算并打印模型每层权重的均值"""
    print(f"=== {model_name} 模型每层权重均值 ===")
    print(f"{'层名称':<40} {'权重均值':<15} {'权重形状'}")
    print("-" * 80)
    
    for name, module in model.named_modules():
        # 只关注有可学习权重的层
        if hasattr(module, 'weight') and module.weight is not None:
            # 计算权重均值
            weight_mean = module.weight.data.mean().item()
            # 获取权重形状
            weight_shape = tuple(module.weight.data.shape)
            # 打印信息
            print(f"{name:<40} {weight_mean:.8f} {weight_shape}")
    
    print("\n" + "-" * 80 + "\n")

def main():
    # ==============================================
    # 在这里指定你的模型路径
    # ==============================================
    # vgg16bn_path = "path/to/your/vgg16bn_model.pth"  # VGG16BN模型路径
    # resnet18_path = "path/to/your/resnet18_model.pth"  # ResNet18模型路径
    
    try:
        # 加载模型
        print("正在加载模型...")
        vgg16bn_model = models.vgg16_bn(num_classes=1000)
        resnet18_model = models.resnet18(num_classes=1000)  
        
        # 计算并打印权重均值
        get_layer_weight_means(vgg16bn_model, "VGG16BN")
        get_layer_weight_means(resnet18_model, "ResNet18")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
