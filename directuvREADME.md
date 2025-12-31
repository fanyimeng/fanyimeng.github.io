# Radio Diffusion Super-Resolution

基于扩散模型的射电望远镜图像超分辨率重建系统。

## 项目概述

本项目使用条件扩散模型（Conditional Diffusion Model）从低质量的射电望远镜"dirty"图像生成高清图像。模型以PSF（点扩散函数）和原始dirty图像作为条件输入，通过扩散-去噪过程生成高质量的regrid dirty图像。

## 特性

- 🚀 基于 diffusers 库的 UNet2DModel 架构
- 🎯 支持 DDPM（高质量）和 DDIM（快速推理）采样
- 📊 完整的训练、验证和测试流程（自动划分 train/val/test）
- 🔧 灵活的 YAML 配置系统
- 📈 PSNR、SSIM 等评估指标
- 🧪 独立测试集评估（训练时自动保存 test_indices.txt）
- 🔄 **残差扩散模式**：模型学习增强残差而非完整重建
- 🎨 **混合损失函数**：MSE + L1 + SSIM 组合损失提升结构保持能力

## 项目结构

```
diffusion_superres/
├── configs/
│   └── default.yaml          # 默认配置文件
├── data/
│   ├── __init__.py
│   ├── dataset.py            # PyTorch Dataset 类
│   └── utils.py              # FITS 加载和预处理工具
├── models/
│   ├── __init__.py
│   ├── diffusion.py          # 扩散 Pipeline
│   └── unet.py               # 条件 UNet 模型
├── train.py                  # 训练脚本
├── eval.py                   # 评估脚本
└── requirements.txt          # 依赖包
```

## 数据格式

### 输入数据结构

数据目录应包含多个子文件夹，每个子文件夹包含一组数据：

```
data_dir/
├── sample_001/
│   ├── *_dirty.psf.fits      # PSF 函数
│   ├── *_dirty.image.fits    # 原始 dirty 图像
│   └── *_rg_dirty.fits       # 高清目标图像（用于训练）
├── sample_002/
│   ├── ...
```

### FITS 文件说明

| 文件类型 | 说明 | 维度 |
|---------|------|------|
| `*_dirty.psf.fits` | 点扩散函数 | 96×96 (squeeze from 1×96×96) |
| `*_dirty.image.fits` | 原始 dirty 图像 | 96×96 (squeeze from 1×96×96) |
| `*_rg_dirty.fits` | 高清 regrid dirty 图像 | 96×96 |

## 安装

```bash
# 克隆项目
cd diffusion_superres

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

使用默认配置训练：

```bash
python train.py --config configs/default.yaml
```

指定数据目录：

```bash
python train.py --config configs/default.yaml --data_dir /path/to/simobs
```

从检查点恢复训练：

```bash
python train.py --config configs/default.yaml --resume outputs/20231229_120000/latest.pt
```

训练完成后，输出目录会包含：
- `best.pt` - 最佳验证损失的模型检查点
- `latest.pt` - 最新的模型检查点
- `epoch_XXXX.pt` - 周期性保存的检查点
- `config.yaml` - 训练时使用的配置
- `test_indices.txt` - 测试集样本名列表（用于独立评估）

### 2. 评估模型

**评估所有数据：**

```bash
python eval.py --checkpoint outputs/best.pt --data_dir /path/to/simobs --save_plots
```

**仅评估测试集（推荐）：**

```bash
python eval.py --checkpoint outputs/<timestamp>/best.pt --data_dir ../directuv_data/simobs --test_indices outputs/<timestamp>/test_indices.txt --output_dir test_results --save_plots
```

**完整评估（保存 FITS 和可视化）：**

```bash
python eval.py \
    --checkpoint outputs/best.pt \
    --data_dir /path/to/simobs \
    --test_indices outputs/<timestamp>/test_indices.txt \
    --output_dir results \
    --save_fits \
    --save_plots \
    --num_inference_steps 50
```

### 3. 配置说明

`configs/default.yaml` 中的主要配置项（针对 RTX 4070 Ti Super 16GB 优化）：

```yaml
data:
  data_dir: "../directuv_data/simobs"  # 数据目录
  val_ratio: 0.1                        # 验证集比例
  test_ratio: 0.1                       # 测试集比例（独立保留）
  augment: true                         # 数据增强
  noise_sigma: 0.005                    # 噪声增强强度
  num_workers: 4                        # 数据加载线程数

model:
  type: "standard"                      # 模型类型（standard 更大，适合大数据集）
  sample_size: 96                       # 图像尺寸
  num_train_timesteps: 1000             # 扩散步数
  beta_schedule: "scaled_linear"        # Beta 调度方式
  prediction_type: "epsilon"            # 预测类型
  residual_mode: true                   # 残差扩散模式（推荐开启）
  loss_type: "hybrid"                   # 损失函数：mse/l1/hybrid

training:
  num_epochs: 300                       # 训练轮数
  batch_size: 32                        # 批次大小
  learning_rate: 2.0e-4                 # 学习率
  save_interval: 25                     # 检查点保存间隔
```

### 4. 数据集划分

训练时数据会自动划分为三部分：
- **训练集 (80%)**：用于模型训练
- **验证集 (10%)**：用于训练过程中监控损失
- **测试集 (10%)**：完全独立，仅用于最终评估

测试集样本名会保存到 `test_indices.txt`，确保评估时使用的是训练过程中完全未见过的数据。

## 模型架构

### 条件扩散模型

本项目使用条件扩散模型进行图像生成：

1. **输入通道**：3通道
   - 1通道：噪声/当前去噪状态
   - 2通道：条件（PSF + dirty 图像）

2. **输出通道**：1通道（预测噪声或干净图像）

3. **架构**：基于 diffusers 的 UNet2DModel
   - Lightweight 版本：[32, 64, 128, 256] 通道
   - Standard 版本：[64, 128, 256, 512] 通道

### 残差扩散模式（推荐）

当 `residual_mode: true` 时，模型学习预测残差而非完整目标：

```
Residual = Target (rg_dirty) - Dirty Image
```

**优势**：
- 模型只需学习"增强"部分，任务更简单
- 保留 dirty 图像已有的结构信息
- 通常能获得更高的 PSNR 和 SSIM

**推理时**：
```
Output = Predicted_Residual + Dirty_Image
```

### 混合损失函数

当 `loss_type: "hybrid"` 时，使用组合损失：

```
Loss = 0.5 × MSE + 0.3 × L1 + 0.2 × SSIM_Loss
```

| 损失类型 | 作用 |
|---------|------|
| MSE | 整体像素级重建精度 |
| L1 | 增强边缘锐度，减少模糊 |
| SSIM | 保持结构相似性 |

支持的 loss_type：
- `"mse"` - 仅 MSE 损失（默认）
- `"l1"` - 仅 L1 损失
- `"hybrid"` - MSE + L1 + SSIM 组合（推荐）

### 训练流程

**标准模式** (residual_mode: false)：
```
[Clean Image] ---> [Add Noise] ---> [Noisy Image]
                        |                 |
                        v                 v
                   [Timestep]      [Condition: PSF + Dirty]
                        |                 |
                        +--------+--------+
                                 |
                                 v
                            [UNet] ---> [Predicted Noise]
                                 |
                                 v
                          [Hybrid Loss with Target Noise]
```

**残差模式** (residual_mode: true)：
```
[Residual = Clean - Dirty] ---> [Add Noise] ---> [Noisy Residual]
                                      |                 |
                                      v                 v
                                 [Timestep]      [Condition: PSF + Dirty]
                                      |                 |
                                      +--------+--------+
                                               |
                                               v
                                          [UNet] ---> [Predicted Noise]
                                               |
                                               v
                                        [Hybrid Loss with Target Noise]
```

### 推理流程

**标准模式**：
```
[Random Noise] ---> [DDIM Denoise] ---> [Generated Image]
                          ^
                          |
                   [Condition: PSF + Dirty]
```

**残差模式**：
```
[Random Noise] ---> [DDIM Denoise] ---> [Predicted Residual] ---> [+ Dirty] ---> [Output]
                          ^
                          |
                   [Condition: PSF + Dirty]
```

## 评估指标

| 指标 | 说明 |
|------|------|
| PSNR | 峰值信噪比，越高越好 |
| SSIM | 结构相似性，越接近1越好 |
| MSE | 均方误差，越低越好 |
| MAE | 平均绝对误差，越低越好 |

## 注意事项

1. **数据量**：当前数据集约2000+对样本，推荐使用 `standard` 模型充分利用数据
2. **归一化**：训练时图像归一化到 [0, 1]，PSF 归一化为 sum=1
3. **GPU 内存**：
   - `lightweight` 模型：4GB VRAM 即可
   - `standard` 模型 + batch_size=32：需要 12-16GB VRAM（推荐 RTX 4070 Ti Super）
4. **推理速度**：DDIM 50步约需1-2秒/样本（GPU）
5. **测试集独立性**：始终使用 `--test_indices` 参数评估，确保测试数据未参与训练

## 常见问题

**Q: 如何增加训练数据？**

A: 将新数据按相同结构放入数据目录即可自动识别。

**Q: 模型过拟合怎么办？**

A:
- 启用数据增强：`augment: true`
- 减小模型：使用 `lightweight` 类型
- 添加噪声：增大 `noise_sigma`

**Q: 推理太慢？**

A: 使用 DDIM 并减少步数：`--num_inference_steps 20`

**Q: 如何使用独立测试集评估？**

A: 训练时会自动保存 `test_indices.txt`，评估时指定该文件：
```bash
python eval.py --checkpoint best.pt --data_dir ../directuv_data/simobs \
    --test_indices outputs/<timestamp>/test_indices.txt --save_plots
```

## 引用

如果使用本项目，请引用：

```bibtex
@software{radio_diffusion_superres,
  title = {Radio Diffusion Super-Resolution},
  year = {2024},
  description = {Diffusion model for radio telescope image super-resolution}
}
```

## 许可证

MIT License