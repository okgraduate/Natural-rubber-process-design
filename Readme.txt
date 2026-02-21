# 天然橡胶初加工工艺设计系统

本仓库包含一个用于**天然橡胶初加工工艺设计与反向推荐**的 Python 脚本，通过对实验数据的建模，实现：

- 从三种凝固方式（acid / enzyme / natural）的指标表构建统一数据集  
- 计算塑性指标 *plasticity*（作为 6 个指标之一）  
- 使用类 NNI 的方式对每种凝固方式做数据增强（约 3000 条样本）  
- 训练 BP 神经网络完成 **6 指标 → 凝固方式 (acid / enzyme / natural) 三分类**  
- 基于增强后的数据，进行**反向工艺推荐**，给出终点状态样本（dry / package）及完整工艺路径示例  

---

## 功能概述

脚本主要实现以下功能（与源码开头注释一致）：

1. **解析数据**  
   - 解析 acid / enzyme / natural 三个 indicator 表，构建 `base_df`  
   - 指标包括：  
     - `nitrogen`（氮含量）  
     - `ash`（灰分含量）  
     - `volatiles`（挥发分含量）  
     - `impurity`（杂质含量）  
     - `mooney`（门尼粘度）  
     - `p0` / `pri`  

2. **计算塑性指标 `plasticity`**  
   - 定义为：`plasticity = 0.5 * P0 + 0.5 * PRI`  
   - 并计算对应的标准差 `plasticity_sd`  
   - 作为 6 指标之一，和其他 5 个指标完全对等使用  

3. **标记烘干类型 `dry_type`**  
   - acid / natural 的 `dry` → `pellet_drying`（Pellet drying，造粒烘干）  
   - enzyme 的 `dry` → `sheet_drying`（Sheet drying，挂片烘干）  

4. **NNI 风格数据增强**  
   - 对每个 `coag_type`（acid / enzyme / natural）分别进行插值 + 噪声增强  
   - 目标是每类约 `target_per_class=1000` 条样本（默认）  
   - 得到增强后的 `aug_df`，作为 BP 网络训练数据  

5. **训练 BP 神经网络**  
   - 模型：`BPClassifier`，两层全连接 + ReLU  
   - 输入：6 项指标 `[nitrogen, ash, volatiles, impurity, mooney, plasticity]`  
   - 输出：3 类凝固方式（acid / enzyme / natural）  
   - 使用 `StandardScaler` 对输入做标准化  
   - 使用 `train/val/test` 划分评估模型精度（打印 Test Accuracy）  

6. **模型及数据保存**  
   - 训练完成后，保存以下文件到当前目录：  
     - `bp_model.pt`：BP 神经网络权重（PyTorch）  
     - `scaler.npz`：标准化器参数（mean, scale）  
     - `label_map.npz`：凝固方式标签映射（acid / enzyme / natural）  
     - `aug_data_with_drytype.pkl`：完整增强 DataFrame（包含 dry_type），用于反向推荐  

7. **交互模式：工艺阶段性预测 + 反向工艺推荐**  
   - 运行脚本后选择模式 `2` 即可进入交互式推荐  
   - 输入 6 项指标（空格分隔）：  
     - 氮含量 `nitrogen`  
     - 灰分含量 `ash`  
     - 挥发分含量 `volatiles`  
     - 杂质含量 `impurity`  
     - 门尼粘度 `mooney`  
     - 塑性初值 `plasticity`  
   - 输出顺序：  
     1. **凝固方式概率**：acid / enzyme / natural 的预测概率  
     2. **推断原料路径**：  
        - acid / enzyme → `latex`  
        - natural → `glue`  
     3. **推荐凝固方式**（概率最高的一类）  
     4. **推荐破碎和烘干方式**（英文）：  
        - enzyme       → `Sheet drying`（挂片烘干）  
        - acid/natural → `Pellet drying`（造粒烘干）  
     5. 在对应凝固方式下，选取 **终点状态样本**（`dry` 或 `package`）：  
        - 计算与目标 6 指标的欧氏距离  
        - 找到与目标最接近的 5 条样本  
        - 输出推荐表，包含：  
          - 工艺阶段 `process_step`  
          - 烘干类型 `dry_type`  
          - 6 指标值  
          - 综合相对误差 `RelErr`（6 项相对误差的平均）  
     6. 给出一条**示例完整工艺路径**，方便论文说明  

---

## 环境与依赖

建议使用 Python 3.9+（3.8 以上一般都可以）。  
安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 示例内容：

```text
numpy
pandas
scikit-learn
torch
```

---

## 使用方法

### 1. 训练模型并生成增强数据

首次使用时，先运行训练模式：

```bash
python your_script_name.py
```

进入后根据提示输入：

```text
选择模式：
  1 = 训练模型并保存增强数据
  2 = 进入交互式反向工艺推荐
请输入 1 或 2：1
```

脚本会：

- 构建基础数据 `base_df`  
- 进行 NNI 数据增强，生成 `aug_df`  
- 训练 BP 神经网络，并在终端打印训练/验证/Test 精度  
- 在当前目录生成：

  - `bp_model.pt`  
  - `scaler.npz`  
  - `label_map.npz`  
  - `aug_data_with_drytype.pkl`  

### 2. 交互式反向工艺推荐

训练完成后，再次运行脚本：

```bash
python your_script_name.py
```

这次输入：

```text
请输入 1 或 2：2
```

随后按提示输入 6 项指标（用空格分隔），例如：

```text
0.25 0.30 0.50 0.20 60 50
```

程序会输出：

- 各凝固方式的预测概率  
- 推荐的凝固方式  
- 推断的原料路径（latex 或 glue）  
- 推荐的破碎与烘干方式（Sheet drying / Pellet drying）  
- 与目标指标最接近的 5 条终点样本及综合相对误差  
- 一条示例完整工艺路径  

输入 `q` / `quit` / `exit` 可退出交互模式。

---

## 文件说明

- `your_script_name.py`  
  主脚本，包含数据解析、增强、模型训练和交互式推荐的全部逻辑。

- `bp_model.pt`  
  训练后的 BP 神经网络参数（PyTorch `state_dict`）。

- `scaler.npz`  
  训练时使用的 `StandardScaler` 参数（`mean`, `scale`），用于对输入指标做同样的标准化。

- `label_map.npz`  
  凝固方式类别与索引的映射，例如 `["acid", "enzyme", "natural"]`。

- `aug_data_with_drytype.pkl`  
  经过 NNI 数据增强后的完整 DataFrame，包含：  
  `coag_type`, `process_step`, `dry_type` 以及 6 指标和对应标准差。

---

## 许可证

你可以在 GitHub 仓库中添加一个开源许可证（例如 MIT）。  
示例（MIT License）可在创建仓库时直接通过 GitHub 自动生成。