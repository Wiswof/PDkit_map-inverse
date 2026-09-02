# GLP-1R RDKit–PCA–PyTorch QSAR

这个项目使用 RDKit 从 SMILES 计算分子描述符，并训练一个用于预测 GLP-1R Assay EC50 的 PyTorch 回归模型。项目还包含一个 notebook，用于单分子预测、查看全部 RDKit 描述符以及计算预测结果对描述符的反向传播梯度。

## 模型流程

```text
SMILES
  → RDKit Mol
  → 216维原始 RDKit 描述符
  → 缺失值填补 + 标准化 + 冻结 PCA（合并为模型第一层）
  → Linear(PCA维数, 16)
  → SiLU
  → Linear(16, 1)
  → 标准化 pEC50
  → 还原 pEC50
  → EC50 (nM)
```

模型使用的活性转换为：

```text
pEC50 = 9 - log10(EC50_nM)
EC50_nM = 10^(9 - pEC50)
```

输入描述符的标准化和 PCA 已经折叠进冻结的第一个 PyTorch `Linear` 层。因此推理时直接输入按训练顺序排列的 216 维原始 RDKit 描述符，不需要额外调用 `StandardScaler.transform()` 或 `PCA.transform()`。

## 主要文件

- `train_glp1r_pytorch.py`：数据清洗、RDKit 描述符计算、骨架交叉验证、模型训练及模型保存。
- `glp1r_ec50_inverse_design_demo.ipynb`：加载最终模型、计算分子描述符、预测 EC50 和计算描述符梯度。
- `Gen2_Table3_molecular_formula_with_SMILES.csv`：训练数据。
- `model_output_torch_pca_fin/`：当前最终模型及训练结果。
- `requirements.txt`：已验证环境的核心 Python 依赖版本。

## 安装环境

推荐使用 Python 3.11：

```bash
conda create -n glp1r_qsar python=3.11 -y
conda activate glp1r_qsar
cd "/Users/wiswolf/Desktop/PDkitmap script"
python -m pip install -r requirements.txt
```

为 VS Code/Jupyter 注册 kernel：

```bash
python -m ipykernel install --user \
  --name glp1r_qsar \
  --display-name "Python (glp1r_qsar)"
```

然后在 VS Code notebook 右上角选择 `Python (glp1r_qsar)`。

## 训练模型

训练 Assay 1 的最终 SiLU/PCA 模型：

```bash
conda activate glp1r_qsar
cd "/Users/wiswolf/Desktop/PDkitmap script"

python train_glp1r_pytorch.py \
  Gen2_Table3_molecular_formula_with_SMILES.csv \
  --assay 1 \
  --epochs 20 \
  --batch-size 16 \
  --pca-variance 0.95 \
  --output-dir model_output_torch_pca_fin
```

关键参数：

- `--assay`：选择 Assay 1 或 Assay 2。
- `--epochs`：训练轮数，默认 20。
- `--batch-size`：随机 mini-batch 大小，默认 16；每个 epoch 重新打乱训练样本。
- `--pca-variance`：PCA 保留的累计方差比例，默认 0.95。
- `--hidden`：隐藏层宽度，默认 16。
- `--lr`：AdamW 学习率，默认 0.003。
- `--weight-decay`：AdamW 权重衰减，默认 0.01。
- `--seed`：随机种子，默认 42。

训练时终端会输出 PCA 前后的维数，例如：

```text
PCA dimension: 216 -> 25 (retained variance=95.0%)
```

## 数据处理与验证

训练脚本会：

1. 解析精确的 EC50 数值，排除 `>`、`<`、`~` 等截断值；
2. 将 SMILES 规范化；
3. 排除同一规范结构中 pEC50 差异超过阈值的冲突记录；
4. 合并重复结构；
5. 使用 Bemis–Murcko scaffold 分组的 5-fold `GroupKFold` 计算 out-of-fold 指标；
6. 使用全部清洗数据训练并保存最终模型。

输出目录包含：

- `*_model.pt`：PyTorch checkpoint，包括冻结 PCA 第一层、SiLU 网络权重、描述符顺序和目标还原参数。
- `*_metrics.json`：MAE、RMSE、R²、PCA 维数和训练配置。
- `*_training.png`：训练/验证 loss 曲线和实测–预测散点图。
- `*_history.csv`：每个 fold、每个 epoch 的 loss。
- `*_oof_predictions.csv`：骨架交叉验证的 out-of-fold 预测。
- `*_conflicts.csv`：被冲突过滤规则排除的记录。
- `*_preprocessor.joblib`：供检查或复现实验使用的 sklearn 预处理器；notebook 推理不依赖它。

## Notebook 使用

打开：

```text
glp1r_ec50_inverse_design_demo.ipynb
```

notebook 优先加载：

```text
model_output_torch_pca_fin/glp1r_assay1_torch_model.pt
```

核心接口分为两步：

```python
mol = Chem.MolFromSmiles(SMILES)
raw_descriptors = calculate_descriptor_vector(mol)
pEC50, EC50_nM = predict_rdkit_vector(raw_descriptors)
```

- `calculate_descriptor_vector(mol)`：依次调用训练时相同顺序的 216 个 RDKit 描述符函数。
- `predict_rdkit_vector(rdkit_vector)`：只接收一个 216 维原始描述符向量，返回预测 pEC50 和 EC50 (nM)。
- `descriptor_table`：将描述符名称与数值组合成便于阅读的 Pandas 表格，不参与模型预测。

## 梯度说明

notebook 计算：

```text
dpEC50_dDescriptor = ∂pEC50 / ∂原始描述符
```

EC50 的原始单位导数通过链式法则得到：

```text
∂EC50/∂x = -ln(10) × EC50 × ∂pEC50/∂x
```

将原始导数乘以训练集描述符标准差，表示描述符变化一个训练集标准差时的局部预测变化。这适合跨描述符比较，但不是原始单位导数。

## 重要限制

- 当前清洗后数据量较小，约 77 个唯一结构。
- 当前 `model_output_torch_pca_fin` 的 scaffold cross-validation R² 仍为负，说明模型对新骨架的泛化能力不足。
- 描述符梯度是模型的局部敏感度，不代表因果关系或可实现的结构修改。
- SMILES 和分子图是离散变量，不能把描述符空间的连续梯度结果直接转换成保证有效的分子。
- 该模型适合流程演示、假设生成和候选排序，不应替代真实实验验证。
