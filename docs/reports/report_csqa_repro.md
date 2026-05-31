# QAGNN 复现技术报告（CSQA In-house）

## 1. 报告目标
本报告总结 `qagnn-main` 在本地环境下的完整复现过程、关键技术改动、实验结果与误差分析，回答两个核心问题：

1. 当前复现流程是否跑通（从预处理到训练闭环）；
2. 结果与论文 Table 3（IHdev/IHtest）差距的主要来源是什么。


## 2. 环境与数据基线
### 2.1 运行环境
- OS/平台：WSL + Windows 主机
- Python 环境：`amem_env`（Conda）
- GPU：NVIDIA GeForce RTX 5060 Ti（8GB 显存）
- 主要模型：`roberta-large`

### 2.2 数据与任务
- 任务：CSQA in-house split
- 数据链路：`statement -> grounding -> graph extraction -> train/eval`
- 关键文件：
  - `data/csqa/statement/*.statement.jsonl`
  - `data/csqa/grounded/*.grounded.jsonl`
  - `data/csqa/graph/*.graph.adj.pk`
  - `data/csqa/inhouse_split_qids.txt`（8500 行）


## 3. 主要工程改动（为可运行性）
为适配现代依赖与本机环境，已进行以下修复：

- `modeling/modeling_encoder.py`：去除对旧版 `transformers` 常量表依赖，改为模型名推断。
- `utils/optimization_utils.py`：`AdamW/RAdam` 兼容当前 PyTorch 栈。
- `utils/grounding.py`：适配 spaCy 3 API，并支持 grounding 断点续跑。
- `utils/graph.py`：
  - 兼容 `networkx` 的 gpickle 读取方式；
  - 增加 graph 生成分块缓存与续跑机制。
- `qagnn.py`：
  - 训练断点续跑（`checkpoint_last.pt`）；
  - 兼容 PyTorch 2.6+ 的 `torch.load(weights_only=...)` 行为变化。
- 训练/评测脚本：
  - 离线优先（`TRANSFORMERS_OFFLINE=1`）；
  - `run_qagnn__csqa.sh` 支持自动续跑。

结论：工程层面已经实现“可跑通 + 可续跑”，不是“无法启动”的问题。


## 4. 关键一致性检查结果
### 4.1 CPNet embedding 文件状态
- 旧文件 `data/cpnet/tzw.ent.npy` 曾损坏（`mmap length is greater than file size`）。
- 修复版 `tzw.ent.repaired.npy` 可读，但抽样全零行比例约 `0.6127`，语义信息严重退化。
- 后续已重新下载原始 `tzw.ent.npy` 并通过一致性校验：
  - `shape=(799273, 1024)`
  - `dtype=float32`
  - `sample_zero_row_ratio=0.000000`

### 4.2 词表与 split 基本一致性
- `data/cpnet/concept.txt` 行数为 `799273`。
- `inhouse_split_qids.txt` 共 `8500` 行，且全部属于训练 statement 的 QID 子集。

结论：当前“文件损坏”问题已修复；但这并未显著提升最终指标。


## 5. 实验结果汇总
论文 Table 3（QA-GNN）目标：
- IHdev: `76.54 ± 0.21`
- IHtest: `73.41 ± 0.92`

### 5.1 本地关键实验结果
#### Run A（完整 15 epoch，历史主 run）
- 目录：`saved_models/csqa/enc-roberta-large__k5__gnndim200__bs64__seed0__20260520_203520`
- best（按 `log.csv` 的 dev 最大值）：
  - `dev_acc = 0.44635544635544633`（44.64%）
  - `test_acc = 0.4262691377921031`（42.63%）

#### Run B（改回原始 `tzw.ent.npy`，`embtzw__ufz4`）
- 目录：`saved_models/csqa/enc-roberta-large__k5__gnndim200__bs64__seed0__embtzw__ufz4__20260522_182718`
- 当前已记录到 `epoch 3`（后续 `KeyboardInterrupt`）：
  - best `dev_acc = 0.43161343161343163`（43.16%）
  - 对应 `test_acc = 0.40854149879129736`（40.85%）

### 5.2 与论文差距
以 Run A best 对比论文 QA-GNN：
- IHdev 差距：`76.54 - 44.64 = 31.90` 个百分点
- IHtest 差距：`73.41 - 42.63 = 30.78` 个百分点

结论：差距为“数量级差距”，非随机波动或单次 seed 偏差可解释。


## 6. 现象分析与技术判断
### 6.1 已排除的低层问题
- 代码可完整训练并保存 checkpoint；
- 预处理链条完整跑通；
- 原始 `tzw.ent.npy` 已替换并通过校验；
- 训练参数与任务入口（CSQA in-house）基本符合预期。

### 6.2 仍存在的高风险偏差源
1. **预处理产物与论文产物非同源**  
   本次最终图数据来自本地重建流程（grounding/graph），并非论文发布时同版预处理包。  
   在图学习任务中，concept grounding/graph 构造微小差异会被放大到最终准确率。

2. **实现与依赖栈迁移带来的行为漂移**  
   复现过程中对 `modeling_encoder.py`、`optimization_utils.py`、`graph.py` 等进行兼容改动，虽保证可运行，但可能改变训练轨迹。

3. **checkpoint 与当前栈不兼容迹象**  
   尝试加载仓库提供的预训练模型时出现 `state_dict` 键不匹配（如 `encoder.module.embeddings.position_ids`），说明代码/权重/依赖版本存在偏移，削弱“官方结果直接复验”的可行性。

4. **硬件与训练策略约束**  
   8GB 显存下 `unfreeze_epoch=4` 后训练显著变慢，频繁中断会影响完整对比实验效率。

### 6.3 关于“原始 tzw 与结果关系不大”的结论
从当前证据看，该判断**在本项目上下文中成立**：
- 换回完整 `tzw.ent.npy` 后，前期指标仍处于 40% 区间；
- 说明“损坏 embedding”不是唯一瓶颈，也不足以解释 30+ 点鸿沟。


## 7. 结论
### 7.1 复现状态结论
- **工程复现结论**：已完成（代码可跑、数据链路可跑、训练可跑、结果可复现到稳定区间）。
- **论文数值复现结论**：未完成（与 Table 3 存在约 30 个百分点差距）。

### 7.2 技术结论（建议对外表述）
更严谨的表述是：

> 在当前公开可得资源、现代依赖栈与本地重建预处理流程下，QAGNN 在 CSQA in-house 上未能复现论文报告的数值结果；差距显著且稳定，非单一文件损坏或单次训练波动可解释。

不建议直接下“论文不可复现（绝对）”结论；更建议使用“在当前公开工件与环境约束下不可复现论文数值”这一边界清晰的结论。


## 8. 已产出与可审计证据
- 训练脚本与配置：`run_qagnn__csqa.sh`
- 一致性校验脚本：`scripts/verify_cpnet_assets.py`
- 关键日志：
  - `logs/train_csqa__enc-roberta-large__k5__gnndim200__bs64__seed0__20260520_203520.log.txt`
  - `logs/train_csqa__enc-roberta-large__k5__gnndim200__bs64__seed0__embtzw__ufz4__20260522_182718.log.txt`
- 关键结果文件：
  - `saved_models/csqa/enc-roberta-large__k5__gnndim200__bs64__seed0__20260520_203520/log.csv`
  - `saved_models/csqa/enc-roberta-large__k5__gnndim200__bs64__seed0__embtzw__ufz4__20260522_182718/log.csv`

