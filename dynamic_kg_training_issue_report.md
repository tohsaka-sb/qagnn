# 动态图训练问题报告（CSQA / QAGNN）

## 1. 结论摘要
本轮“动态/即时 KG”训练结果显著劣于原始 QA-GNN（静态 ConceptNet 子图）。  
核心结论是：**当前动态构图在 train split 上大面积退化为极稀疏图（几乎单边图）**，导致模型几乎学不到有效图推理信号，最终 `dev/test` 准确率落在 `~0.20` 附近。

## 2. 对比对象与证据路径
- 原始 QA-GNN（baseline）:
  - 模型目录: [enc-roberta-large__k5__gnndim200__bs64__seed0__20260520_203520](/mnt/d/Users/Administrator/Desktop/qagnn-main/saved_models/csqa/enc-roberta-large__k5__gnndim200__bs64__seed0__20260520_203520)
  - bad case 分析: [bad_cases_csqa_base_e10](/mnt/d/Users/Administrator/Desktop/qagnn-main/analysis/bad_cases_csqa_base_e10)
- 动态图 QA-GNN（dynamic）:
  - 模型目录: [enc-roberta-large__k5__gnndim200__bs64__seed0__embtzw__ufz4__g_dynamic__20260530_231437](/mnt/d/Users/Administrator/Desktop/qagnn-main/saved_models/csqa/enc-roberta-large__k5__gnndim200__bs64__seed0__embtzw__ufz4__g_dynamic__20260530_231437)
  - bad case 分析: [bad_cases_csqa_dynamic_e3](/mnt/d/Users/Administrator/Desktop/qagnn-main/analysis/bad_cases_csqa_dynamic_e3)

## 3. 指标对比（同一 in-house test 集）
- baseline 最佳 `dev_acc`: `0.4464`（step 1330）
- baseline 最佳 `test_acc`: `0.4279`（step 1729）
- dynamic 最佳 `dev_acc/test_acc`: `0.2351 / 0.2039`（epoch 3, step 532）

对应 summary:
- baseline: [summary.json](/mnt/d/Users/Administrator/Desktop/qagnn-main/analysis/bad_cases_csqa_base_e10/summary.json)
- dynamic: [summary.json](/mnt/d/Users/Administrator/Desktop/qagnn-main/analysis/bad_cases_csqa_dynamic_e3/summary.json)

## 4. Bad Case 对比结果
- 评测样本数（共同 qid）: `1241`
- baseline 准确率: `0.4013`
- dynamic 准确率: `0.2039`
- 从 baseline 到 dynamic 的**退化样本**（原对现错）: `380`
- 从 baseline 到 dynamic 的**改进样本**（原错现对）: `135`
- 双方都错: `608`

对比文件:
- 总结: [dynamic_vs_base_compare_summary.json](/mnt/d/Users/Administrator/Desktop/qagnn-main/analysis/dynamic_vs_base_compare_summary.json)
- 典型退化样本（Top20）: [dynamic_vs_base_regressions_top20.csv](/mnt/d/Users/Administrator/Desktop/qagnn-main/analysis/dynamic_vs_base_regressions_top20.csv)

按 gold label 的准确率（baseline -> dynamic）:
- `A`: `0.410 -> 0.194`
- `B`: `0.394 -> 0.171`
- `C`: `0.367 -> 0.204`
- `D`: `0.419 -> 0.226`
- `E`: `0.415 -> 0.221`

可见退化不是局部标签问题，而是全局退化。

## 5. 动态图结构质量诊断
### 5.1 图规模崩塌
- baseline `train` 平均节点数 / 边数: `120.77 / 953.36`
- dynamic `train` 平均节点数 / 边数: `9.52 / 1.00`
- baseline `dev` 平均节点数 / 边数: `117.94 / 924.28`
- dynamic `dev` 平均节点数 / 边数: `9.47 / 1.20`
- baseline `test` 平均节点数 / 边数: `118.53 / 935.60`
- dynamic `test` 平均节点数 / 边数: `11.15 / 2.83`

### 5.2 单边图占比（edge nnz=1）
- dynamic `train`: `1.0000`
- dynamic `dev`: `0.9491`
- dynamic `test`: `0.5126`

这说明动态图在大量样本上只剩“保底边”（近似空图），GNN 几乎无可推理结构。

### 5.3 运行期 fallback（网络异常诱发）
从 [terminal.log](/mnt/d/Users/Administrator/Desktop/qagnn-main/terminal.log) 可解析到多次构图运行；其中一次 `train` 运行出现 `fallback=48675/48705`（几乎全降级），这与上面的“单边图占比 1.0”一致。

## 6. 根因判断
1. **网络/代理不稳定 + `ALLOW_FAILURES=1`**  
   请求失败后直接 heuristic 回退，且数量可非常大，导致图信息熵骤降。
2. **LLM 三元组到 `concept.txt` 的映射命中率不足**  
   未命中时无法形成有效边，样本进一步退化为“近空图”。
3. **当前 `cid2score` 设计过于平坦**  
   动态图里分数基本恒定，难以给模型提供有区分度的节点强弱信号。
4. **训练速度异常（unfreeze 后 ms/batch 暴涨）**  
   这是稳定性问题，不是准确率主因，但会放大实验不确定性与中断概率。

## 7. 结论
本次 dynamic KG 实验结果“差”并非偶然波动，而是由**图质量系统性退化**导致。  
在当前实现下，dynamic 图并未替代 baseline 的图推理信号，反而在 train/dev/test 上引入了大面积“近空图”样本，因此出现 `0.20x` 级别准确率是可解释的。

---
（本报告基于本地现有日志、`log.csv`、`test_e*_preds.csv` 与 bad case 脚本输出生成。）
