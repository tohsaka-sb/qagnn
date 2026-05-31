# 文件夹归类总览（qagnn-main）

最后更新：2026-05-31

## 1) 代码主体（应上传）
- `qagnn.py`：主训练/评测入口
- `preprocess.py`：预处理总入口
- `modeling/`：模型结构（encoder + QAGNN）
- `utils/`：数据处理、图处理、优化器等工具
- `utils_biomed/`：生物医学相关预处理笔记本
- `scripts/`：分析脚本与辅助脚本
- `run_*.sh` / `eval_*.sh`：训练与评测启动脚本

## 2) 文档（应上传）
- `README.md`：项目说明
- `report.md`：主复现报告（CSQA）
- `QA-GNN阅读报告.md`：阅读/理解报告
- `dynamic_kg_training_issue_report.md`：动态图实验问题报告
- `DEPRECATED_DYNAMIC_KG_CODE.md`：动态图代码弃用说明
- `docs/`：文档归档区（按主题分类）
  - `docs/reports/`：复现报告副本
  - `docs/notes/`：阅读笔记副本
  - `docs/archive/`：历史实验/弃用内容

## 3) 数据与产物（默认不上传）
- `data/`：原始与预处理数据（大）
- `data_old/`：历史数据
- `data_preprocessed_release*/`：预处理包及恢复目录
- `saved_models/`：模型权重与 checkpoint（大）
- `analysis/`：分析中间结果（可再生）
- `logs/`、`terminal.log`：运行日志
- `.cache/`、`__pycache__/`：缓存文件

## 4) 静态资源
- `figs/`：论文图示与说明图

## 5) 说明
- 当前 `.gitignore` 已按“轻量上传”配置：默认忽略大文件与可再生产物。
- 若后续要上传数据样例，建议另建 `data_sample/`（仅保留极小样本）。
