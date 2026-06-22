# Project Instructions

## Training and evaluation constraints

- 模型训练、评估、推理和性能测试仅在远程服务器上执行。
- 完成代码修改后，不要自动启动训练脚本、评估脚本、推理脚本或完整测试流程。
- 不要执行会加载完整数据集、使用 GPU 或可能长时间运行的命令。
- 默认只进行静态检查，例如语法检查、格式检查、类型检查和代码审查。
- 不要运行 `pytest` 或其他可能触发模型运行的测试命令，除非用户在当前请求中明确授权。
- 交付代码时应提供供用户在服务器上执行的训练、测试或评估命令，并说明尚未进行实际运行验证。

## Project objective

- 本项目以官方 STAGATE 实现为 baseline。
- 第一阶段目标是复现官方实验结果，当前重点包括 DLPFC 数据集实验。
- baseline 结果完成复现、验证和记录之前，不进行算法结构改进。
- 后续改进必须建立在可复现的 baseline 上，并与 baseline 进行公平对比。

## Baseline fidelity

- baseline 应尽量保持官方数据预处理、空间图构建、模型结构、超参数、聚类方法和评估流程不变。
- 不得以重构、性能优化或兼容性调整为由静默改变 baseline 的算法行为。
- baseline 所需的兼容性修复应采用最小改动，并记录修改原因及其可能造成的结果差异。
- baseline 与改进模型应保持代码和实验配置隔离，不得直接覆盖官方实现。

## Experiment protocol

- 固定并记录数据集版本、切片编号、随机种子、依赖版本、硬件环境和全部超参数。
- DLPFC baseline 默认遵循官方预处理、空间网络构建、mclust 聚类和 ARI 评估流程。
- baseline 与改进模型必须使用相同的数据、预处理和评估协议。
- 多次运行的实验应报告均值和标准差，不应只报告最佳结果。
- 实验应保存配置、日志、指标和对应的 Git commit，确保结果可追溯。

## Result and code isolation

- baseline 复现实验与改进模型实验应分别存放，例如 `experiments/baseline/` 和 `experiments/improved/`。
- 官方 baseline 实现保留在 `STAGATE_pyG/`；未经明确要求，不在其中加入改进算法。
- 不将数据集、模型权重、`.h5ad` 文件、生成图片或其他大型运行产物提交到 Git。
- Codex 只提供服务器运行命令，并根据用户返回的日志和指标分析结果。
