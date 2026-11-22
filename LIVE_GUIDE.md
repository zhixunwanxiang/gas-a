# PyTRiO、SwanLab 与时间回放

本地联调真实调用 PyTRiO，工业分类器仍在 CPU 上续训。正式比赛通过 SDK 宿主网关执行，不能用本地 API 回执替代赛事回执。

当前稳定基线为 `stable-20261120`，清单见 `D:\aiproject\gas1\stability\STABLE_VERSION.json`。使用 `--config configs/a.stable.json status` 或 `serve` 读取保留的 `batch03_sparse_weights`。该基线保留 batch6 控制器校验失败这一已知问题，完整闭环验收状态仍为未通过。完整状态见 `D:\aiproject\gas1\runtime\temporal_current_20261120_141414\validation-report.txt`。

## 环境与凭据

```powershell
.\scripts\setup_live.ps1
.\.venv\Scripts\python.exe -m gas_a live-configure
.\.venv\Scripts\python.exe -m gas_a doctor
```

Python 3.10～3.12，NumPy 1.26.4，SwanLab 0.10.5。SDK 使用随包版本。`.env.local` 只保存本机凭据，不进入镜像或交付包：`PYTRIO_API_KEY`、`PYTRIO_MODEL`、`SWANLAB_API_KEY`，可选 `GAS_A_SWANLAB_PROJECT` / `GAS_A_SWANLAB_WORKSPACE`。

`doctor` 默认只查配置；`doctor --probe-model` 会实际请求模型并产生费用。运行模型由 `configs/a.live.json` 的 `live.controller_model` 显式指定，当前为 `Qwen/Qwen3.8-27B`；权限和白名单以平台为准。

## 执行与查看

```powershell
.\.venv\Scripts\python.exe -m gas_a agent-run --rounds 1
.\.venv\Scripts\python.exe -m gas_a --config configs/a.live.json status
.\.venv\Scripts\python.exe -m gas_a --config configs/a.live.json review-run
.\.venv\Scripts\python.exe scripts/verify_live_run.py --config configs/a.live.json --expected 1
```

默认使用真实模型与一个私有 SwanLab 实验；云记录初始化失败时停止。`--local-records` 显式选择仅本地记录，不宣称完成云端留痕。正常 SDK 轮次保留真实规划、版本选择回执；无效最终选择允许纠正一次。无效规划停止新增训练，继续安全版本选择。

## 时间回放

```powershell
.\.venv\Scripts\python.exe -m gas_a temporal-run --dataset-directory 'D:\liyue\工业模型赛题研究\gas+sensor+array+drift+dataset\Dataset' --output ../gas1/runtime/temporal-new-run
.\.venv\Scripts\python.exe ../gas1/scripts/verify_temporal.py ../gas1/runtime/temporal-new-run
.\.venv\Scripts\python.exe ../gas1/scripts/verify_temporal_cloud.py ../gas1/runtime/temporal-new-run
```

输出必须使用新目录。batch3～10 是8个时间窗口：先固定当前模型和静态基线的预测，再开放当前批标签做诊断与影子评估；新部署仅影响后续预测。梯度只用原训练集，选 epoch 只用原验证集；末批仅检验。已有 DAT 不自动成为平台授权训练集。

Macro-F1 纳入真实或预测中出现的类别；缺类窗口的不同模型可能平均不同的类别数。逐批比较需同时检查类别集合，并可补充固定六类结果；汇总成绩合并全部冻结预测后重新计算。该计分行为沿用当前参考兼容实现。

需要复跑时，可加 `--prior-attempt <旧运行目录>` 保留原始失败记录并累计成本；不得把旧尝试的费用记为零。

时间研究可显式使用 `--config configs/a.adaptive.json`（该参数放在 `temporal-run` 前）。它继承 live 配置：先冻结候选与其真实父版本对下一窗口的预测，再揭示标签，向27B提供配对 Macro-F1 和六类召回变化；同一窗口的多个探针先平均，避免重复观测虚增证据。探针收益与实际部署成绩分开保存。

adaptive 在全部已发布版本中按最近三窗 0.6/0.3/0.1 权重选择，原验证集作为静态基线保底。类别软目标与硬下限分开：默认乙醛召回目标是预警；业务必须保障的类别应写入 `adaptive.hard_minimum_class_recall`。全部版本不满足硬下限时明确标记降级。历史最优不保证下一窗口最优。

27B每窗最多为一个实验调整两个白名单训练参数，学习率仅允许菜单值0.5～2倍并受绝对上限约束；结构、Adam、随机种子、scaler和数据路径不能修改。原始回复保留；单实验平铺参数可无歧义解析，展示使用摘要，事实与参数仍逐项校验。模型不会重写训练样本的标签。

`--controller rule` 仅用于无模型调用的 CPU 筛选，关闭云端且明确标记；不能冒充真实27B试验。adaptive 是本地时间研究配置，正式 SDK 默认门槛不变，是否可提交仍按赛事宿主的数据与评估授权执行。

## 决策、门槛与预算

公共参数维护在 `configs/a.json`，live继承它。每轮候选上限、epoch 和时间预算以配置为准。正常执行实验的时间窗口有2次调用，模型选择 observe 为1次，规则暂停为0次；每窗最多额外纠正1次。SDK暂停训练时仍保留宿主要求的真实模型决策回执。

规划提供六类指标、最近三窗失败原因，区分相同父版本、验证集哈希与同批影子收益。回复须包含证据编号、假设、实验、通过条件和停止条件；编号、预算、历史最佳/未试断言及门槛可自动校验，自由文本假设仍需实验验证。PSI不是传感器重要性或物理故障证明。

同父版本、同训练和验证数据连续3窗无合格实验时暂停训练一窗，再重新判断，阈值为 `controller.rejected_windows_before_pause`。候选训练、重载评估、门槛和执行资源计量由 `candidates.py` 共用；调度、宿主提交和部署仍由各流程负责。

默认检查 Macro-F1 增益与静态基线下限、准确率退化、乙醛绝对召回、其他有足够支持样本类别的召回退化。历史恢复同样检查类别保护，原始基线作为显式紧急兜底。具体数值为工程配置，不等同赛事官方要求。

`sensor_robust`、`noise_robust` 仅训练时扰动样本，不改变预测输入或拟合新的标准化参数。模型结构、Adam、种子、步数和标准化沿用父检查点。验证集不参与梯度。

## 完整记录与引用

`model_calls/call-*.json` 是单次模型交互的完整原文，先保存请求，再更新真实响应、用量或错误。新事件和历史保存 `gas-a.model-call-ref.v1`：记录ID、相对路径、SHA-256、用量和简短摘要；读取时验证路径、哈希和ID。重复的响应文本从原始 response 派生。

SwanLab每次调用上传一次完整记录，`events.jsonl` 保留引用；SDK自身的上传缓存由SDK管理。报告或导出会按引用展开完整请求/响应，兼容旧版记录。`export-model-calls` 生成引用索引和阅读报告，不复制原文文件。迁移不会改写既有实验日志；搬移完整运行目录时保留相对目录结构。

候选三件套和源码哈希证明续训来源；训练资源保留在 `training.json`，共同执行步骤的总资源记录在 trial.resources，包含重载和影子评估。失败、未采用和未导出 epoch 均计入成本；服务端GPU时长、平台训练额度未知时保持未知。实际成绩、影子成绩和回滚演练分别记录。

## 镜像与正式验收

```powershell
.\scripts\build_image.ps1
# 或单独验收已有镜像
.\scripts\verify_image.ps1 -Image gas-a-harness:1.0.4
```

构建使用当前Dockerfile的精简运行环境；Python、NumPy、官方SDK、运行源码、公共配置和许可证保留。打包通过 `docker save`，脚本检查归档大小与镜像源码哈希；支持脚本参数中的 Docker 路径、代理和归档大小目标。

镜像按文件清单复制正式宿主代码，不包含本地HTTP客户端、SwanLab和凭据。`verify_image.ps1` 以禁网、只读容器检查协议、非root身份、客户端隔离与真实SDK交互；宿主/模型回复明确使用测试替身，不消耗模型额度。报告位于 `D:\aiproject\gas1\output\docker-verification/`。

当前实际镜像是否包含最新代码，以镜像验收结果为准；源码测试不能代替重建镜像与平台验收。正式训练需宿主提供 round_start/RPC 及实际父检查点；提交后由平台独立评测。

旧代码与旧实验已经移出项目，外部恢复归档的位置和校验值见 `D:\aiproject\gas1\stability\cleanup-plan.json`。最新实验的原始引用保留当时路径，历史对比需要从归档恢复对应数据；日常状态、预测和稳定版本核验不依赖旧目录。`D:\aiproject\gas1\scripts\verify_stable.py` 只检查当前清单、模型和固定验证预测，不联网、不新建实验。

本地临时脚本、图片、绘图脚本、回放结果、备份与导出文件统一写入 `D:\aiproject\gas1`。正式容器运行使用 SDK 授权的 `/workspace`，不把本地 Windows 路径带入宿主写入流程。
