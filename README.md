# Gas A：气体分类续训与漂移闭环

原始代码快照：**snapshot-20261121**。项目采用 CPU NumPy MLP，原基线为128输入、32个隐藏单元、6类输出；支持续训、漂移监测、诊断、实验选择和回滚。新增的显式架构迁移支持保留父模型状态的扩宽及残差加层，见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 目录与输出约定

`D:\aiproject\gas` 保留业务源码、配置、官方数据与原始模型、SDK、测试和必要的构建工具。临时脚本、多轮回放、图片、绘图脚本、报告、备份及镜像导出统一放在 **`D:\aiproject\gas1`**。

本地配置的默认运行目录和镜像工具的默认输出位置已指向 `gas1`。后续手动指定 `--output`、`--report` 或生成图片时，也使用该目录。正式比赛仍遵循宿主授权的 `/workspace`、`/state`、`/model` 等协议目录。

## 比赛入口与数据

- `harness.py` / `contest_submission.py`：宿主 SDK 的训练和预测入口。
- `Dockerfile`、`pyproject.toml`、`vendor/contest-harness-sdk-1.0.4.tar.gz`：镜像构建与依赖。
- `src/gas_a/`：当前业务实现。
- `configs/a.json`：公共训练配置与实验菜单；`a.live.json`、`a.adaptive.json` 为本地联网和时间回放配置。
- `data/official_a/`：原始检查点、公开 train/validation、原训练记录和官方任务指南。
- `tests/` 与 `scripts/`：当前代码测试、环境设置、比赛镜像构建和协议核验。

梯度只使用授权 train，validation 用于评估和选 epoch。沿用实际父检查点的权重、Adam、步数、种子和标准化状态。每轮训练证据包括 `checkpoint.npz`、`training.json`、`history.jsonl`；宿主提交同时声明必要代码与预测入口。隐藏集由平台独立评测。

默认由27B根据业务诊断直接组合训练配置：学习率、批大小、轮数、损失、类别权重、冻结、训练扰动和架构尺寸可一起提出，不受预设名称或最多两项修改的限制。无效配置先反馈给AI修正，通过校验后才训练，见[PROMPTS.md](PROMPTS.md)。

支持真正冻结隐藏层、输出头或指定参数，冻结权重及对应Adam矩保持不变；下一轮可以解冻继续训练，见[FREEZING.md](FREEZING.md)。

默认配置从原32单元参考模型起步，27B可直接给出目标隐藏宽度和层数，无需另切配置。程序按实际父模型限制容量并先通过迁移预测一致性检查，再续训、评估及检查上线门槛。`configs/a.architecture.json`仅是单独研究架构的CPU规则示例。扩展检查点由随产物提交的更新代码加载；原始参考加载器只适用于旧格式。

## 常用命令

```powershell
.\.venv\Scripts\python.exe -B scripts/check.py --report ../gas1/stability/contest-core-tests.json
.\.venv\Scripts\python.exe -m gas_a evaluate
.\.venv\Scripts\python.exe -m gas_a --config configs/a.stable.json status
.\.venv\Scripts\python.exe -m gas_a --config configs/a.stable.json serve --host 127.0.0.1 --port 8080
.\scripts\build_image.ps1
```

`scripts/check.py` 默认执行 **61 项核心检查**：SDK入口与提交确认、逐行预测、真实回执和模型授权兜底、镜像隔离、AI配置组合及拒绝反馈、全传感器连接更新、禁止自动追加固定参照、训练/验证隔离、实际后续父模型续训及Adam重放、参数冻结、架构迁移、恢复、锁及输入预检，以及无标签预测、独立评分和稳健门槛。新增检查覆盖父成绩缺项、宿主退化拒绝、检查点名称与摘要绑定、干净/稳健两种epoch排序的完整Adam恢复，以及SDK 1.0.4分块上传、资源限额和中断会话。不能替代正式平台验收。旧菜单协议显式使用兼容模式测试；test_planning覆盖当前默认组合协议。

SwanLab、本地 HTTP、时间回放、展示和算法专项已移出默认执行。需要排查对应功能时使用 `scripts/check.py --full`；`scripts/check.py --list` 可查看默认清单。

`tests/` 按功能组织，原有92项检查保留，新增架构专项与SDK迁移检查：

- `test_protocol.py`：比赛 SDK 协议与提交、预测入口。
- `test_training.py`：训练、续训、数据隔离、参考模型兼容和优化器状态。
- `test_freezing.py`：层和张量冻结、解冻续训、Adam逐字节保留、架构迁移、类别权重和27B建议实际执行。
- `test_architecture.py`：父权重和Adam迁移、预测保持、继续训练、输入输出、梯度及混合版本恢复。
- `test_resilience.py`：监测、诊断、模型请求、训练、选择、发布、回滚和进程中断故障注入。
- `test_prompts.py`：业务提示词的信息保留、四字段回复绑定及SDK/回放实际发送路径。
- `test_planning.py`：默认AI配置组合、多项参数联合执行、拒绝后再次请求AI修正及调用预算。
- `test_optimization_policy.py`：全传感器训练、旧屏蔽连接恢复学习、弱类零召回与退化门槛、合法降权，以及实际后续父模型的Adam重放。
- `test_hidden_safe.py`：全部传感器故障验证、干净样本混合、父状态继承、稳健拒绝、无标签连续预测和独立评分。
- `test_control.py`：漂移回放、自适应选版、决策证据和回滚。
- `test_live.py`：联网运行流程与错误恢复，使用离线替身。
- `test_records.py`：模型调用记录、成本、审计、摘要和公共配置。
- `fixtures.py`：共享的规划回复、模型与云服务测试替身。

`a.stable.json` 连接已迁移到 `gas1` 的本地部署，不作为比赛宿主的父模型替代。比赛训练从宿主实际提供的父产物继续。

HTTP 接口为 `/health` 与 `/predict`；预测请求为 `{"inputs":{"features":[128个数值]}}`。本地凭据使用 `.env.local`，不进入比赛镜像。

错误恢复方式见 [FAILURE_RECOVERY.md](FAILURE_RECOVERY.md)。有效输入上的统计计算异常会标为未知，控制器无效回复可按已验证契约恢复；候选失败不会替换当前可用模型。切版使用恢复意图和幂等日志，操作系统锁在进程崩溃后释放。HTTP活动模型损坏时尝试恢复完整基线；鉴权、余额或全部模型不可用等问题明确报告，不能写成成功。

27B提示词按“根据分类错误规划实验”和“根据实测成绩解释选版”两个任务组织，格式与边界见 [PROMPTS.md](PROMPTS.md)。执行证据ID和固定门槛由程序绑定，模型只输出业务意图。

默认使用全部16个传感器，由训练学习连接权重；配置不再预设五传感器子集，SDK和时间回放不再自动追加历史参照候选，初始父模型和目标类零召回时也不例外。AI预算全部用于本轮建议。`kept_sensors`仍作为显式研究变量保留：它会强制屏蔽其他连接，并非自动学习传感器重要性；需要逐类评估其代价，单类召回或总分提高不能代表所有类别改善。目标类为零、目标召回相对父模型下降超过2个百分点，或固定验证F1相对父模型下降超过0.02时，新候选不能晋升。乙醛0.5目标仍另行报告是否达到。

取消默认配方不会改写历史检查点。`a.stable.json`仍指向历史`batch03_sparse_weights`模型；该模型已置零的连接要在后续不带子集约束、且未冻结隐藏层的训练中重新学习，不能把配置修改当作已恢复部署性能。

续训约束按随包TRAINING_GUIDE核验：项目检查点、train、validation和TASK与用户提供的包逐字节一致；梯度只用train，scaler/seed/Adam/累计计数继承。SDK遵循workspace.v2，候选写新目录，最终选择引用真实调用和history ID，收到round_ack后退出。选版均无效时，仅在真实模型规划明确授权retain_parent且父模型合格的条件下兜底；全空或无授权回复不伪造SDK完成记录。

父检查点按宿主绑定摘要从已声明的checkpoint路径中定位，支持其他文件名、子目录和多个声明文件；每个实际读取的文件仍由SDK核验。父、子宿主验证成绩齐全时，采用候选仍须通过宿主增益门槛。SDK未提供父成绩时，使用同一授权validation上父、子的本地复评进行增益比较和统一排序，候选仍必须有有效的宿主测量回执，并通过原有类别与稳健门槛。记录中的`validation_comparison`明确标记指标来源；此时`host_gate_passed=null`表示没有可计算的宿主增益，不将本地成绩改写为宿主或隐藏评测成绩。

当前使用用户提供的SDK 1.0.4原包，摘要见`vendor/sdk-provenance.json`，协议仍为workspace.v2。训练开始先确认宿主声明`artifact_upload_protocol=chunked-v1`；候选和状态由SDK按清单、偏移和摘要分块上传，不能接旧内联上传宿主。每次最多64文件、512MiB；每轮候选累计1GiB、状态累计512MiB，实际以`rnd.limits`为准，开始上传即预留预算。RPC发出后遇到中断或超时，会话结果未知，立即中止，不再纠正、提交、读取迟到响应或重试；原始请求与失败证据保留。有效回复中的无效JSON仍可在会话可用时按原预算纠正。

## 本地验证资料

### 面向隐藏评测的稳健续训与无反馈验证

当前新增可选的相对响应分类头（`response_adapter`），解决仅调学习率、类别倍率时公开验证乙醛持续漏检的问题。全部16个传感器参与，附加连接权重由训练学习；显式迁移保留父权重、Adam、计数和原scaler，迁移前后预测一致。新头先在授权train进行类平衡岭回归增量，再随原网络Adam续训；附加表示统计只在train首次拟合且冻结。输入超出冻结训练覆盖范围时，关闭附加修正，使用保留的MLP分支。新检查点完整保存附加参数、Adam、统计、覆盖检测和来源，支持SDK逐条预测及下一轮续训。算法和经验数值见[PROMPTS.md](PROMPTS.md)，增益以当轮验证和压力测试为准。

公开数据开发对照复现：`python scripts/prepare_response_controls.py --output 新目录`。该命令只读取官方公开train/validation，不访问后续DAT文件、不调用模型；结果标为开发CPU实验。27B仅在父模型、数据和算法摘要均一致时读取其聚合结果；正式候选仍须真实规划、训练、宿主测量和选版。镜像尚未重建；Dockerfile已列入新增源码及公开对照摘要，运行验证采用旧Linux环境加最新只读源码覆盖。

第一版保持单模型和原始推理预处理，不在隐藏流更新参数、拟合scaler、读取标签或等待批次反馈。Qwen27B仍在授权训练阶段提出配置并选择合格产物。默认启用授权validation的16个单传感器缺失场景，替换值来自真实父模型的训练均值；这是一组压力测试，不证明真实物理故障。候选须同时通过原有干净验证门槛、最差场景回退容差（当前默认0.005）和单场景回退容差（默认0.02）；满足保护条件后按比赛主指标Macro-F1选择checkpoint。该数值是可审阅保护策略，不是隐藏集拟合值。

训练增强仍由AI按当前授权证据提出。新增`clean_fraction`可保留一部分干净训练输入；模型、优化器和标准化从真实父检查点继续。SDK训练入口与本地agent-run共用这些检查，所有实际执行的epoch、原始AI建议和拒绝原因保留。

```powershell
# 真实Qwen27B规划、CPU续训与SwanLab审计；不会读取dat3–10标签
.\.venv\Scripts\python.exe -m gas_a --config configs/a.robust.json agent-run --rounds 1

# 将实际选中checkpoint的路径替换下方占位；整个预测流结束后才独立评分
.\.venv\Scripts\python.exe -m gas_a blind-evaluate --checkpoint "实际选中模型/checkpoint.npz" --dataset-directory "D:/liyue/工业模型赛题研究/gas+sensor+array+drift+dataset/Dataset" --output "../gas1/runtime/blind-evaluation-new"
```

`blind-evaluate`的独立工作进程仅收到当前features和冻结检查点，不收到DAT路径、标签、批次ID、凭据或评估结果。加载模型后禁止Python文件/网络/子进程访问，每条立即返回，支持重启；这属于应用层隔离，正式隐藏文件隔离仍由平台挂载合同保证。所有窗口冻结、两个预测进程退出后，评分器才解析标签；预测完成凭证、各批概率和SHA-256保留在新目录中。评分结果不会回灌训练或选版。若评分数据损坏，保留已完成预测并报告失败。

`temporal-run`继续作为有标签的研究回放保留，不能用于证明SDK隐藏预测的自主学习能力。dat3–10已经多次用于研究，新结果只能叫历史回测，不能冒充未见数据或平台成绩。

每次完整回放统一查看运行目录中的`closed-loop-report.html`：图表、逐批AI建议、实际配置、选版原因和原始对话可合并为单文件。已有报告可用`scripts/consolidate_report.py --run 运行目录`就地合并，原始记录保留用于追溯。

接入方法见 [LIVE_GUIDE.md](LIVE_GUIDE.md)。本地实验与清理记录均位于 `D:\aiproject\gas1`：

- `runtime/temporal_current_20261120_141414/`：最新回放的模型、训练、预测和交互证据。
- `scripts/`：时间回放核验、历史对比与临时分析工具。
- `stability/`：稳定基线清单与离线核验结果。
- `backups/`：旧代码、旧文档与旧实验的恢复归档。
- `output/`：镜像导出和镜像核验报告。

历史回放的batch6控制器失败及未通过结论仍保留，不因修复代码而重写。新代码增加了规划契约恢复和跨步骤故障保护；具体运行结果以对应源码版本的验证记录及平台独立评测为准。
