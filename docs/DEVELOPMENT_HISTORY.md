# 智训引擎工业模型实验系统

2026-11-17：已接入官网发布的contest_harness 1.0.2，使用官方
`contest.harness.workspace.v2`与`host_gateway_rpc`。正式SDK构建入口为
`submission/build_official_image.py`，固定容器入口为`/app/harness.py describe/train/predict`。
镜像不内置数据、历史权重或外部PyTRIO采样客户端；模型身份由平台绑定，训练和预测使用平台挂载。
当前本地验证：116项项目测试、15项官方SDK测试、A连续三轮及B/C各一轮容器检查通过。
容器检查的模型回包明确为本地替身；真实Sampler与隐藏分数尚待平台验证。
作品技术说明见`submission/作品说明_官方SDK版.md`，下文保留此前研发和试上传过程。

当前版本提供可复核的 CPU 工业模型续训、公开验证、候选采用与 SwanLab 记录。
A 为六类气体识别，B 为轴承分钟 RUL，C 为发动机循环 RUL。

新增独立的试上传构建：`submission/build_trial_image.py` 使用最小构建目录和
`Dockerfile.trial`，无需将未知的官方运行契约标记为已验证。该镜像包含已核验的
A/B/C公开训练包和原始检查点，默认各续训10轮、每轮5epoch，离线记录SwanLab，
默认不调用大模型。脚本仅在真实容器健康/三任务单轮训练及归档检查均通过后导出tar。
该入口为本项目CLI，正式平台适配、隐藏评测及大模型工具调用兼容性尚未验证。

2026-11-16 平台提交准备：`submission/` 提供源码材料整理、受官方运行契约约束的构建脚本，
以及单个 Linux/amd64、未压缩 `docker save` 归档的静态检查。
环境后续确认：用户终端已显示 WSL 3.0.5.0、默认 WSL2，且 `docker info` 返回 `linux`；
本机 Linux 引擎已恢复。当前 Codex 会话连接 Docker 命名管道仍报权限不足。
官方 Harness 运行协议与实际适配、真实源码仓库 URL 仍缺失。
试上传镜像的构建结果以实际构建报告为准，脚本不自动上传或启动平台试运行。
根目录 `Dockerfile` 默认执行 `doctor`，仅为早期开发环境骨架，不能作为平台 Harness 提交。
源码 ZIP 不能上传镜像栏。正式构建须使用平台提供的数据和初始检查点；提交构建上下文
排除本地数据、历史权重、实验归档及账号凭证。依赖在 Linux 下安装和完整运行尚待验证。
`submission/runtime_contract.json` 是项目内部待确认记录，不是官方接口规范。

2026-11-16 第8轮补充：提供带时间截止条件的历史版本集成、已释放标签的逐类可靠性、
分块/对角/均值与协方差拆分的域适应，以及延迟反馈的域视图选择器。
100组预测对照和两次真实45轮闭环接入检验分别保留原始结果。直接平均历史版本未超越
现有单模型，预先声明的前段选型/后段复核未产生可替换默认配置的胜者。

新增分块CORAL研究候选在既有质量轨迹上为76.06%准确率、68.24%平均批次Macro-F1；
batch8–10为73.37%、71.34%，对应旧温和CORAL为71.31%、70.49%。这是看完对照后发现
并重新计算一致的探索候选，没有被预先声明的前段规则选中，不能称为独立验证胜者。
原75.39%候选保留；新研究模型包在 `artifacts/model-candidate-block-coral-v1`。
其全流甲苯召回下降、G-mean略降，数据复用与正式整批推理兼容性限制继续有效。

```powershell
python -B run.py ensemble-study --mode offline
python -B run.py domain-study --mode offline
python -B run.py selector-study --mode offline
python -B run.py fair-replay --strategy flat --config configs/replay_guarded.json --prediction-config configs/predict_ensemble_k3_coral.json --mode offline
```

`ensemble.py` 只允许已采用、当前批预测前已提交的版本；完整历史注册表须经
`temporal_versions`恢复截止点，不能把未来采用版本用于旧批次。k=1匹配当前父模型，
回归可按时间权重平均非负输出。逐类可靠性只用当时实际发出的预测，避免回头在模型自己的
训练样本上评分。`ViewSelector`先输出预测，释放标签后才更新下一批的权重或选择。
该规则选择器不是Qwen，也不声称它是新训练的大模型Agent。

`fair-replay --prediction-config`显式开启集成/CORAL推理，预测与真实更新循环接通，
版本反馈状态随游标提交并支持恢复。候选采用仍比较原工业模型，单独标记该门槛的范围。
筛查缓存耗时、实际闭环耗时和额外前向次数分开，零额外神经网络训练不等于零推理成本。

新模型包继续使用 `predict-batch`，只需将两个路径改为block-coral模型包。
8维分块是对既有特征索引的工程分区，不冒称官方提供了新的传感器字段字典。
本轮没有新增PyTRIO语言模型训练或官方数据；上轮24步权重和派生语料保留。

2026-11-16 新增预测阶段的历史先验、正则化 soft BBSE 和目标到源 CORAL 对照，
以及带同状态准确率下限的 Harness 派生语料与真实 PyTRIO Agent 回放。
`configs/targeted_study.json` 预先登记两条固定模型轨迹上的16组实验；预测保存后
才用当前标签评估，旧检查点、vendor、归一化参数和官方包内数据均保留。
曲线见 `artifacts/study-targeted-20261116T065925/comparison.png`。

历史先验与 BBSE 总体下降。强度0.25的CORAL在45轮轨迹上将准确率从72.78%提至
73.18%、平均批次Macro-F1从65.41%提至65.66%；在既有108轮轨迹上将两者从
73.46%/66.43%提至75.39%/68.04%。这是多批模型轨迹的公开开发成绩，不是某一个
最终检查点在全部数据上的独立测试分数，尚未达到85%。
按批2–7选出的强度0.5在后段Macro-F1下降，未采用；强度0.25为看完对照后的
平衡候选，仍须新数据验证。现有结果不能证明漂移只有某一种机制。

```powershell
python -B run.py adaptation-study --mode offline
python -B run.py fair-replay --strategy flat --config configs/replay_guarded.json --mode offline
python -B run.py fair-replay --strategy agent_trained --config configs/replay_guarded.json --training-run artifacts/pytrio-20261116T070849-b1561c37 --mode offline
```

`metrics.py` 补充balanced accuracy、G-mean、Cohen's Kappa；逐类recall在详细结果和
新对照的SwanLab标量中记录。Macro-F1固定六类，G-mean/balanced accuracy仅取当前
真实出现的类别；Kappa期望一致率为1时记0，并显式记录计算约定。

`data/harness-guarded-v4-expanded` 是项目派生数据：31条训练、18条评估，来自实际
回放状态和101个新增候选验证，保留旧152个候选证据，没有增加或生成工业标签。
训练标签的离线下一批评估最多到batch7；未来分数、chosen/rejected及候选结果
不进入运行时Prompt。下限为每个源状态实际flat动作下一批分数的99%，不是全流
最终分数。该离线约束不保证未知未来批次也达标。

本轮Qwen3.8-27B通过真实PyTRIO完成24个优化步，保存Train/Sampler权重。
`agent.py` 真实请求决策，经有限动作校验、预算检查和候选采用门槛后执行续训；
超时及无效响应明确回退规则。日志记录响应、token、动作、候选结果和耗时。
能力接通与效果改善分别验收；策略未通过全流和后段门槛时不替换flat。

实验性批量预测接口如下，输入仅为无标签的 `{"inputs": ...}` JSONL：

```powershell
Get-Content inputs-only.jsonl | python -B run.py predict-batch --checkpoint artifacts/model-candidate-coral-soft-v1/checkpoint.npz --adapter artifacts/model-candidate-coral-soft-v1/adapter.npz
```

源几何仅由batch1冻结归一化后的特征计算，不重拟合vendor的mean/scale。
CORAL需要整个当前无标签特征批次，拆成逐条请求不等价；正式赛事是否允许这种
批量转导推理尚未确认。普通 `predict` 仍使用原检查点逐条预测。

运行环境为 Python 3.11–3.12。安装 `requirements.txt`，将 `src` 加入 `PYTHONPATH`。
可通过根目录 `run.py` 启动；本机操作时使用指导中提供的隔离环境解释器路径。

```powershell
python -B run.py doctor
python -B run.py train --task A --mode offline
python -B run.py train --task B --mode online --project zhixun-school
python -B run.py predict --checkpoint artifacts/实际实验目录/checkpoint.npz
```

预测从标准输入接收 JSONL，仅接受 `inputs` 与可选的 `task`。分类输出六类标签及分数，
回归输出非负原单位数值。模型输入结构必须与相应初始检查点一致。

训练只读取指定任务的公开训练集。验证集用于监控和末轮比较，不参与参数更新、
标准化或目标缩放拟合。每轮新增一个候选，固定末轮后比较，不按最高验证 epoch 选型。
A 要求 Macro-F1 提高且 Accuracy 不下降；B/C 要求 RMSE 降低且 MAE 不增加。
未通过时选定检查点保留父模型。后续续训通过 `--parent` 指向实际选定的检查点。

每轮产物包括 `registration.json`、`history.jsonl`、`metrics.csv`、`candidate.npz`、
选定的 `checkpoint.npz`、逐条候选预测、`run.json` 以及真实 SwanLab 日志。
失败运行保留 `failure.json`。输入摘要、代码摘要、父子检查点和优化器步数可以复核。

`offline` 记录仅在本地。只有完成实际同步或在线运行后，才能提交云端记录链接。
在线模式的初始化失败会报错，不将自动降级误报为云端成功。

PyTRIO 接口与 CPU 模型续训分开。`pytrio-models` 查询账号支持的模型，
`pytrio-train` 仅接受明确提供且有来源记录的语言模型训练数据。没有默认模型名、
内置伪造语料或默认训练额度。远端训练尚须账号、可用额度和合规的训练数据。

`package` 将程序、配置、数据及运行证据打包到本地交付目录，排除指导、本地环境、
缓存和凭证。来源和辅助开发情况见 `NOTICE.md`。

当前成果是校赛本地实验基础版本。漂移、规则反馈策略和版本恢复已提供开发实现。
正式隐藏流对接、驱动 Harness 的赛事模型及目标环境资源验收仍需逐项完成。
本地公开验证分数不等于正式平台成绩；此处不宣称已完成正式赛事最终验收。

输入契约（来自收件任务包）

| 任务 | inputs 字段 | 监督目标 |
| --- | --- | --- |
| A | features 数组，128 个数值 | answer 为六类气体英文名 |
| B | features.horizontal 与 features.vertical 两组已提取特征，共 34 维 | answer 为非负剩余分钟 |
| C | settings 数组 3 维，sensors 数组 21 维 | answer 为非负剩余循环 |

训练文件包含 id、inputs、answer 和 meta；只有 inputs 进入预测器。验证 id/meta 只用于集合隔离和追溯。具体路径顺序由原始检查点固定。

云端记录核验

`python -B run.py verify-cloud --url 实际SwanLab实验链接` 通过官方 SDK 只读核验运行名称、原始配置及逐点指标，成功后保存 evidence/cloud_records.json。省略 --url 时复核已保存链接。此命令不上传实验、不启动训练，也不修改远端项目。

## 公开验证漂移闭环

```powershell
python -B run.py stream --task A --mode offline
python -B run.py stream --task B --mode online --project zhixun-school
python -B run.py stream --task A --resume artifacts/实际stream实验目录 --mode offline
```

`stream` 使用 `configs/stream.json`：原始训练特征拟合 PSI 分位数边界和真实频率，
分类另测平均最大预测概率变化；输入信号决定保持或最多 5 轮续训。归一化和 Adam
状态继承，更新只使用原始训练文件。验证文件始终完整保留，只做开发评估和采用门槛。
特征 PSI 排名表示分布关联，不代表因果归因。没有分布漂移也可能仍有性能改进空间。

每次调用创建独立 `run-stream-*` 实验，含配置、标量日志、诊断文本、SwanLab 日志和
选定的 `selected-checkpoint.npz`。初次实验目录另保存 `drift-reference.json`、
不可覆盖的版本检查点、`stream-state.json` 和批次结果。失败时保留 `failure.json`。
与 `train` 的 latest-A/B/C 指针分开，不自动替换此前实验模型或部署任何服务。

当前 CLI 每个任务只回放收到的 **一份完整公开验证文件**。内部 EvaluationBatch 为
本地工程接口，多批次顺序和组隔离通过单元测试，不宣称已有正式隐藏流或真实多轮成绩。
所有示例中的多个批次测试夹具均仅用于单元测试，不属于训练/验证数据。

版本表可回到任意已采用且摘要一致的历史版本。置信度下降时，最多重评 3 个历史版本，
只有在同一当前批次上通过相同采用门槛才回滚；不按不同批次的历史分数排名。
规则策略在同类漂移上次更新被拒绝后降低学习率；这是确定性反馈启发式，尚非训练过的
大模型策略，当前真实数据只有一批，尚不能证明它通过多批反馈提高收益。

预算是本地设置：最多 3 次更新、累计训练 900 秒、单轮协作式 300 秒、0 次 LLM 调用。
进程锁防止同一会话并发写；版本、游标和预算原子提交。中断发生在提交前时保留整轮预算
预约并记录 interrupted_attempt，恢复后在剩余额度内重试。已提交批次不会重复训练。
单个 NumPy epoch 不能被该协作式检查强制打断；此处不是正式 CPU/内存/容器资源认证。
`train/seconds` 和预算扣账不含数据加载、PSI、评估、同步，不能当作端到端比赛成本。
恢复严格检查输入、批次顺序、配置和源代码摘要；改代码后应创建新实验，不拼接旧实验。

## 训练池与 UCI 本地回放

```powershell
python -B scripts/prepare_gas_sensor.py --archive <实际UCI原始ZIP路径>
python -B run.py replay --strategy static --mode offline
python -B run.py replay --strategy always --mode offline
python -B run.py replay --strategy performance --mode offline
```

转换器锁定已核对的原始ZIP摘要，先用 batch1/2 完整复现收到的 train/validation 每行字段，
成功后才输出 batch3–10。派生数据、原始ZIP及独立manifest在 `data/replay-gas`，
不是赛事官方隐藏流。收到的 `data/A` 文件和原始manifest不覆盖。

`replay` 是与 `stream` 分开的显式开发协议。batch2–10按原时间批次顺序回放。
每批先保存当前模型的预测，再使用该批标签诊断与选择；候选仅在之前已释放标签的训练池上
续训，选择完成后才将该批加入池。故 batch2 标签可用于 batch3 以后的续训，batch2 不再是
这些后续模型的独立留出集。全部标签即时可获得是本地实验假设，不是假称正式赛场授权。
原始 batch1 归一化永不重拟合，Adam状态与步数继承；池子不负责拟合任何缩放参数。

性能策略使用首批 Macro-F1 作为固定参照，下降超过配置的0.05才触发更新；有标签的回放中
PSI仅作诊断，避免离初始分布越来越远而反复触发。规则、预算、epochs和学习率在配置里预先
固定。所有模型都按同样双指标门槛选择。没有按最终成绩回头挑阈值或删失败候选。

主报告是 `prequential/*`：当前标签用于选择模型前的预测分数。`selected_same_batch/*`
单独保存，并承认它受同批选择偏差影响。汇总accuracy按样本数加权，Macro-F1为9批分数
算术平均，六类集合固定。单个时间流的结果不能代表跨任务普遍收益或统计显著性。
训练池、模型版本、批次游标和预算一同提交，恢复时从已提交批次重建并核对池内容摘要。

## Harness 语言模型训练与评估

`prepare-harness --run <static目录> --run <always目录> --run <performance目录>`
从三个真实完整回放的实际决策生成 `data/harness-sft`，逐行保存来源与时间批次。
这是用户授权的项目派生模仿语料，标签为程序实际动作，**不是官方决策标签或最优动作**。
批2–7的18条示范训练，批8–10的9条示范留出；输入不含当前候选结果、当前采用结果或未来批次。

```powershell
python -B run.py pytrio-train --data data/harness-sft/train.jsonl --provenance data/harness-sft/provenance.json --evaluation data/harness-sft/evaluation.jsonl --model Qwen/Qwen3.8-27B --epochs 3 --batch-size 3 --learning-rate 0.0001
python -B run.py pytrio-evaluate --run <实际完成的PyTRIO实验目录>
```

训练通过真实PyTRIO SDK执行LoRA，按chat模板构造输入、只对completion/EOS计损失，使用
随机种子42的epoch洗牌与小批次优化。默认按epochs完成计划，移除原先最多10步的初期限制，
没有10元金额上限。训练前后在时间留出集计算交叉熵，保存Train和Sampler两种远端检查点路径。
本地ZIP中的这些路径不等于27B权重已下载或正式离线部署完成。

`pytrio-evaluate` 以相同256生成token上限、temperature=0对基础模型与已保存权重比较
JSON合法率及真实示范动作一致率。所有输出只校验和记录，不执行代码，也不直接控制工业模型。
此项指标是动作模仿能力，不是气体识别准确率。闭环收益由新增fair-replay另行检验。
绘图可选依赖见 `requirements-plots.txt`，绘图脚本为 `scripts/plot_replay.py`。

## 训练效能扩展与对照

`model_extension.py` 提供从原版Model派生的加权交叉熵训练及显式宽度迁移，vendor文件未改。
类别权重只从当轮实际训练样本计数计算，按样本平均权重归一到1。权重幂0时与原版Adam
续训逐值一致。B宽度迁移克隆已有隐藏单元，并给同源单元的输出权重施加零和扰动，保持
迁移前预测函数；旧Adam前缀保留、新增矩为0，不能称作优化轨迹严格等价。

`training_schedule.py` 在策略已经要求更新后分配2–20轮，并在后40%轮次降低学习率。
真实性能下降不会被低PSI否决。`TrainingPool.sample` 可按历史批次分配保留量并逐epoch
确定性重采样；样本仍只能来自已释放标签的过去批次，种子、样本ID摘要、类别计数和权重
均进入运行证据。原始训练统计量不重拟合。

```powershell
python -B scripts/run_efficiency_study.py --mode offline
python -B run.py replay --strategy always --config configs/replay_soft_scheduled.json --mode online
python -B run.py capacity-train --hidden 24 --epochs 20 --learning-rate 0.0003 --lr-decay 0.5 --mode online
```

初始对照与追加开发实验分别由 `configs/efficiency_study.json` 和
`configs/efficiency_followup.json` 登记。追加实验明确是在首轮负结果后提出，不当作盲测。
原性能策略配置 `configs/replay.json` 保持不变；新的性能触发配置未超过原配置。
开发数据上的质量候选为温和加权+调度的always策略，B质量候选保留24单元、采用更低学习率。
完整结果、负结果、135个批次及6项B实验复核与云端记录见本轮evidence文件和原始artifacts。
不能将更长训练的改善称为等成本收益，也不能将本地分数称为官方统一静态基线成绩。

第5轮JSON提案和UpdatePlan的轮数边界扩大到20，当时PyTRIO权重仍来自0–5轮语料。
旧语料和检查点未覆盖；语料构造器拒绝把超过5轮的新动作写进旧格式的示范。
允许解析20轮动作不意味着当时已学会新调度或接管回放。新增v3/v4策略单独验收。
