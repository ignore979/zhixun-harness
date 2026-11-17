# 来源与使用说明

工业模型参考实现 `src/zhixun/vendor/model.py`、`data.py` 来自用户取得的
`cpu-a-guide-20261110-v1.zip` 中的 `baselines/`，保持原始内容。三个任务的数据、
初始权重和配置均从相应下载包按原样提取，摘要见 `data/manifest.json`。
不将上述参考实现、预训练权重或公开数据声明为本项目原创成果。

原始数据发布来源为 UCI Gas Sensor Array Drift、XJTU-SY Bearing Datasets、
NASA C-MAPSS。使用和再分发应遵守发布者及赛题数据包的许可与参赛条款。
当前记录了收件包和下载域名，未取得组委会签名或校验公告，不把自报字段当作审批证明。

本项目的增量开发使用 OpenAI Codex 辅助编程。参赛队员需要理解、测试、复核作品，
按学校与主办方要求披露辅助开发工具和引用来源。本说明不替代主办方的原创性认定。
实验数值来自程序实际运行；未取得的云端记录、PyTRIO 训练结果和隐藏评测成绩不作声称。

来源链接

- https://archive.ics.uci.edu/dataset/224/gas
- https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets
- https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data
- https://docs.swanlab.cn/
- https://docs.pytrio.com/docs
# 第4轮派生数据说明

`data/replay-gas` 来自已提供的 UCI Gas Sensor Array Drift Dataset 原始ZIP，转换器以
batch1/2与收件文件全字段一致作为前置条件。batch3–10是项目转换文件，非赛事官方隐藏流。
来源与每个转换文件的摘要保存在独立manifest，原始标签未增加、生成或更改。

`data/harness-sft` 经用户明确授权，由真实回放日志派生。监督目标是实际记录的程序动作，
不是官方发布的大模型语料、人工最优决策或新的工业样本标签。逐行出处和时间留出规则见
该目录的provenance.json。原始资料和Codex辅助开发的既有说明继续有效。

第5轮的ClassWeightedModel.fit由随包vendor Model.fit的训练循环派生，保留原Adam、
梯度裁剪和epoch索引洗牌机制，新增按训练标签计数的损失权重。原vendor文件未修改。
扩容是从既有检查点克隆参数的显式迁移，保留旧优化器前缀状态，不将新建随机模型冒称续训。

第6/7轮的 `data/harness-oracle-v3`、`harness-guarded-v4` 与其expanded版本均是
由真实源模型状态、有限候选的实际续训和下一批开发分数生成的项目派生决策语料，
不是官方大模型标签。chosen仅表示已枚举候选中的选择，不是全局最优。
全部工业样本、来源标签和收到的训练/验证文件保持原样，实验单元测试夹具不加入语料。

历史标签频率修正不称作严格BBSE。soft BBSE和CORAL实现分别参考以下原始论文，
本项目的正则化、弱化修正和目标到源变换为工程变体，不声称论文证明自动适用于本数据。
label shift假设是p(x|y)稳定而p(y)变化；单次修正成功或失败不能证明或排除该机制。
CORAL需要当前整个无标签特征批次，其正式赛制可用性仍待确认。

- [Lipton等，Detecting and Correcting for Label Shift with Black Box Predictors](https://proceedings.mlr.press/v80/lipton18a.html)
- [Sun等，Return of Frustratingly Easy Domain Adaptation](https://arxiv.org/abs/1511.05547)

第8轮历史分类器集成的研究方向参考
[Vergara等，Chemical gas sensor drift compensation using classifier ensembles](https://www.sciencedirect.com/science/article/pii/S0925400512002018)。
本项目沿同一MLP续训轨迹产生的历史版本，不等同于论文中的学习器与多样性设计；
引用论文不保证本次集成提高分数。新代码和负结果均单独保留。
连续8维分块为工程划分，不将猜测的物理对应关系写成官方数据定义。
新增在线选择器的反馈来自实际已经预测并释放的标签，不引入合成工业样本、伪官方标签
或当前标签提前参与预测。分块候选为同一公开流上后验发现的研究对象，未经独立测试。
