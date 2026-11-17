# Zhixun Harness

智训引擎校赛工业模型续训系统，适配官方 `contest_harness` 1.0.2 与 `contest.harness.workspace.v2`。

本仓库对应已完成本地检查的官方SDK版源码。运行时由平台网关绑定提交表单中的模型身份，
每回合调用 `rnd.model()`，从平台挂载的公开数据和父检查点继续训练，提交候选并记录选择。

## 当前评测入口

```text
python3 /app/harness.py describe
python3 /app/harness.py train
python3 /app/harness.py predict
```

核心实现是 [contest_runtime.py](src/zhixun/contest_runtime.py)。
它读取历史和父产物，校验模型返回的有界训练计划，保留冻结归一化与Adam状态，
提交真实候选后将测量回执交给模型决定采用或保留。预测每读一行即输出并刷新。

训练与预测目录、协议字段和网关使用方式见保留的
[官方SDK说明](submission/official_sdk_1_0_2/HARNESS_SDK.md)。
[作品技术说明](submission/作品说明_官方SDK版.md) 解释方法、来源和验证范围。

## 与交付镜像的对应关系

- 已交付镜像：`zhixun-official-linux-amd64-20261117T002217094783.tar`
- SHA256：`bbf87d277281d307242fe42b8749520a9e28789f757c760779d7d05db29813e3`
- 大小：86,419,456字节。
- [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) 保存已验证镜像与关键源码摘要的对应关系。

项目116项单元测试、官方SDK15项离线测试和五回合容器检查通过。
其中A连续三轮，B/C各一轮，验证只读根目录、非root用户、状态与历史以及逐行预测。
本地模型回包是明确标注的宿主替身，训练使用收到的真实公开数据。
真实Sampler推理和隐藏评测成绩应以平台运行结果为准。

## 从源码构建

根目录Dockerfile用于联网构建同一套核心实现及固定版本依赖：

```sh
docker build --platform linux/amd64 -t zhixun-harness:source .
docker run --rm --read-only --network none zhixun-harness:source describe
docker save --platform linux/amd64 -o zhixun-harness-source.tar zhixun-harness:source
```

构建需要能访问Python官方镜像和Python包索引。交付镜像此前使用本地OCI基础镜像与wheel缓存构建，
上述联网构建的层和文件时间戳可能不同，不能据此承诺导出tar的字节摘要完全相同。
`submission/build_official_image.py` 保留桌面工程的离线构建流程；它需要原工程的本地缓存，
不能在缺少缓存的全新clone中直接运行。

镜像通过平台获取 `/data`、`/previous`、`/state`、`/model`，写入仅位于 `/workspace` 和 `/tmp`。
本仓库不分发数据集、验证集、检查点、API Key、登录文件、镜像tar或队员操作指导。

## 本地测试

Python 3.11–3.12。完整开发测试需另外放入本人从官方收到的原始A/B/C包及本地清单，
缺少数据时不要生成假数据来填补测试。依赖锁定在requirements文件中。

官方SDK的15项协议测试不依赖真实工业数据或模型账号：

```sh
python -m pip install -r requirements.txt
cd submission/official_sdk_1_0_2
python -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
```

Windows运行SDK子进程测试时设置 `PYTHONUTF8=1` 与 `PYTHONIOENCODING=utf-8`。
源代码中的模型替身与SDK测试夹具仅用于协议验证，不是官方工业训练数据或真实模型回执。

## 实验与来源

既有大模型训练记录：
[PyTRIO 24步训练对应的SwanLab记录](https://swanlab.cn/@25165030/zhixun-school_from_813_1794808892434/runs/dkotrqyc)。
本轮接口适配未重新训练语言模型。运行日志和SwanLab离线文件由程序保存到平台允许的原生状态。

官方SDK下载来源：<https://rsi-index.115.zone/downloads/contest-harness-sdk-1.0.2.tar.gz>，
原包SHA256为 `c676c8beeef69cece9cbec5a633d05dbb9832eb22b6f761913f2315bd1f85133`。
第三方参考实现及AI辅助开发说明保留于 [NOTICE.md](NOTICE.md)。
历史研发说明见 [docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md)。
