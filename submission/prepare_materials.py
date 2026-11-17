"""Prepare reviewable source/documents; never mislabel them as a runnable image."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_files(root):
    files = []
    for folder in ['src', 'configs', 'scripts', 'tests', 'submission']:
        files.extend(p for p in (root / folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix in {
                         '.py', '.md', '.json', '.toml', '.txt', '.ps1', '.in', '.sh'})
    files.extend(root / name for name in ['README.md', 'NOTICE.md', 'pyproject.toml', 'run.py',
                 'requirements.txt', 'requirements-pytrio.txt', 'requirements-plots.txt',
                 'requirements.in', 'requirements-pytrio.in'])
    for path in files:
        if path.is_symlink():
            raise ValueError('Source archive cannot contain linked files')
        content = path.read_text(encoding='utf-8-sig')
        if re.search(r'\b(?:sk-[A-Za-z0-9]{24,}|swlab_[A-Za-z0-9]{20,})\b', content):
            raise ValueError(f'Potential credential in source: {path.relative_to(root)}')
    return sorted(set(files))


def prepare_materials(root=ROOT):
    output = root / '指导/交付/平台试运行准备-20261116'
    output.mkdir(parents=True, exist_ok=True)
    training_name = 'pytrio-20261116T070849-b1561c37'
    training_path = root / 'artifacts' / training_name / 'run.json'
    training = json.loads(training_path.read_text(encoding='utf-8'))
    cloud = json.loads((root / 'evidence/cloud_records.json').read_text(encoding='utf-8'))
    assert training['status'] == 'completed' and training['completed_optimizer_steps'] == 24
    sampler, url = training['sampler_checkpoint_path'], cloud[training_name]['cloud_url']
    files = source_files(root)
    archive_path = output / '03_源码归档_不能当镜像上传.zip'
    if archive_path.exists():
        raise ValueError('Preparation archive already exists; version it instead of overwriting')
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files}
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            info = zipfile.ZipInfo(p.relative_to(root).as_posix(),
                                   date_time=(2026, 11, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, p.read_bytes())
        archive.writestr('SOURCE_SHA256.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert not any(Path(n).parts[0] in {'data', 'artifacts', '指导', '.git'}
                       for n in archive.namelist())
    doc = f'''# 智训引擎：工业模型受控更新与实验记录

## 作品范围与运行状态

本项目提供CPU工业模型续训、漂移诊断、有限动作Harness、实验记录、候选采用和恢复。
A为六类气体识别，B为轴承分钟RUL，C为发动机循环RUL。工业模型从随包检查点继续训练，
继承原始归一化、参数及Adam状态；vendor参考实现与原始数据保持不变。

截至2026年11月16日，已有本地实现与实验；正式平台Harness入口与工作区契约尚未对接，
尚未完成平台A/B/C各10轮运行或隐藏评测。本文档不声称镜像已通过平台测试。

## 方法与可控更新

当前批次先预测并保存结果，再使用已释放标签诊断；梯度更新只用合法的更早训练池。
分类候选要求Macro-F1提高且Accuracy不下降；回归候选要求RMSE下降且MAE不增加。
更新受轮数、训练时间和LLM调用上限约束，失败保留证据，游标、版本与预算一同提交。
本地公开回放即时反馈是假设，正式实现须以组委会运行契约为准。

PyTRIO策略模型只返回有限JSON动作，宿主校验后执行。现有实现不是任意代码工具调用的
通用编程Agent；若平台要求由大模型读写工作区代码，须依据正式协议完成相应适配。
历史版本集成只允许当前预测前已采用的版本，逐类可靠性仅由当时实际发出的预测更新。
域适应使用冻结源归一化外的受控变换；整批无标签输入访问是否允许仍须核对平台规则。

## 实验结果与限制

本地主要A实验是UCI原始时间批次2–10的公开开发回放，数据重复用于多轮研究。
该范围不是独立留出集，也不是平台隐藏评测成绩。B/C为收到的公开训练/验证文件实验。

| A的同一既有质量训练轨迹 | 温和全局CORAL | 分块域适应研究候选 |
| --- | ---: | ---: |
| 全流准确率 | 75.39% | 76.06% |
| 平均批次Macro-F1 | 68.04% | 68.24% |
| 后段准确率 | 71.31% | 73.37% |
| 后段乙醛召回率 | 32.20% | 36.58% |

分块候选为看过对照后发现的研究对象，未被预先设定的前段规则选中，且全流甲苯召回下降，
因此没有替换原默认候选。直接历史模型集成与在线选择器未通过完整验收，负结果保留。
100组预测配置对照不等于100次神经网络训练。此前103项工程单元测试通过，927批保存预测复算一致。
本轮新增提交材料检查，结果另见准备状态；未开展真实容器运行。

## 驱动Harness的大模型

基础模型：{training['base_model']}。本次对应LoRA rank32、3个epoch、batch size4、
学习率0.0001，实际24个优化步完成。动作评估损失0.12247降至0.07202。
这是动作文本损失，不是工业分类准确率。真实回放中训练后的策略仍未通过后段质量门槛，
所以本地默认保留规则控制。

持久化Sampler路径：

`{sampler}`

对应大模型训练记录：[{training_name}]({url})。
该Sampler与此前语言模型训练相对应；本轮域适应没有产生新的语言模型权重。
平台仍需独立核对账号访问、模型白名单和推理能力。

## 实验记录与复现

[完整SwanLab项目](https://swanlab.cn/@25165030/zhixun-school_from_813_1794808892434)。
目前75条完成实验、31461个标量点通过程序与本地配置和标量记录逐点核验。
原始失败与中断实验另行保留。源码入口为run.py，独立试验命令与依赖锁定见README。
本材料附源码归档，源码仓库尚未创建，未提供虚构仓库URL。

平台运行时必须使用平台提供的工作区、数据和初始检查点。
源码归档不包含本地原始数据、额外公开回放标签、历史权重、环境缓存或账号凭证。
本地doctor/replay入口不冒充正式平台Harness入口。

## 来源与辅助开发声明

参考CPU实现与初始权重来自收到的赛题包。原始数据来源为UCI Gas Sensor Array Drift、
XJTU-SY Bearing及NASA C-MAPSS；不将这些资源声明为原创。
项目增量代码使用OpenAI Codex辅助开发，保留NOTICE及原始来源。
模型、框架和数据的使用须遵守相应许可和校赛要求，参赛队员负责理解、复核与答辩。
当前材料用于平台试运行准备，尚无正式平台成绩或已完成的演示视频。
'''
    (output / '01_作品说明_可上传文档栏.md').write_text(doc, encoding='utf-8')
    assert len(doc.encode('utf-8')) < 2 * 1024 * 1024
    info = {'format': 'zhixun.submission-prefill.v1', 'model_source': 'sampler',
            'model': sampler, 'training_url': url, 'source_url': None, 'image_archive': None,
            'api_key': '仅由本人在平台官网填写，不保存于材料或镜像',
            'document': '01_作品说明_可上传文档栏.md',
            'ready_for_submission': False,
            'missing': ['Docker Linux engine unavailable: WSL not installed',
                        'Official Harness runtime protocol and actual entrypoint adapter',
                        'Real source repository URL']}
    (output / '02_平台填写信息.json').write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                               encoding='utf-8')
    (output / '04_下一步先看指导.md').write_text(
        '# 当前不是镜像交付包\n\n镜像tar尚未生成。03号ZIP是源码，不能选择它上传“镜像归档”栏。\n\n'
        '先看桌面工程的 `指导/14_平台试运行准备与阻塞.md`。Docker/官方运行协议/源码链接补齐后，'
        '由项目submission/build_and_save.ps1完成构建、容器检查和docker save导出。\n',
        encoding='utf-8')
    inventory = {'prepared_at': datetime.now(UTC).isoformat(), 'source_files': len(files),
                 'source_archive_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                 'source_archive_crc_verified': True, 'image_tar_created': False,
                 'container_tests_run': False, 'platform_run_started': False,
                 'evaluation_attempts_used_by_this_task': 0, 'credentials_included': False,
                 'runtime_contract_verified': False, 'source_repository_created': False}
    status_path = output / '05_准备状态_尚不可上传镜像.json'
    status_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    checksums = []
    for p in sorted(output.iterdir()):
        if p.is_file() and p.name != 'SHA256SUMS.txt':
            checksums.append(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name)
    (output / 'SHA256SUMS.txt').write_text('\n'.join(checksums) + '\n', encoding='utf-8')
    return {'folder': str(output), **inventory}


if __name__ == '__main__':
    print(json.dumps(prepare_materials(), ensure_ascii=False, indent=2))
