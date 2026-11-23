# 复算与 GPU 重跑

## 1. 离线复算

安装 `.[test]` 后运行 README 的测试、`verify_public.py` 和两条 `bench.analyze` 命令。分析结果写入 `reproduced/`，不要覆盖仓库中的发布结果。此路径不需要模型或 GPU。

复算核验 40 组、11,560 次请求、11,559 次成功、1 次失败；失败保留在 offered 分母。主实验期望 36 组，验证期望 4 组。成功请求延迟与失败率分开解释。种子为统计配对单位，请求不是独立重复。

## 2. 功能演示

README 提供原生 vLLM 与单 worker 网关启动命令。`gateway.demo.yaml` 保留本轮技术参数，默认 `cancel_path_verified=false`。该值意味着取消上游工作尚未在当前后端验证；不要把旧机器的成功观测直接沿用到新部署。

接口仅支持单条 token-ID prompt、流式 completions、`n=1`；使用权重对应 tokenizer 准备输入。业务应用需要的文本/聊天接口、多 worker 共享状态、鉴权和高可用不在本次范围。

## 3. 重跑全 GPU 实验

在 Linux 新工作目录或新 checkout 中执行。公开 `configs/engine.yaml` / `frozen.yaml` / `validation.yaml` 的 GPU 标识均为占位符；替换为当前获分配 GPU，并调整模型路径。`configs/pod.yaml` 是保留资源约束的部署示例，必须按自己的集群修改节点、卷和用户设置。

安装运行依赖、vLLM 与匹配的 CUDA/Torch；GPU 环境还需要 `transformers`、`pytest` 和 C/C++ 工具链。参考版本以依赖快照和最终报告为准，不要把 runtime 镜像标签当成实际应用 Torch/CUDA 版本。

先建立全新的 `artifacts/` 输出位置，不能把旧观测与新测量写进同一目录。可把本仓库发布证据保留在旁边，创建独立 checkout 专用于新运行。原始 pipeline 不负责自动隔离输出目录。

按顺序：

1. `python -m pytest -q`；`python scripts/record_environment.py` 记录当前环境。
2. `python -m bench.workload --help` 准备 tokenizer 与开发轨迹；`python -m bench.serve --help` 启动对应原生后端。
3. 运行模型冒烟与 `bench.cancel_probe`、`bench.gateway_cancel_probe`，确认 token 和真实取消链路。
4. 使用 `bench.develop` 在开发种子重新标定/选择配置，再生成独立主轨迹、冻结配置与来源快照。
5. `bench.matrix` 运行主矩阵；通过受限的恢复边界处理意外中断。不要删除已有记录来强行重跑。
6. 单独运行验证轨迹并分析。原型脚本的具体参数以各模块 `--help` 为准；自动化顺序可审阅 `bench/pipeline.py`。

GPU 重跑应当视为新实验，不能把公开的旧冻结预算或 SLO 当作任意模型硬件上的最优设置。先保留原生 FCFS 对照，记录失败、发送落后与尾部排空，不预设自定义策略会获胜。

## 4. 本轮固定条件

90 秒到达窗口；light 2 req/s，near_capacity 3.8 req/s；burst 平时 2.4 req/s、35–55 秒为 8 req/s。参考 4 req/s 是开发探测规则的产物，不是物理容量上限。请求类别 S/P/D 分别为输入/输出 512/128、8192/128、512/512。

主种子 1101/1102/1103，开发 2201，验证 4401。关闭 prefix cache、启用 chunked prefill、所有条件统一 `VLLM_USE_FLASHINFER_SAMPLER=0`。每组开始/结束检查空闲；发送落后 P95 超过 20 ms 判为失效。
