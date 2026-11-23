"""Generate factual Chinese tables only after both GPU matrices are complete."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET


root = Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((root / name).read_text())

main_complete = read("reports/completeness.json")
validation_complete = read("reports/validation/completeness.json")
if not main_complete["complete"] or not validation_complete["complete"]:
    raise RuntimeError("Cannot write a completed-experiment report from partial data")
main, validation = read("reports/all_results.json"), read("reports/validation/all_results.json")
pairs = read("reports/paired_comparisons.json")
calibration = read("artifacts/calibration/calibration.json")
cancel = read("artifacts/cancellation_probe.json")
gateway_cancel = read("artifacts/gateway_cancellation_probe.json")
if not gateway_cancel["verified"]:
    raise RuntimeError("Real gateway cancellation verification did not pass")
environment = read("artifacts/environment.json")
test_xml = ET.parse(root / "artifacts/tests.rtx5090.xml").getroot()
suites = [test_xml] if test_xml.tag == "testsuite" else list(test_xml)
test_count = sum(int(s.get("tests", 0)) for s in suites)
failures = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
if failures:
    raise RuntimeError("Test failures remain")
offered = sum(run["stats"]["offered"] for run in main + validation)
failed = sum(run["stats"]["offered"] - run["stats"]["completed"] for run in main + validation)
lines = ["# RTX 5090 工作量准入与公平排队实测报告", "",
         f"主矩阵 {len(main)} 组、验证 {len(validation)} 组均完成且有效，共 {offered:,} 次计划请求，其中未成功完成 {failed:,} 次，保留在分母中。", "",
         f"环境：Python / vLLM {environment['packages']['vllm']} / Torch {environment['packages']['torch']}，CUDA {environment['torch_cuda']}。模型 Qwen2.5-7B-Instruct BF16；一个通过 实验命名空间分配的 RTX 5090，TP=1，全部条件顺序运行。", "",
         f"测试 {test_count} 项通过。真实取消探针 verified={cancel['verified']}，关闭连接后 {cancel['idle_after_close_seconds']:.4f} 秒观察到引擎空闲。", "",
         f"独立真实网关断连检查通过：预留预算在 {gateway_cancel['gateway_clear_after_close_seconds']:.4f} 秒内观察到清零，后端在 {gateway_cancel['backend_idle_after_close_seconds']:.4f} 秒内观察到空闲；之后经网关的有效请求成功。该检查不参与性能矩阵指标。", "",
         f"成本模型四次低负载留出观测的中位绝对百分比误差 {calibration['heldout_median_absolute_percentage_error']:.4%}；不将它扩大解释为高并发资源需求预测精度。", "",
         "## 主矩阵配对效应", "",
         "先在同一种子、同一 GPU、同一轨迹内求差值，再求中位数。n 是种子级配对数量；不给出请求级独立样本显著性结论。", "",
         "| 候选 / 基线 | 负载 | n | goodput 差值中位数 req/s | 配对相对变化中位数 |",
         "|---|---|---:|---:|---:|"]
for item in pairs:
    relative = item["median_paired_relative_change"]
    ratio = f"{relative:+.2%}" if relative is not None else "基线为零，不定义"
    lines.append(f"| {item['candidate']} / {item['baseline']} | {item['load']} | {item['paired_runs']} | "
                 f"{item['median_paired_difference_req_s']:+.4f} | {ratio} |")
lines += ["", "![主矩阵 goodput 与达标比例](figures/01_goodput.png)", "",
          "## 独立轨迹验证", "",
          "使用种子4401、同一 GPU 在主矩阵结束后顺序执行，仅为新到达轨迹验证，不是跨设备或第四张卡验证。", "",
          "| 条件 | 计划请求 | SLO 达标 | cohort goodput req/s | 失败比例 | 成功请求 TTFT P95 ms |",
          "|---|---:|---:|---:|---:|---:|"]
for run in validation:
    stats = run["stats"]
    latency = stats["successful_ttft_p95_ms"]
    text = f"{latency:.2f}" if latency is not None else "缺失"
    lines.append(f"| {run['manifest']['run_id'].split('-')[0]} | {stats['offered']} | {stats['slo_passed']} | "
                 f"{stats['cohort_goodput']:.4f} | {stats['failure_fraction']:.2%} | {text} |")
lines += ["", "## 如何解释", "",
          "B0=原生FCFS；B1=开发集选中的原生配置；B2=同预算网关FCFS；P=成本排序、等待补偿及最老请求保护。B1/B0和P/B2回答不同问题，不能把相对网关FCFS的收益写成相对原生vLLM的收益。", "",
          "cohort goodput 是固定到达窗口内这批请求最终同时满足 TTFT/E2E SLO 的数量除以到达窗口，不是稳态最大吞吐。wall goodput 包含尾部排空；失败和超时始终保留分母。", "",
          "全局 SLO 达标数量改善可能伴随长解码请求尾延迟恶化。请同时查看各类别、失败比例和每个种子的效应，不能只挑选一个正向百分比。", "",
          "三种子、单设备、单主机和共享 CPU 是解释边界。后端组内随机次序、组间交替及空闲检查减少部分漂移和串扰，不能消除所有时间影响。不声称统计显著、生产保证或跨硬件加速。", "",
          "旧 PRO 6000 的26组阶段结果独立保留，未用于凑齐本轮矩阵，也不作跨硬件配对。", "",
          "## 证据入口", "",
          "逐请求记录见 artifacts/formal 和 artifacts/validation；冻结配置与源码、模型和环境溯源见各 provenance。运行完整性见 completeness.json，具体效应见 paired_comparisons.json，失败与类别明细见 all_results.json。", ""]
(root / "reports/experiment_report.zh-CN.generated.md").write_text("\n".join(lines), encoding="utf-8")
resume = ["# 可核验的简历表述", "",
          "**vLLM 推理准入与 SLO 评测平台｜Python / AsyncIO / FastAPI / GPU 服务评测**", "",
          f"- 实现工作量与并发双预算流式网关，支持有界排队、等待补偿、取消监督及幂等回收；通过 {test_count} 项测试及真实 GPU 取消探针。",
          f"- 在 Kubernetes 实验命名空间完成 {len(main)} 组同 GPU 配对实验与 {len(validation)} 组独立轨迹验证，累计 {offered:,} 次计划请求；固定开放环到达、保留失败分母并检查来源和终态。",
          "- 对照原生 FCFS、原生优先级与同预算网关策略，分析 SLO goodput、类别尾延迟和批处理效率的取舍，形成可复核结果与原始数据包。", "",
          "性能提升表述需从完整配对表中选取有业务含义且限制明确的结果；不默认自定义网关胜过原生引擎。没有实现 CUDA 内核改写、张量并行或分布式高可用。", ""]
(root / "reports/resume.zh-CN.generated.md").write_text("\n".join(resume), encoding="utf-8")
print(json.dumps({"formal": len(main), "validation": len(validation), "requests": offered, "tests": test_count}))
