# 视觉质量打分标定 · 2026-11-15

三轮标定，两个骨架缺陷已修，第三个缺陷（运动混淆）定位清楚但未修。

## 结论

| 轮次 | 参考集 | 结果 | 判定 |
| --- | --- | --- | --- |
| v1 | 87 条真实片段，帧级样本 | 12/13 H3 判 fail，但静止片段 temporal 拿 92 分位 | 骨架有缺陷 |
| v2 | 207 条，加共享低通 + 动态度门控 | imaging 轴被"越锐越好"带偏，H3 普遍 90+ | 方向错 |
| v3 | 207 条，参考改为窗口级 | 留出真实片段只得 19–39 分（应约 50） | 仍有缺陷 |

**v3 的决定性证据**：`corr(motion 百分位, temporal 轴分) = -0.969`，
`corr(motion 百分位, structure 轴分) = -0.855`。

也就是说 temporal 与 structure 两个轴实际测的是"画面动得多不多"，不是"稳不稳"。
原因：探针只做整数平移补偿，旋转与视差带来的残差随运动量线性增长，
而参考分布把所有运动强度的片段混在一起，于是百分位等价于运动排序。

## 三轮各修了什么

- **v1 → v2**：参考与打分统一到 `Canonical(fps=10, height=480)`；
  两侧共享 BoxBlur（真值 1280p 下采样已低通，H3 原生 480p 没有）；
  新增 motion 探针做门控，`motion < p10` 时 temporal 置为不可判定。
- **v2 → v3**：参考分布改为窗口均值（2484 个/轴），不再是逐帧值。
  帧级分布远比窗口均值宽，拿窗口均值去查帧级 CDF，几乎每条视频都落在尾部。
- **v2 中**：imaging 改为两侧分布（two-sided），因为生成画面比压缩后的行车记录仪更锐，
  "真实"不是该轴的上界。

## 数据

参考分布（v3，各轴 2484 个窗口样本）：

| 轴 | min | median | max | 说明 |
| --- | --- | --- | --- | --- |
| imaging | 1.056 | 7.46 | 26.24 | 两侧，偏离真实中位数越远越可疑 |
| motion | 0.00091 | 0.0174 | 0.0995 | 仅门控 |
| structure | 0.845 | 0.9855 | 1.0 | 平移到齐后灰度相关 |
| temporal | 0.00091 | 0.01644 | 0.08998 | 越低越好 |

H3 13 条（refs_v3）：11 fail / 1 warn / 1 pass。`smoke-h3` 77.7 分 pass，
而它是近乎静止的画面——门控阈值 p10 太松，没拦住。

留出真实片段 8 条：total 19–39，中位 30 左右，明显低于应有的 50。

## 下一步（按优先级）

1. **换掉平移补偿**：用 RAFT 或 OpenCV Farneback 光流做真正的 warp 残差，
   让旋转与视差被吸收；这是两个轴同时失效的根因。
2. 若暂不引入光流，退而求其次做**运动分层参考**：按 motion 十分位分桶，
   每桶一份 CDF，打分时查同桶分布。实现一天内可完成，但只是打补丁。
3. 动态度门控阈值从 p10 提到 p20–25，并补一条"运动过强/过弱都不可判定"。
4. 参考集 207 条已就绪（87 条实验片段 + 120 条从 Waymo validation 直接抽取），
   规模够用，不必再扩。

## 产物

服务端 `experiments/render_reference/`：

- `clips/` 87 条（实验 manifest 去重）、`clips_extra/` 120 条（Waymo validation 抽取）、
  `clips_all/` 207 条符号链接
- `refs_v3/` 最新参考分布（4 个 JSON）
- `calibrate_v3.log`、`scores_v2.json`（v3 打分结果）
- 抽取脚本 `/tmp/extract_waymo_clips.py`

## 复现

```bash
python -m driving_world.judge.render calibrate \
  --videos experiments/render_reference/clips_all \
  --out experiments/render_reference/refs_v3
```

环境 `/root/miniconda3/envs/ad-vlm-eval`，207 条标定耗时约 40 分钟（纯 CPU）。
代码版本 `13a15b1`。
