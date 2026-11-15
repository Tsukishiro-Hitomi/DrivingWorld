# Render — 视觉质量打分设计

`judge/render/` 回答一个问题：这条 H3 视频的画面，**相对真实行车视频差到什么程度**，
并且必须能指出**差在哪一秒**。它不回答"事件有没有发生"（那是 ObjectChecker），
也不回答"天气/车道对不对"（那是 ConditionChecker）。

## 分数语义

分数 = 原始指标在**真实驾驶视频分布**里的百分位，`0..100`：

- `50` = 与中位真实片段持平
- `10` = 比 90% 的真实片段更差
- 参考分布由 `reference.py` 一次性从 ≥200 条 Waymo 真值片段构建，落盘为 JSON

绝对分（MUSIQ=62.3）没有语义，不能进 gate，也不能和人工标签对齐；
百分位是唯一同时支持排序、闭环 gate 和人工校准的坐标系。

## 三个轴

| 轴 | 缺陷 | 探针 | 方向 |
| --- | --- | --- | --- |
| `imaging` | 模糊、噪声、过曝、压缩块 | 帧内高频能量（Laplacian 方差）；可替换为 MUSIQ | 越高越好 |
| `temporal` | 闪烁、抖动、顿帧、物体融化 | 全局运动补偿后的光度残差（相位相关 warp） | 越低越好 |
| `structure` | 深度跳变、结构漂移、同一目标变样 | 相邻帧 warp 后的结构相似度；可替换为 DepthAnythingV2 + DINO | 越高越好 |
| `motion` | 画面几乎不动（不是质量轴） | 未补偿的帧间灰度差 | 只做门控 |

`motion` 不参与打分，只用来拒绝回答：近静止片段的补偿残差天然极小，会被误读成
"时序稳定"。当 `motion` 百分位低于 `motion_floor`（默认 10）时，`temporal` 轴置为
不可判定、从总分中剔除，`reasons` 记 `temporal_indeterminate_low_motion`。
VBench 在评估 temporal flickering 前过滤静态视频，是同一件事。

第 4 类"语义级 artifact"（融化、多出轮胎、路面文字乱码）不进主分：
VLM 只负责解释和定位，不改分。理由：VLM 分不可复现、不可校准。

探针是可替换的策略对象：`Probe` 协议只要求 `axis` / `direction` / `measure(frames)`。
重后端（MUSIQ、RAFT、DepthAnything）惰性导入，缺失时抛 `ProbeUnavailable`，
不影响确定性核心在无 torch 环境里导入和测试。

## 解码归一化（必须做，否则百分位没有意义）

实测两边的几何参数并不一致：Waymo 真值片段是 1920×1280 @ 10fps，H3 输出是 832×480 @ 24fps
（部分实验是 448×256）。Laplacian 方差随像素尺度变化，帧间残差随帧间隔变化，
直接比较会被这两个差异主导。

所以标定和打分都必须先解码到同一个 `Canonical(fps=10.0, height=480)`：

- 空间：双线性缩放到统一高度，宽度按原始长宽比
- 时间：按最近邻重采样到 10fps；源帧率更慢时保留原帧率，不插帧
- 频率：两侧统一施加 BoxBlur（默认半径 2）。真值从 1280p 下采样时已被低通，
  H3 原生 480p 还带着高频；只对齐尺寸不对齐截止频率，imaging 与 structure 会被系统性压低

`RenderReport.canonical` 记录本次使用的设置，两份报告只有 canonical 相同才可比。

## 窗口与聚合

```
window = 1.0 s, stride = 0.5 s
```

每个探针对每个窗口给一个原始值，查参考分布得到窗口百分位，然后：

```
axis_score = 0.5 * p10 + 0.5 * median
```

**参考分布必须与打分同粒度**：标定时存的是每个片段的窗口均值，不是逐帧值。
帧级分布比窗口均值分布宽得多，拿窗口均值去查帧级 CDF，几乎每条视频都会落在尾部
（实测留出的真实片段只能拿 20–39 分，而不是 50）。

用 p10 而不是 mean：8 秒视频崩 1 秒，mean 只掉 12%，p10 会直接掉到 10 以下。
最差窗口直接写进 `Finding.temporal_segment`。

总分用加权几何平均，缺失轴在权重上重新归一化：

```
total = exp( Σ w_i ln(max(s_i, 1)) / Σ w_i ),  w = {imaging: .30, temporal: .45, structure: .25}
```

几何平均惩罚"单轴塌陷"；算术平均会让成像高分掩盖时序崩坏。

## 判定

```
status = "fail"  if any(axis_score < fail_below)      # 单轴 gate，宁可误杀
       = "warn"  if any(axis_score < warn_below)
       = "pass"  otherwise
```

阈值必须由人工标签定（见"校准"），默认值只是占位：

| 参数 | 默认 | 来源 |
| --- | --- | --- |
| `motion_floor` | 10 | 待 pilot 定 |
| `fail_below` | 15 | 待 pilot 定 |
| `warn_below` | 35 | 待 pilot 定 |

## 契约

`RenderReport` 与 `ObjectResult` 同构（frozen dataclass + `to_dict`），
`findings` 复用 `contracts/review.Finding`：

```python
RenderReport(
    schema_version=1, model="deterministic", video=str(path),
    video_duration_seconds=8.0,
    windows=[WindowScore(...)],
    axes={"imaging": AxisScore(...), "temporal": ..., "structure": ...},
    total=41.2,
    status="fail",
    findings=[Finding(...)],
)
```

`Finding.checks` 只填这四个键，无证据时为 `null`（沿用"允许 null，不强制 boolean"）。
任一轴的 `axis_score < warn_below` 就为该轴输出一条 finding（取该轴百分位最低的窗口），
`status="warn"` 也必须能指出"差在哪一秒"：

| check | 由谁决定 |
| --- | --- |
| `artifact_free` | `imaging` 轴窗口 < `warn_below` |
| `motion_smooth` | `temporal` 轴窗口 < `warn_below` |
| `geometry_plausible` | `structure` 轴窗口 < `warn_below` |
| `identity_consistent` | 恒为 `null`，等 actor 跟踪接入 |

## 校准

1. `reference.py` 跑 200 条真值片段，每个探针落一份经验 CDF
2. 分层抽 100 条 H3 结果，人工标"能否直接交付"（二值）+ 1–5 总体分
3. 每轴对人工分算 SRCC，**< 0.3 的轴删掉**，不凑维度
4. ROC 取 Youden 点定 `fail_below` / `warn_below`
5. 用 `pyiqa-gmad` 检查指标是否真的能区分你们的失败样本

## 模块

```
driving_world/judge/render/
  __init__.py    导出 RenderChecker / RenderReport / score_clip
  probes.py      Probe 协议 + 确定性探针 + 重后端惰性适配
  reference.py   ReferenceDistribution：构建、百分位查表、JSON 读写
  windows.py     按 1s/0.5s 切窗，逐窗口聚合
  score.py       ScoreConfig、AxisScore、ClipScore、几何平均、gate
  render.py      RenderChecker、decode_video、build/load/write_references
  __main__.py    CLI：calibrate / check
tests/
  test_render_score.py  合成序列验证百分位、p10、gate、几何平均、定位
```

## 用法

```bash
# 1. 从真实片段构建参考分布（一次性，≥200 条）
python -m driving_world.judge.render calibrate --videos data/waymo_front --out refs/

# 2. 给一条 H3 视频打分
python -m driving_world.judge.render check --video out/h3.mp4 --references refs/

# 两边都可以显式覆盖归一化设置（默认 10fps / 480p）
python -m driving_world.judge.render calibrate --videos data/waymo_front --out refs/ --fps 10 --height 480
```

## 验收

- 合成数据：在 8s 序列的第 3–4s 注入 30 帧闪烁，`temporal` 轴 p10 必须掉到
  `fail_below` 以下，且 `temporal_segment` 覆盖 `[3.0, 4.2]`
- 缺失轴：只给 `imaging` 时，总分等于该轴分（权重重归一化）
- 确定性：同一输入两次打分完全一致
- 无 torch 环境下 `import driving_world.judge.render` 不报错
