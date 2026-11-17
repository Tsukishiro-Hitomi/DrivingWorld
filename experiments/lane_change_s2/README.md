# S2 前车变道数据集（PREVENTION）

221 条 8 秒片段，10 fps，1920x600，取自 PREVENTION 数据集的前视相机。
正样本的事件时刻是**人工标注**，与任何检测器无关，因此这批数据第一次可以测召回。

## 来源

PREVENTION（Izquierdo et al., IEEE ITSC 2019），Universidad de Alcalá。
许可 **CC BY-NC-SA 4.0** —— 可做研究和内部评测，不可商用。

只用了 3 个 drive：11 个 drive 里其余 8 个的 `lane_changes.txt` 全部 `ID = -1`，
无法绑定到具体车辆；另有 1 个 drive 有 ID 但没有轨迹。

## 组成

| stratum | 数量 | 含义 |
|---|---|---|
| `cut_in` | 46 | 从相邻车道切入自车车道——最贴「前车变道」 |
| `cut_out` | 20 | 自车车道内前车向相邻车道变出去 |
| `lateral` | 10 | 相邻车道之间横移，不涉及自车车道 |
| `negative_clean` | 119 | 随机窗口，其主体在该 Drive 里没有任何标注事件——**负样本由「无标注」定义，不由检测器定义** |
| `negative_hard` | 26 | 同一主体、机动开始前 40 帧以前的车道保持窗口，是最容易被误报的难负样本 |

## 每个 Drive 的漏斗

| drive | 可构建正样本 | 丢弃原因 |
|---|---|---|
| RECORD1_DRIVE1 | 41 | crossing not tracked 10, end not tracked 10, more than one lane 1, no track for the subject 3, same lane at both ends 1, start not tracked 6 |
| RECORD2_DRIVE1 | 14 | crossing not tracked 15, end not tracked 3, more than one lane 2, same lane at both ends 3, start not tracked 2 |
| RECORD5_DRIVE2 | 21 | crossing not tracked 17, end not tracked 3, more than one lane 5, no track for the subject 55, same lane at both ends 3, start not tracked 10 |

## 文件

- `clips/<clip_id>.mp4` — 8 秒原速片段，画出 `labels.txt` 的 2D 框（主体红粗框）
- `sheets/<clip_id>.png` — 6 帧连拍
- `samples.json` — 全部元数据（含 f0/f1/f2、车道转换、深度、转向灯）
- `labels.jsonl` — 与 S1 同字段的扁平标签
- `index.md` — 编号对照
- `overview_positives.jpg` / `overview_negatives.jpg` — 总览
- `REPORT.md` — 各路线的指标、失败机制与结论
- `scripts/` — 可复现的实验脚本，见 [`scripts/README.md`](scripts/README.md)
- `results/` — 逐条判定结果

媒体文件（`clips/`、`sheets*/`）与可重算的大结果（`s2_dets*.jsonl`、`da3_depth*.jsonl`）
不进版本库；`.md` 与 `.py` 进库，规则见仓库根目录 `.gitignore`。

## 字段

`f1` 是**跨线帧**——车尾中点正好压在两车道之间，852 条标注零缺失。
`f0` 是机动起点，有 22% 标注为 `-1`，此时本集用「`f1` 前 30 帧」作为起始帧。
片段以 `f1` 为锚点，前留 4.5 s、后留 3.5 s，所以**机动起点可能落在片段之外**。

## 必须记住的限制

1. **只有 76 条正样本**，少于 S2 原定的 100 条。想扩量只能放宽「最多跨一条车道」或改用 `ID = -1` 的 633 条窗口级事件，后者会让判据无法严格评测。
2. **`lanes.txt` 没有用在标注上。** `c0` 确实是米制车道线偏移，但在标注的跨线帧上，主体与最近检测线的距离中位 0.28 m、只有 41/57 落在 0.5 m 内，检测不全。车道归属用的是「横向偏移 / 3.5 m」的固定车道宽度模型。
3. **坐标系是相机系，不是道路系。** 相机装在标定车上、轴与车体对齐，所以横向偏移可近似为自车横向位置；但**自车自己变道会污染**这一假设，`cut_in`/`cut_out` 的划分在自车变道时可能出错。
4. **66% 的事件带转向灯**，而转向灯在画面里是像素可见的。评测时必须单独跑一路禁用转向灯线索的对照，否则会得到一个在生成视频上失效的分数。
5. **单一国家、单一标定车、以高速为主**，每帧中位只有 2 辆被跟踪的车。
