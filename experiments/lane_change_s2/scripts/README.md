# S2 前车变道判定的实验脚本

这些脚本原先散在仓库根目录，以 `.tmp_*` 命名、并把开发机绝对路径写死在文件里。
现在统一收到这里，命名按「路线 + 动作」组织，路径由 [`_paths.py`](_paths.py) 从文件位置推导，
本地与开发机可以跑同一份代码。

```bash
# 任意工作目录都能跑；脚本会自己找到仓库根
python experiments/lane_change_s2/scripts/eval_s2.py a1
```

覆盖路径用环境变量：

| 变量 | 作用 |
|---|---|
| `DRIVINGWORLD_ROOT` | 仓库根，默认由本目录向上推两层 |
| `DRIVINGWORLD_MODEL_CACHE` | 权重目录，默认 `/xs-fd-nas/workspace_zhaoxinrui/model_cache` |

实验结论与指标见 [`../REPORT.md`](../REPORT.md)；数据集本身见 [`../README.md`](../README.md)。

## 流水线

下面的顺序就是依赖顺序，箭头右侧表示产物。

### 1. 几何基线（不需要 GPU）

| 脚本 | 命令 | 产物 |
|---|---|---|
| `eval_s2.py` | `python .../eval_s2.py a1` | `results/a1.jsonl`：GT 轨迹几何判据逐条结果 |
| `eval_s2.py` | `python .../eval_s2.py report` | 终端按 stratum 汇总召回 / 误报 |

`a1` 依赖 `samples.json` 与 `labels.jsonl`，以及 `legacy_dir("prevention_annot")`。
它是所有感知路线的上限参照，不是可部署方案。

### 2. 检测器

| 脚本 | 命令 | 产物 |
|---|---|---|
| `detect.py` | `python .../detect.py --shard 0 --nshards 4` | `results/s2_dets.p<i>.jsonl` |
| — | `cat results/s2_dets.p*.jsonl > results/s2_dets.jsonl` | `results/s2_dets.jsonl` |
| `analyze_detections.py` | `python .../analyze_detections.py` | `results/s2_dets_report.json`：相机拟合 + 已知车宽几何 |

`detect.py` 用 YOLO11m 逐帧检测，输出每个框的中心与宽高。
离线评测用视频里烘焙的红框中心把检测结果关联到标注主体；**红框不参与任何度量**。
部署到 loop 时，目标关联要由跟踪 / 目标选择模块给出。

`analyze_detections.py` 里的 `cx0 / f` 是按 drive 用 GT 拟合一次得到的，
相当于 loop 中由 proxy 相机提供的标定。

### 3. 消融

| 脚本 | 命令 | 内容 |
|---|---|---|
| `sweep_detectors.py` | `python .../sweep_detectors.py` | 车宽先验、平滑窗口、框高/框宽等几何变体 |
| `sweep_postprocess.py` | `python .../sweep_postprocess.py` | 在缓存检测上做后处理扫描，看误报能否便宜地压掉 |

两者只读 `results/s2_dets.jsonl`，不解码视频。

### 4. 胶片条 + VLM

| 脚本 | 命令 | 产物 |
|---|---|---|
| `build_film.py` | `python .../build_film.py` | `sheets_film/`：第一版主体裁剪胶片条（裁剪窗曾退化到接近整帧） |
| `eval_film.py` | `python .../eval_film.py qwen --tag film_qwen` | `results/film_qwen.jsonl` |
| `luna_film.py` | `python .../luna_film.py` | `results/film_luna.jsonl`（逐张调用 Codex CLI 的 GPT-6-Luna） |
| `build_film_normalized.py` | `python .../build_film_normalized.py --shard 0 --nshards 4` | `sheets_film1b/`、`results/film1b_boxes.jsonl` |
| `probe_film_normalized.py` | `python .../probe_film_normalized.py --n 8` | 带调试叠加的抽检样张 |
| `eval_film_normalized.py` | `python .../eval_film_normalized.py qwen --tag film1b_qwen` | `results/film1b_qwen.jsonl` |
| `eval_luna.py` | `python .../eval_luna.py` | `results/luna.jsonl`：整帧 contact sheet 的 Luna 判定 |
| `report_luna.py` | `python .../report_luna.py` | 终端汇总 Luna 的分层指标 |
| `check_sign.py` | `python .../check_sign.py` | Luna 读数符号与真实方向的一致率 |

`build_film_normalized.py` 用检测框把裁剪窗归一到 3 倍框宽，并画上「1 车宽」标尺——
这一版才真正把裁剪做在目标车上，因此和第一版 `build_film.py` 的结论不能混用。

### 5. DA3 米制深度

| 脚本 | 命令 | 产物 |
|---|---|---|
| `da3_selftest.py` | `python .../da3_selftest.py` | 用随机权重验证调用链，不需要 checkpoint |
| `da3_load.py` | `python .../da3_load.py` | 验证真实权重输出的深度是否具备米制性质 |
| `da3_speed.py` | `python .../da3_speed.py` | 稳态吞吐 |
| `da3_depth.py` | `python .../da3_depth.py --shard 0 --nshards 4 --stride 2` | `results/da3_depth.p<i>.jsonl` → 合并成 `results/da3_depth.jsonl` |
| `da3_score.py` | `python .../da3_score.py` | 深度误差分档 + 判据指标 |
| `da3_rescale.py` | `python .../da3_rescale.py` | 用 GT 反推焦距换算常数 `c` |
| `da3_rules.py` | `python .../da3_rules.py` | 幅值带 / 离散车道 / 端点稳定等判定规则的工作点 |

`da3_rescale.py` 是本路线的关键一步：DA3 输出的是 canonical depth，
不做 `z = z_da3 * f / c` 换算会得到「几乎从不报警」的假象。
三个 drive 拟合出的 `c` 为 926 / 1040 / 1004，全局工作值 982。

### 6. 车道线路线（已停止，仅留档）

| 脚本 | 命令 | 内容 |
|---|---|---|
| `lane_ceiling.py` | `python .../lane_ceiling.py` | 纯 GT 离散车道判据的天花板 |
| `lane_crossing.py` | `python .../lane_crossing.py --shard 0 --nshards 4 --tag lanecross_eval` | 图像空间车道线拟合 + 跨线判定 |
| `probe_lane_crossing.py` | `python .../probe_lane_crossing.py` | 近场车道线拟合的数值探针 |

这条路线已按决策停止：GT 天花板的最优误报没能优于 DA3，真实车道线可见性也不稳。
脚本保留是为了复现「为什么不继续」，不要在此基础上继续开发。

### 7. 横向对照

| 脚本 | 命令 | 内容 |
|---|---|---|
| `compare_methods.py` | `python .../compare_methods.py` | 在都产出 1b 胶片条的子集上，同一张表对比全部方法 |

这是唯一应该被引用的横向对比入口，避免不同路线在各自子集上各说各话。

## 产物与版本控制

- `results/*.jsonl`、`results/*.json`：逐条判定结果，**入库**，是结论的可核查证据。
- `results/s2_dets*.jsonl`、`results/da3_depth*.jsonl`、`results/film1b_boxes*.jsonl`：体积大且可由脚本重算，**不入库**。
- `sheets*/`、`clips/`、`scratch/`：生成物，不入库。

## 已知坑

- 单卡机器上 `CUDA_VISIBLE_DEVICES=1`；开发机 0 号卡常被占用。
- 用 `conda run` 传 `--n` 这类短参数会被 conda 自己吃掉，用解释器绝对路径：
  `/root/miniconda3/envs/h3-ref2va/bin/python`。
- 分片脚本都写 `*.p<i>.jsonl`，合并前先确认四个分片都在，否则指标会静默偏低。
- `depth_anything_3` 用 0.1.1 的本地 wheel 安装；torch < 2.6 时 diffusers 的加速器探测会失败，
  需要 `accel_shim` 这类兼容层。权重放 `$DRIVINGWORLD_MODEL_CACHE/models/da3metric-large/`。
- 写文件务必用 UTF-8：这批脚本曾经因为按系统代码页读写，把中文报告写成过乱码。
