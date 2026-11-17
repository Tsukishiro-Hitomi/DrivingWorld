# A1 误报人工复核（PREVENTION S2）

S2 的 145 条负样本里，A1 判据（`pred` 或 `pred_nc` 任一为 1）报了 28 条。
这里把这 28 条按 |横向位移| 从大到小排好，请你逐个看视频。

## 要判断的问题只有一个

**红框那辆车，在片段里有没有跨过一条车道线？**

- 框是粗红色 → 这才是被跟踪的主体车；细绿框只是场内其他车，不用管。
- 跨线了 → A1 是**对的**，这条负样本本身标错了（PREVENTION 漏标）。
- 没跨线 → A1 是**错的**，真误报。

## 一个必须知道的陷坑

A1 用的是相机坐标系。**如果自车（拍摄车）自己变道，画面里所有车都会横向滑动，
但红框车相对路面并没有变道。** 这时你看车道线会发现线在整幅画面里往一边走，
而红框车在两线之间待着不动 —— 这种算**没变道**，是 A1 的真误报。

## 怎么填

`review.md` 表里最后两列：`是否真变道`（是 / 否）、`方向`（左 / 右 / —）。

| # | clip_id | drive | 深度(m) | A1 横向位移(m) | 是否真变道 | 方向 | 备注 |
|---|---|---|---|---|---|---|---|
| 01 | `neg_record5drive2_13846` | RECORD5_DRIVE2 | 39 | -7.4 | 是| | |
| 02 | `neg_record5drive2_09115` | RECORD5_DRIVE2 | 18 | +6.1 | | | |
| 03 | `neg_record5drive2_06004` | RECORD5_DRIVE2 | 45 | -6.0 | | | |
| 04 | `neg_record2drive1_08011` | RECORD2_DRIVE1 | 56 | -5.8 | | | |
| 05 | `neg_record1drive1_10476` | RECORD1_DRIVE1 | 54 | -5.2 | | | |
| 06 | `neg_record2drive1_15472` | RECORD2_DRIVE1 | 42 | +4.8 | 是| | |
| 07 | `neg_record5drive2_11703` | RECORD5_DRIVE2 | 18 | -4.4 | 是| | |
| 08 | `neg_record5drive2_00176` | RECORD5_DRIVE2 | 58 | +4.4 | | | |
| 09 | `neg_record5drive2_13681` | RECORD5_DRIVE2 | 47 | +4.0 | | | |
| 10 | `neg_record2drive1_14001` | RECORD2_DRIVE1 | 31 | -3.6 | | | |
| 11 | `neg_record5drive2_14677` | RECORD5_DRIVE2 | 60 | -3.4 | | | |
| 12 | `neg_record5drive2_07291` | RECORD5_DRIVE2 | 4 | -3.3 | | | |
| 13 | `neg_record2drive1_09534` | RECORD2_DRIVE1 | 46 | +3.1 | | | |
| 14 | `neg_record5drive2_10070` | RECORD5_DRIVE2 | 13 | -3.0 | | | |
| 15 | `neg_record5drive2_06117` | RECORD5_DRIVE2 | 30 | -3.0 | 是| | |
| 16 | `neg_record5drive2_14183` | RECORD5_DRIVE2 | 94 | +2.9 | | | |
| 17 | `neg_record5drive2_10111` | RECORD5_DRIVE2 | 6 | -2.9 | | | |
| 18 | `neg_record5drive2_07044` | RECORD5_DRIVE2 | 27 | -2.3 | 是| | |
| 19 | `neg_record2drive1_19808` | RECORD2_DRIVE1 | 48 | +2.3 | | | |
| 20 | `neg_record1drive1_16679` | RECORD1_DRIVE1 | 15 | +2.2 | 是| | |
| 21 | `neg_record5drive2_14955` | RECORD5_DRIVE2 | 30 | -2.1 | | | |
| 22 | `neg_record2drive1_10742` | RECORD2_DRIVE1 | 52 | +2.1 | 是| | |
| 23 | `neg_record5drive2_14678` | RECORD5_DRIVE2 | 80 | +2.0 | | | |
| 24 | `neg_record2drive1_17659` | RECORD2_DRIVE1 | 33 | -1.9 | | | |
| 25 | `neg_record5drive2_04640` | RECORD5_DRIVE2 | 44 | -1.0 | | | |
| 26 | `neg_record5drive2_14661` | RECORD5_DRIVE2 | 83 | -0.8 | | | |
| 27 | `neg_record1drive1_19354` | RECORD1_DRIVE1 | 34 | +0.3 | | | |
| 28 | `neg_record2drive1_16518` | RECORD2_DRIVE1 | 31 | -0.0 | | | |

## 文件

- `clips/<clip_id>.mp4` — 8 秒原速片段，红框为主体车。**这是主证据，请以它为准**
- `sheets4/<clip_id>.png` — 同一片段取 4 帧原分辨率堆叠，红框和车道线都还看得清
- `sheets/<clip_id>.png` — 6 帧缩略连拍，只适合快速扫一遍
- `overview.jpg` — 28 条总览（带编号和 A1 的位移读数）
- `review.json` — 每条的程序侧指标（含 `jumps`：单帧横向跳变次数，
  >0 基本说明跟踪 ID 串车了，可作对照，不代表你的结论）

片段锚在跨线帧上、前留 4.5 s 后留 3.5 s，所以**负样本里没有跨线帧可对齐**，
片段中点不代表任何特定时刻。

## 复核结论（2026-11-15 20:01，人工逐条看完 28 条）

**7 条真的变道了** —— `#01 #06 #07 #15 #18 #20 #22`，都是 PREVENTION 漏标，
判据判对了；剩下 **21 条是误报**。

| 机制 | 条数 | 真变道 | 误报 |
|---|---|---|---|
| 跟踪 ID 串车（单帧横向跳 > 0.6 m） | 11 | 1 | 10 |
| 平滑横移、无跳变 | 14 | 6 | 8 |
| 只有加车流补偿才触发 | 5 | 1 | 4 |

结论：原始 FPR 0.19 里约三成是标注缺口。修正标注后（7 条漏标计入正样本）
判据的真实 FPR 为 **17/138 = 0.12**（不带补偿）/ 20/138 = 0.15（带补偿），
召回 79/83 = 0.95。数字见 `../lane_change_s2/results/adjudication.json`。
