# video2colmap

这个文件夹用于把“围绕单个物体拍摄的一段视频”转换成训练仓库可直接读取的
COLMAP scene。它只负责视频抽帧、COLMAP 重建、去畸变和格式落盘；训练入口在
`3DGS_baseline01`。

## 这个仓库真正需要的数据格式

训练仓库的数据读取逻辑要求：

```text
your_scene/
├── images/
│   ├── frame_00001.jpg
│   ├── frame_00002.jpg
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

另外还有一个关键限制：

- 当前加载器只支持去畸变后的 `PINHOLE` 或 `SIMPLE_PINHOLE` 相机模型。
- 所以不能只做普通 COLMAP 重建，还必须再做一次 `image_undistorter`。

本文件夹里的脚本已经把这件事处理好了。

## 提前准备

你需要本机已经安装：

- `COLMAP`
- Python 依赖里的 `opencv-contrib-python`

检查方式：

```bash
colmap -h
python -c "import cv2; print(cv2.__version__)"
```

## 脚本说明

主脚本：

```bash
cd /home/haibo/Documents/Thesis/00_Baselines
PYTHONPATH=/home/haibo/Documents/Thesis/00_Baselines \
python -m data_preparation video2colmap --help
```

它会自动完成以下流程：

1. 从视频抽帧
2. 可选过滤模糊帧
3. 运行 COLMAP 特征提取
4. 运行 COLMAP 匹配
5. 运行 COLMAP 稀疏重建
6. 运行 COLMAP 去畸变
7. 整理成这个仓库可直接使用的目录格式

## 最常用命令

如果你的视频是手机环绕物体一圈拍的，先从这个命令开始：

```bash
PYTHONPATH=/home/haibo/Documents/Thesis/00_Baselines \
python -m data_preparation video2colmap \
  --video-path /path/to/object_video.mp4 \
  --output-dir /home/haibo/Documents/Thesis/04_ProcessedData/010_scenes_colmap/my_object_colmap \
  --sample-fps 3 \
  --matcher sequential \
  --max-image-size 1600 \
  --sift-gpu
```

输出完成后，`04_ProcessedData/010_scenes_colmap/my_object_colmap` 就可以直接给训练脚本使用。

## 然后怎么训练

训练仓库支持通过命令行覆盖训练数据目录，不需要为了切换场景去手改配置文件。

最直接的训练方式是：

```bash
cd /home/haibo/Documents/Thesis/00_Baselines/3DGS_baseline01
python train.py \
  --mode train \
  --data-format colmap \
  --data-dir /home/haibo/Documents/Thesis/04_ProcessedData/010_scenes_colmap/my_object_colmap \
  --save-dir /home/haibo/Documents/Thesis/05_Outputs/010_training_runs/my_object_colmap/colmap/default_run
```

## 参数建议

### 1. `--sample-fps`

建议范围：

- 视频很长：`2` 到 `3`
- 物体细节很多：`3` 到 `5`
- COLMAP 经常失败：先降到 `2`

如果抽帧太密，连续帧太像，反而会拖慢重建。

### 2. `--matcher`

- `sequential`：适合视频，默认推荐
- `exhaustive`：更慢，但有时对回环更稳

如果你绕物体一整圈，`sequential` 先试；如果注册视角明显不够，再试 `exhaustive`。

### 3. `--max-image-size`

- `1600`：比较稳妥
- `1920`：细节更多，但更慢
- `0`：不缩放，只有在显存和时间都充足时才建议

### 4. `--min-sharpness`

可以用来过滤运动模糊帧，比如：

```bash
--min-sharpness 80
```

如果你的视频本身比较清晰，可以先保持默认 `0`。

## 常见问题

### 1. COLMAP 成功了，但训练时报相机模型不支持

这个仓库只接受：

- `PINHOLE`
- `SIMPLE_PINHOLE`

本脚本已经通过 `image_undistorter` 把最终数据转成兼容格式。如果你手动跑了别的 COLMAP 流程，请确认最终拿去训练的是脚本输出目录，不是原始重建目录。

### 2. `Only X frames were extracted`

说明采样后剩下的图像太少。可以：

- 降低 `--sample-fps` 以外的过滤强度
- 关闭 `--min-sharpness`
- 换一段更长的视频

### 3. 稀疏重建失败或只注册了很少几张图

优先检查拍摄方式：

- 相机要围绕物体平稳移动
- 相邻帧要有足够重叠
- 尽量避免强反光、透明体、纯黑纯白背景
- 尽量避免背景也在大幅变化
- 物体不要被手频繁遮挡

然后再尝试：

- 把 `--sample-fps` 从 `3` 降到 `2`
- 把 `--matcher` 改成 `exhaustive`
- 把 `--max-image-size` 提高到 `1920`

## 中间文件

默认情况下，脚本会删除中间缓存，只保留训练真正需要的：

- `images/`
- `sparse/0/`

如果你想保留 COLMAP 原始数据库和原始重建结果，增加：

```bash
--keep-intermediate
```

这样会保留：

```text
.cache_preprocess/
```

## 一条更稳的示例命令

```bash
PYTHONPATH=/home/haibo/Documents/Thesis/00_Baselines \
python -m data_preparation video2colmap \
  --video-path /home/haibo/Documents/Thesis/03_Datasets/003_videos/my_turntable.mp4 \
  --output-dir /home/haibo/Documents/Thesis/04_ProcessedData/010_scenes_colmap/my_turntable_scene \
  --sample-fps 2.5 \
  --matcher exhaustive \
  --max-image-size 1600 \
  --min-sharpness 60 \
  --sift-gpu \
  --keep-intermediate
```
