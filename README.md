# 虚拟声音展品：实时双耳渲染器

单声道声源放置在三维空间中，随听者头部朝向变化，按相对方向选取并**插值**
头相关冲激响应（HRIR），对左右耳分别做**真实分块卷积**输出。空间化来自
独立的双耳滤波器（ITD、ILD、头影、耳郭陷波、前后谱色、躯干反射），不是
左右音量声像调节。

## 坐标与角度约定

右手坐标系，听者位于原点：

| 轴 | 正方向 |
| --- | --- |
| +X | 正前方 |
| +Y | 左侧 |
| +Z | 正上方 |

* **方位角 azimuth**：在水平面内，正前方为 0°，向左为正（左耳 +90°，
  右耳 −90°，正后方为 ±180°，圆周连续）。
* **俯仰角 elevation**：水平面为 0°，向上为正，范围 −90°…+90°。
* 头部姿态用内在旋转 yaw（绕 Z）、pitch（绕 Y）、roll（绕 X）描述；
  世界坐标先经头部旋转矩阵变换到头坐标系，再求相对方位。
* 因此「头向左转 90°、声源在正前方」与「头不动、声源移到右侧
  （az −90°）」产生**逐样本相同**的渲染结果（自检 `head_turn_equals_source_motion`）。

## 算法与可测状态

* **卷积**：均匀分区 overlap-save 频域卷积（`convolution.py`）。FFT 长度
  `2B`，HRIR 按 `B` 分区，环形频谱槽数为不小于分区数 +1 的 2 的幂。
* **算法延迟**：恰好一个处理块 `B` 个样本（look-ahead 缓冲 + 显式输出
  延迟线）。`B=512 @48 kHz → 10.67 ms`。设备缓冲延迟另行统计，二者在
  `reported_latency()` 中分开报告。
* **延迟可测**：`measure_pipeline_latency()` 用单位冲激 HRIR 验证峰值
  精确落在第 `B` 个样本，误差 < 1e-12。
* **方向插值**：在方位/俯仰网格的复频谱上做双线性插值，再 irfft 截断为
  HRIR；方位网格半开周期化（−180…175°），±180° 接缝恒等。
* **连续切换**：方向变化时启用第二组卷积器，等功率余弦交叉淡化
  （默认 5 ms，不少于 2 个块），旧滤波器先排空再回收，避免爆音。
* **实时回调安全**：回调只索引预载入内存的 numpy 缓冲，无文件/网络 I/O；
  `audio_thread_io_audit()` 静态审计回调源码中的禁用调用。

## 用法

```bash
# 环境（仓库自带 .venv 时可直接使用 .venv/bin/python）
python -m venv .venv && . .venv/bin/activate
pip install numpy scipy soundfile sounddevice matplotlib pytest

# 数值自检（对称、正后方、真实卷积、延迟、头转等价、无爆音、回调无 I/O）
python -m binaural.app selftest

# 交互界面：拖动声源 / yaw 滑块 / front-left-right-back 预设
python -m binaural.app gui --signal speech
# 无显示或无 PortAudio 时自动降级为无头模拟场景并打印状态报告

# 离线导出三个信号 × 五个场景（前/左/右/后静态 + 全周扫描）
python -m binaural.app --block-size 512 export --signal broadband --out exports
python -m binaural.app --block-size 512 export --signal impulse   --out exports
python -m binaural.app --block-size 512 export --signal speech    --out exports

# 仅导出三个原始单声道信号
python -m binaural.app signals --out recordings

# 测试
pytest tests/ -q
```

可选 `--sofa /path/to/hrir.sofa` 载入 SOFA `SimpleFreeFieldHRIR` 实测
数据库（需要 `h5py`）；不提供时使用确定性的程序化通用模型
`procedural-generic-v1`。块长可选 64/128/256/512/1024（另允许 2048）。

## 导出物与复核

每次 `export` 在输出目录生成：

* `binaural_<signal>_<scenario>.wav`：真实卷积立体声结果（FLOAT WAV）；
* `hrirs/*.wav`：本次渲染实际使用的每个方向的 HRIR 对；
* `direction_log_*.csv`：逐块方位角、俯仰角、距离与姿态记录；
* `manifest_*.json`：块长、算法延迟、交叉淡化时长、HRTF 网格出处、
  每个文件的 **SHA256（对落盘 WAV 字节计算，可直接复核）**；
* `latency_check.json`：冲激法延迟测量结果。

## HRTF 个体差异（重要声明）

* 每个人的耳郭形状、头宽、躯干几何不同，实测 HRTF 数据库存在显著**个体
  差异**：同一方向的陷波频率、ITD 量级、前后谱色因人而异。
* 本系统的程序化模型是**通用假人模型，不是任何个人的测量结果**；即便
  载入某个受试者的 SOFA 数据，也**不能承诺每位听者都获得相同的定位
  精度**，前后混淆、上下混淆、内侧偏置等现象仍可能发生。
* 系统始终对当前方向**实时求值并卷积**；不使用预先录制好的空间音效
  替代实时计算，方向记录与所用滤波器全部留档可复核。
* 面向公众演示时应将其作为空间听觉原理展品，而非个体化验配设备。
