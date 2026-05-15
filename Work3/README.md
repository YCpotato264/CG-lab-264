# 实验三：Bézier曲线交互
本文将对本次实验的代码逻辑、实现功能以及优化升级进行介绍。程序基于 Taichi，支持用鼠标交互添加控制点、实时生成贝塞尔曲线并在 GUI 中展示控制多边形与曲线。

## 目录

- [项目概述](#项目概述)
- [代码逻辑](#代码逻辑)
  - [`de_casteljau(points, t)`](#de_casteljaupoints-t)
  - [`draw_curve_kernel(n, radius)` / `clear_pixels()`](#draw_curve_kerneln-radius--clear_pixels)
  - [主循环与绘制流程](#主循环与绘制流程)
- [实现功能](#实现功能)
- [视频演示](#视频演示)
- [实验结果说明](#实验结果说明)
- [优化](#优化)

## 项目概述

本项目实现了一个交互式的 Bézier 曲线绘制工具，基于 Taichi 进行 GPU（或 CPU 回退）渲染。用户通过鼠标左键在画布上添加控制点（红色圆点），程序实时将控制点以灰线连接成控制多边形，并使用 De Casteljau 算法在 CPU 上采样曲线点、批量上传到 GPU，再由 GPU 内核并行点亮像素以绘制绿色贝塞尔曲线。按键 c 可清空控制点并重置画面。

---

## 代码逻辑

整体代码可分为三部分：曲线采样（CPU）、像素写入（GPU 内核）与 GUI 交互/绘制。

---

### de_casteljau(points, t)
功能：在 CPU 端实现 De Casteljau 算法，用于对 n 次 Bézier 曲线按参数 t（0..1）求出对应点。

- 输入：
  - points：控制点列表（归一化坐标 (x, y)，范围 0..1）
  - t：0..1 的参数
- 输出：采样点 (x, y)

```python
def de_casteljau(points, t):
    pts = [np.array(p, dtype=np.float32) for p in points]
    if len(pts) == 0:
        return (0.0, 0.0)
    while len(pts) > 1:
        next_pts = []
        for i in range(len(pts) - 1):
            next_pts.append((1.0 - t) * pts[i] + t * pts[i + 1])
        pts = next_pts
    final = pts[0]
    return float(final[0]), float(final[1])
```

作用与说明：
- 以稳定、可读的纯 Python 实现采样（便于调试）；
- 在每帧把 NUM_SEGMENTS+1 个采样点一次性计算并写入 numpy 数组，再批量上传给 Taichi Field。

---

### draw_curve_kernel(n, radius) / clear_pixels()
功能：GPU 内核并行处理像素缓冲——清屏与把采样点映射为像素并点亮。

- clear_pixels(): 并行将 pixels 清为背景色（例如黑色）。
- draw_curve_kernel(n, radius): 从 GPU 缓冲 curve_points_field 读取 n 个归一化坐标，映射到像素网格并以给定像素半径点亮（默认覆盖为绿色）。

关键点：
- 把“点亮像素”的工作放到 Taichi kernel 中并行执行，避免 CPU 循环逐点调用 GPU；
- radius 以像素计（内核里做 int 转换），可通过调节使曲线更连续（例如 radius=1 表示 3x3 方块）。

```python
@ti.kernel
def draw_curve_kernel(n: ti.i32, radius: ti.i32):
    for i in range(n):
        p = curve_points_field[i]
        cx = ti.cast(p[0] * WIDTH, ti.i32)
        cy = ti.cast(p[1] * HEIGHT, ti.i32)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                xi = cx + dx
                yi = cy + dy
                if 0 <= xi < WIDTH and 0 <= yi < HEIGHT:
                    pixels[xi, yi] = ti.Vector([0.0, 1.0, 0.0])
```

---

### 主循环与绘制流程

主循环职责：
1. 处理事件（鼠标、按键）：
   - 鼠标左键：在当前鼠标位置（归一化）添加控制点（限制为 MAX_CONTROL_POINTS）。
   - 按键 c：清空控制点并重置画面。
2. 清空像素缓冲（调用 clear_pixels()）。
3. 若控制点数 >= 2：
   - 在 CPU 上用 de_casteljau 对 t ∈ [0,1] 均匀采样 NUM_SEGMENTS+1 个点，写入 numpy 数组；
   - 一次性把 numpy 数组上传到 curve_points_field；
   - 调用 draw_curve_kernel 在 GPU 上并行点亮相应像素（绿色）。
4. 将 pixels 作为图像显示到 canvas（canvas.set_image）。
5. 准备 gui_points（对象池，未使用项放到画布外，例如 (-2,-2)）并上传；
6. 绘制控制多边形（灰色折线）与控制点（红色圆点）到 canvas。

绘制顺序说明：
- 先 canvas.set_image(pixels)（确保曲线格像素显示），再绘制 canvas.lines / canvas.circles 使控制多边形与控制点覆盖在曲线之上，提高可见性。

---

## 实现功能

本项目实现了以下功能：

- 交互式添加控制点（鼠标左键）并可实时查看效果；
- 实时生成并渲染 Bézier 曲线（De Casteljau 算法采样，CPU 计算，批量上传 GPU 并行绘制）；
- 实时绘制控制多边形（灰线）与控制点（红色圆点）；
- 按 c 清空控制点并重置画面；

---

## 视频演示

![Bézier曲线交互](Work3_show.gif)

---

## 实验结果说明

运行程序后会弹出 `800 × 800` 的窗口，实验结果特点如下：

- 初始为空黑背景；
- 每次左键点击添加红色控制点，控制点按插入顺序在画布显示；
- 控制点通过灰色折线连接形成控制多边形，曲线根据当前控制多边形实时更新并以绿色像素点绘制；
- 曲线平滑度与连通性依赖采样数量（NUM_SEGMENTS）与 POINT_PIXEL_RADIUS；
- 按键 c 可清空所有控制点并恢复初始画面。

---

## 优化

### 目的
- 消除由于坐标量化（float -> int）造成的锯齿（jaggies），提升曲线边缘的平滑度与视觉质量；
- 在保持原有交互逻辑（鼠标添加控制点、按 c 清空）的前提下，实现 GPU 并行的亚像素权重累加方案；
- 在可控的性能开销内（用户可调 AA 半径与采样密度）提供可接受的实时帧率。
    - 当前实验设置：AA_RADIUS = 2（像素），表示每个采样点影响半径为 2 像素，邻域约为 (2*2+1)^2 = 25 个像素。

---

### 实现改动
- 新增累加缓冲 accum（同尺寸 RGB float field），用于在 kernel 中用原子加法累积每个采样点对周边像素的颜色贡献；
- 修改曲线绘制 kernel：对每个采样点，不再只写入一个像素，而是在其附近 r = ceil(AA_RADIUS) 范围内遍历像素，计算像素中心与精确子像素位置的距离，根据权重函数分配贡献并用 ti.atomic_add 累加；
- 新增合成步骤：将 accum 的累加值做亮度压缩/裁剪后写入 pixels 以供 canvas.set_image 使用；
- 保留原有 CPU 端 De Casteljau 采样、批量上传、以及 canvas 上绘制控制点/控制多边形的逻辑（只改变像素点亮逻辑，不影响交互接口）；
- 兼顾实现的易用性与可调性：AA_RADIUS、权重函数、亮度压缩函数均可快速替换与调参。

---

### 关键代码片段
全局参数（AA_RADIUS = 2）
- 反走样参数
```python
AA_RADIUS = 2.0            # 影响半径（像素）
r_int = math.ceil(AA_RADIUS)  # = 2
R_f32 = float(AA_RADIUS)      # 传给 kernel 的浮点半径
累加 & 清空 buffers
@ti.kernel
def clear_buffers():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])
        accum[i, j]  = ti.Vector([0.0, 0.0, 0.0])
线性权重的 AA kernel（核心：每个采样点影响邻域并原子累加）
@ti.kernel
def draw_curve_kernel(n: ti.i32, r_int: ti.i32, R: ti.f32):
    # 线性衰减核： w = max(0, 1 - dist / R)
    for idx in range(n):
        p = curve_points_field[idx]
        fx = p[0] * WIDTH    # 浮点像素坐标
        fy = p[1] * HEIGHT
        cx = ti.cast(ti.floor(fx), ti.i32)
        cy = ti.cast(ti.floor(fy), ti.i32)
        for dx in range(-r_int, r_int + 1):
            xi = cx + dx
            if not (0 <= xi < WIDTH):
                continue
            px_center_x = ti.cast(xi, ti.f32) + 0.5
            dx_f = px_center_x - fx
            for dy in range(-r_int, r_int + 1):
                yi = cy + dy
                if not (0 <= yi < HEIGHT):
                    continue
                py_center_y = ti.cast(yi, ti.f32) + 0.5
                dy_f = py_center_y - fy
                dist = ti.sqrt(dx_f * dx_f + dy_f * dy_f)
                if dist < R:
                    w = 1.0 - dist / R  # 线性权重
                    # 曲线为纯绿色：只对 G 分量累加
                    ti.atomic_add(accum[xi, yi][1], w)
```

- 合成并写入显示像素（带亮度压缩以避免过曝）
```python
@ti.kernel
def composite_to_pixels():
    # 把累加缓冲裁剪并做简单的亮度压缩映射（示例：1 - exp(-k * val)）
    k = 1.5  # 亮度压缩系数，可调
    for i, j in pixels:
        g = accum[i, j][1]
        # 非线性压缩：避免累加造成高亮溢出，同时保留感知上的线宽
        g_out = 1.0 - ti.exp(-k * g)
        if g_out > 1.0:
            g_out = 1.0
        pixels[i, j] = ti.Vector([0.0, g_out, 0.0])
```
---
### 视频演示
![Bézier曲线交互（返走样）](Work3_update_show.gif)