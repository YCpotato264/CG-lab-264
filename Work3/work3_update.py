import taichi as ti
import numpy as np
import math

# 尝试GPU，失败回退CPU
try:
    ti.init(arch=ti.gpu)
except Exception as e:
    print("Warning: GPU init failed, falling back to CPU. Reason:", e)
    ti.init(arch=ti.cpu)

# ------- 配置 -------
WIDTH, HEIGHT = 800, 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000                     # 曲线采样点数量
NUM_POINTS_IN_BUFFER = NUM_SEGMENTS + 1

# 反走样相关参数（亚像素半径）
AA_RADIUS = 2          # 影响半径（像素），可调
# 使用 canvas circles/lines 参数（归一化）
USE_PIXEL_RADIUS = False
CIRCLES_RADIUS_NORM = 0.015
LINES_WIDTH_NORM = 0.002
CIRCLES_RADIUS_PIXEL = 6
LINES_WIDTH_PIXEL = 1

# ------- Taichi fields -------
# fields 的索引方式为 pixels[x, y]
pixels = ti.Vector.field(3, ti.f32, shape=(WIDTH, HEIGHT))        # 最终要显示的图像
# 累加缓冲（用于存放多个采样点贡献的累加值，带原子加法）
accum = ti.Vector.field(3, ti.f32, shape=(WIDTH, HEIGHT))

curve_points_field = ti.Vector.field(2, ti.f32, shape=(NUM_POINTS_IN_BUFFER,))
gui_points = ti.Vector.field(2, ti.f32, shape=(MAX_CONTROL_POINTS,))
gui_indices = ti.field(dtype=ti.i32, shape=(MAX_CONTROL_POINTS * 2,))  # 存线段索引对

# ------- 内核：并行清屏 & 并行点亮曲线像素（带反走样权重） -------
@ti.kernel
def clear_buffers():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])
        accum[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32, r_int: ti.i32, R: ti.f32):
    # 使用线性衰减核：w = max(0, 1 - d / R)
    # 对每个采样点在邻域内加权贡献（使用原子加法累加到 accum）
    for idx in range(n):
        p = curve_points_field[idx]
        # 子像素浮点位置（像素坐标系）
        fx = p[0] * WIDTH
        fy = p[1] * HEIGHT
        cx = ti.cast(ti.floor(fx), ti.i32)
        cy = ti.cast(ti.floor(fy), ti.i32)
        # 遍历邻域（-r_int ... r_int）
        for dx in range(-r_int, r_int + 1):
            xi = cx + dx
            if not (0 <= xi < WIDTH):
                continue
            # 计算 x 方向偏移到像素中心（像素中心假设为 i + 0.5）
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
                    w = 1.0 - dist / R  # 线性内核，离得越近权重越高
                    # 颜色贡献：绿色曲线
                    # 使用原子加法避免线程写冲突
                    ti.atomic_add(accum[xi, yi][0], 0.0)            # R 分量（曲线为纯绿）
                    ti.atomic_add(accum[xi, yi][1], w)              # G 分量
                    ti.atomic_add(accum[xi, yi][2], 0.0)            # B 分量

@ti.kernel
def composite_to_pixels():
    # 把累加缓冲裁剪到 [0,1] 并写入 pixels
    for i, j in pixels:
        col = accum[i, j]
        r = col[0]
        g = col[1]
        b = col[2]
        # Clamp
        if r > 1.0:
            r = 1.0
        if g > 1.0:
            g = 1.0
        if b > 1.0:
            b = 1.0
        pixels[i, j] = ti.Vector([r, g, b])

# ------- CPU：De Casteljau（计算曲线采样点） -------
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

# ------- 主循环 -------
def main():
    window = ti.ui.Window("Bézier Curve - Anti-aliased", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points = []                           # Python 列表存归一化坐标
    gui_np_pool = np.ones((MAX_CONTROL_POINTS, 2), dtype=np.float32) * -2.0
    gui_indices_np = np.zeros((MAX_CONTROL_POINTS * 2,), dtype=np.int32)

    # 选择用于 canvas 的 radius/width 值
    if USE_PIXEL_RADIUS:
        circles_radius = CIRCLES_RADIUS_PIXEL
        lines_width = LINES_WIDTH_PIXEL
    else:
        circles_radius = CIRCLES_RADIUS_NORM
        lines_width = LINES_WIDTH_NORM

    # 内核用的邻域整数半径
    r_int = math.ceil(AA_RADIUS)
    R_f32 = float(AA_RADIUS)

    while window.running:
        # 事件处理（LMB 添加，c 清空）
        if window.get_event(ti.ui.PRESS):
            if window.is_pressed(ti.ui.LMB):
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append((pos[0], pos[1]))
            if window.is_pressed('c'):
                control_points.clear()

        # 清空 buffers
        clear_buffers()

        # 计算贝塞尔曲线并上传（当控制点 >= 2 时）
        n_ctrl = len(control_points)
        if n_ctrl >= 2:
            np_curve = np.zeros((NUM_POINTS_IN_BUFFER, 2), dtype=np.float32)
            for i in range(NUM_POINTS_IN_BUFFER):
                t = i / NUM_SEGMENTS
                x, y = de_casteljau(control_points, t)
                # clamp
                x = min(max(x, 0.0), 1.0)
                y = min(max(y, 0.0), 1.0)
                np_curve[i, 0] = x
                np_curve[i, 1] = y
            # 批量上传到 GPU field
            curve_points_field.from_numpy(np_curve)
            # GPU 并行计算反走样贡献并累加到 accum
            draw_curve_kernel(NUM_POINTS_IN_BUFFER, r_int, R_f32)
            # 把累加结果合成到 pixels（并 clamp）
            composite_to_pixels()

        # 把 pixels 放到 canvas（曲线/反走样已写入 pixels）
        canvas.set_image(pixels)

        # 准备并上传用于绘制控制点的对象池（前 n_ctrl 有效，其余留空）
        gui_np_pool.fill(-2.0)
        for i, p in enumerate(control_points):
            gui_np_pool[i, 0] = p[0]
            gui_np_pool[i, 1] = p[1]
        gui_points.from_numpy(gui_np_pool)

        # 绘制控制多边形（灰线）
        if n_ctrl >= 2:
            idx_list = []
            for i in range(n_ctrl - 1):
                idx_list.extend([i, i + 1])
            gui_indices_np.fill(0)
            gui_indices_np[:len(idx_list)] = np.array(idx_list, dtype=np.int32)
            gui_indices.from_numpy(gui_indices_np)
            try:
                canvas.lines(gui_points, width=lines_width, indices=gui_indices, color=(0.6, 0.6, 0.6))
            except Exception:
                canvas.lines(gui_points, width=0.001, indices=gui_indices, color=(0.6, 0.6, 0.6))

        # 绘制控制点（红色）
        if n_ctrl > 0:
            try:
                canvas.circles(gui_points, radius=circles_radius, color=(1.0, 0.0, 0.0))
            except Exception:
                canvas.circles(gui_points, radius=0.008, color=(1.0, 0.0, 0.0))

        window.show()

if __name__ == '__main__':
    main()