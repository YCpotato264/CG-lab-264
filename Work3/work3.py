import taichi as ti
import numpy as np

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

# 曲线上每个采样点点亮的方块半径（以像素计，GPU内核里使用）
POINT_PIXEL_RADIUS = 1                  # 0 -> 单像素；1 -> 3x3

# canvas circles/lines参数；归一化比例 (0..1)
USE_PIXEL_RADIUS = False                # 使用像素单位就设为 True
CIRCLES_RADIUS_NORM = 0.015             # 归一化：0.015 * WIDTH ≈ 12 px (在 800 宽度)
LINES_WIDTH_NORM = 0.002

# 像素单位模式参数（当 USE_PIXEL_RADIUS=True）
CIRCLES_RADIUS_PIXEL = 6
LINES_WIDTH_PIXEL = 1

# ------- Taichi fields -------
pixels = ti.Vector.field(3, ti.f32, shape=(WIDTH, HEIGHT))        # pixels[x,y] = rgb
curve_points_field = ti.Vector.field(2, ti.f32, shape=(NUM_POINTS_IN_BUFFER,))
gui_points = ti.Vector.field(2, ti.f32, shape=(MAX_CONTROL_POINTS,))
gui_indices = ti.field(dtype=ti.i32, shape=(MAX_CONTROL_POINTS * 2,))  # 存线段索引对

# ------- 内核：并行清屏 & 并行点亮曲线像素 -------
@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

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
                    # 直接覆盖为绿色
                    pixels[xi, yi] = ti.Vector([0.0, 1.0, 0.0])

# ------- CPU：De Casteljau（计算曲线采样点） -------
def de_casteljau(points, t):
    # points: list of (x,y) 归一化
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
    window = ti.ui.Window("Bézier Curve - De Casteljau", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points = []     # Python 列表存归一化坐标
    gui_np_pool = np.ones((MAX_CONTROL_POINTS, 2), dtype=np.float32) * -2.0
    gui_indices_np = np.zeros((MAX_CONTROL_POINTS * 2,), dtype=np.int32)

    # 选择用于 canvas 的 radius/width 值
    if USE_PIXEL_RADIUS:
        circles_radius = CIRCLES_RADIUS_PIXEL
        lines_width = LINES_WIDTH_PIXEL
    else:
        circles_radius = CIRCLES_RADIUS_NORM
        lines_width = LINES_WIDTH_NORM

    while window.running:
        # 清显存像素缓冲（并行）
        clear_pixels()

        # 事件处理：鼠标左键添加点，按 c 清空
        if window.get_event(ti.ui.PRESS):
            if window.is_pressed(ti.ui.LMB):
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()   # 归一化坐标 (x,y)
                    control_points.append((pos[0], pos[1]))
            if window.is_pressed('c'):
                control_points.clear()

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
            # 在 GPU 上并行点亮像素
            draw_curve_kernel(NUM_POINTS_IN_BUFFER, POINT_PIXEL_RADIUS)

        # 把 pixels 放到 canvas（曲线已经绘制在 pixels 中）
        canvas.set_image(pixels)

        # 准备并上传用于绘制控制点的对象池（前 n_ctrl 有效，其余留空）
        gui_np_pool.fill(-2.0)
        for i, p in enumerate(control_points):
            gui_np_pool[i, 0] = p[0]
            gui_np_pool[i, 1] = p[1]
        gui_points.from_numpy(gui_np_pool)

        # 准备线段索引，避免未使用点参与连线（更稳健）
        # indices 格式：[0,1, 1,2, 2,3, ...] (成对出现)
        if n_ctrl >= 2:
            idx_list = []
            for i in range(n_ctrl - 1):
                idx_list.extend([i, i + 1])
            # 把 idx_list 放进固定大小的 np 数组，然后上传
            gui_indices_np.fill(0)
            gui_indices_np[:len(idx_list)] = np.array(idx_list, dtype=np.int32)
            gui_indices.from_numpy(gui_indices_np)
            # 绘制控制多边形（灰线）
            try:
                canvas.lines(gui_points, width=lines_width, indices=gui_indices, color=(0.6, 0.6, 0.6))
            except Exception:
                # 兜底：使用更小的归一化 width
                canvas.lines(gui_points, width=0.001, indices=gui_indices, color=(0.6, 0.6, 0.6))

        # 绘制控制点（红色）
        if n_ctrl > 0:
            try:
                canvas.circles(gui_points, radius=circles_radius, color=(1.0, 0.0, 0.0))
            except Exception:
                # 兜底：小归一化半径，避免大半径把屏幕覆盖
                canvas.circles(gui_points, radius=0.008, color=(1.0, 0.0, 0.0))

        window.show()

if __name__ == "__main__":
    main()