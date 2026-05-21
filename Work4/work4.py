import taichi as ti

ti.init(arch=ti.gpu)

# =========================
# 分辨率与图像
# =========================
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# =========================
# 材质参数
# =========================
Ka = ti.field(dtype=ti.f32, shape=())
Kd = ti.field(dtype=ti.f32, shape=())
Ks = ti.field(dtype=ti.f32, shape=())
shininess = ti.field(dtype=ti.f32, shape=())

# =========================
# 工具函数
# =========================
@ti.func
def normalize(v):
    n = v.norm()
    out = v
    if n > 1e-6:
        out = v / n
    return out

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def clamp_color(c):
    return ti.math.clamp(c, 0.0, 1.0)

# =========================
# 球体求交
# =========================
@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    n = ti.Vector([0.0, 0.0, 0.0])

    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    disc = b * b - 4.0 * c

    if disc > 0.0:
        s = ti.sqrt(disc)
        t0 = (-b - s) * 0.5
        t1 = (-b + s) * 0.5

        if t0 > 1e-4:
            t = t0
        elif t1 > 1e-4:
            t = t1

        if t > 0.0:
            p = ro + rd * t
            n = normalize(p - center)

    return t, n

# =========================
# 圆锥求交
# apex 在上，底面在 base_y
# =========================
@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    t = -1.0
    n = ti.Vector([0.0, 0.0, 0.0])

    h = apex.y - base_y
    k = radius / h
    k2 = k * k

    # 以 apex 为原点的局部坐标系
    o = ro - apex
    d = rd

    # 锥面方程：x^2 + z^2 = k^2 * y^2
    # 其中局部 y 范围为 [-h, 0]
    A = d.x * d.x + d.z * d.z - k2 * d.y * d.y
    B = 2.0 * (o.x * d.x + o.z * d.z - k2 * o.y * d.y)
    C = o.x * o.x + o.z * o.z - k2 * o.y * o.y

    if ti.abs(A) > 1e-6:
        disc = B * B - 4.0 * A * C
        if disc > 0.0:
            s = ti.sqrt(disc)
            t0 = (-B - s) / (2.0 * A)
            t1 = (-B + s) / (2.0 * A)

            # 先尝试更近的根
            if t0 > 1e-4:
                y0 = o.y + t0 * d.y
                if y0 <= 0.0 and y0 >= -h:
                    t = t0

            if t < 0.0 and t1 > 1e-4:
                y1 = o.y + t1 * d.y
                if y1 <= 0.0 and y1 >= -h:
                    t = t1

            if t > 0.0:
                p_local = o + t * d

                # 侧面法线：F = x^2 + z^2 - k^2 y^2
                # grad F = (2x, -2k^2 y, 2z)
                n = normalize(ti.Vector([p_local.x, -k2 * p_local.y, p_local.z]))

    # 底面圆盘
    if ti.abs(d.y) > 1e-6:
        t_cap = (-h - o.y) / d.y
        if t_cap > 1e-4:
            p_cap = o + t_cap * d
            if p_cap.x * p_cap.x + p_cap.z * p_cap.z <= radius * radius:
                if t < 0.0 or t_cap < t:
                    t = t_cap
                    n = ti.Vector([0.0, -1.0, 0.0])

    return t, n

# =========================
# 渲染核心
# =========================
@ti.kernel
def render():
    for i, j in pixels:
        x = (i - 0.5 * res_x) / res_y * 2.0
        y = (j - 0.5 * res_y) / res_y * 2.0

        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([x, y, -1.0]))

        closest_t = 1e10
        hit_n = ti.Vector([0.0, 0.0, 0.0])
        hit_col = ti.Vector([0.0, 0.0, 0.0])

        # 球体
        t_s, n_s = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0.0 < t_s < closest_t:
            closest_t = t_s
            hit_n = n_s
            hit_col = ti.Vector([0.82, 0.12, 0.12])

        # 圆锥
        t_c, n_c = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0.0 < t_c < closest_t:
            closest_t = t_c
            hit_n = n_c
            hit_col = ti.Vector([0.62, 0.22, 0.82])

        # 背景
        color = ti.Vector([0.05, 0.15, 0.15])

        if closest_t < 1e9:
            p = ro + rd * closest_t

            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0])

            L = normalize(light_pos - p)
            V = normalize(ro - p)

            ambient = Ka[None] * light_color * hit_col
            diff_term = ti.max(0.0, hit_n.dot(L))
            diffuse = Kd[None] * diff_term * light_color * hit_col

            R = normalize(reflect(-L, hit_n))
            spec_term = ti.max(0.0, R.dot(V)) ** shininess[None]
            specular = Ks[None] * spec_term * light_color

            color = ambient + diffuse + specular

        pixels[i, j] = clamp_color(color)

# =========================
# 主程序
# =========================
def main():
    window = ti.ui.Window("Phong Shading Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    shininess[None] = 32.0

    while window.running:
        render()
        canvas.set_image(pixels)

        gui.begin("Material Parameters", 0.70, 0.05, 0.28, 0.22)
        Ka[None] = gui.slider_float("Ka", Ka[None], 0.0, 1.0)
        Kd[None] = gui.slider_float("Kd", Kd[None], 0.0, 1.0)
        Ks[None] = gui.slider_float("Ks", Ks[None], 0.0, 1.0)
        shininess[None] = gui.slider_float("Shininess", shininess[None], 1.0, 128.0)
        gui.end()

        window.show()

if __name__ == "__main__":
    main()