import taichi as ti
import math

ti.init(arch=ti.gpu)

# -----------------------------
# 分辨率
# -----------------------------
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# -----------------------------
# 交互参数
# -----------------------------
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())

# -----------------------------
# 材质常量
# -----------------------------
MAT_DIFFUSE = 0
MAT_MIRROR = 1

EPS = 1e-4
INF = 1e10

# -----------------------------
# 工具函数
# -----------------------------
@ti.func
def normalize(v):
    n = v.norm()
    res = v
    if n > 1e-8:
        res = v / n
    return res

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])

    oc = ro - center
    a = rd.dot(rd)
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * a * c

    if delta > 0.0:
        s = ti.sqrt(delta)
        t1 = (-b - s) / (2.0 * a)
        t2 = (-b + s) / (2.0 * a)

        t_candidate = INF
        if t1 > EPS:
            t_candidate = t1
        elif t2 > EPS:
            t_candidate = t2

        if t_candidate < INF:
            t = t_candidate
            p = ro + rd * t
            normal = normalize(p - center)

    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])

    if ti.abs(rd.y) > 1e-6:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > EPS:
            t = t1

    return t, normal

@ti.func
def scene_intersect(ro, rd):
    min_t = INF
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 红球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0.0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.8, 0.1, 0.1])
        hit_mat = MAT_DIFFUSE

    # 镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0.0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0.0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE

        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)

        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

# -----------------------------
# 渲染
# -----------------------------
@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])

    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0

        ro = ti.Vector([0.0, 1.0, 5.0])
        rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

        final_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        for bounce in range(max_bounces[None]):
            t, N, obj_color, mat_id = scene_intersect(ro, rd)

            if t > INF * 0.5:
                final_color += throughput * bg_color
                break

            p = ro + rd * t

            if mat_id == MAT_MIRROR:
                ro = p + N * EPS
                rd = normalize(reflect(rd, N))
                throughput *= 0.8 * obj_color

            elif mat_id == MAT_DIFFUSE:
                L = normalize(light_pos - p)

                shadow_ray_orig = p + N * EPS
                shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)

                dist_to_light = (light_pos - p).norm()
                in_shadow = 0.0
                if shadow_t > 0.0 and shadow_t < dist_to_light:
                    in_shadow = 1.0

                ambient = 0.2 * obj_color
                direct_light = ambient

                if in_shadow == 0.0:
                    diff = ti.max(0.0, N.dot(L))
                    diffuse = 0.8 * diff * obj_color
                    direct_light += diffuse

                final_color += throughput * direct_light
                break

        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)

# -----------------------------
# 主程序
# -----------------------------
def main():
    window = ti.ui.Window("Ray Tracing Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 3

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.22):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)

        window.show()

if __name__ == '__main__':
    main()