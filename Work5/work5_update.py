import taichi as ti
import math

ti.init(arch=ti.vulkan)

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
spp = ti.field(ti.i32, shape=())

# -----------------------------
# 材质常量
# -----------------------------
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

EPS = 1e-4
INF = 1e10
IOR_GLASS = 1.5

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
def refract(I, N, eta):
    cosi = ti.max(-1.0, ti.min(1.0, I.dot(N)))
    sint2 = eta * eta * (1.0 - cosi * cosi)
    T = ti.Vector([0.0, 0.0, 0.0])
    tir = False
    if sint2 > 1.0:
        tir = True
    else:
        cost = ti.sqrt(ti.max(0.0, 1.0 - sint2))
        T = eta * I - (eta * cosi + cost) * N
    return T, tir

@ti.func
def fresnel_schlick(cos_theta, ior):
    r0 = (1.0 - ior) / (1.0 + ior)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * ti.pow(1.0 - cos_theta, 5.0)

@ti.func
def beer_lambert(color, dist):
    # 玻璃吸收：距离越长，透过越少
    absorption = ti.Vector([0.12, 0.05, 0.02])
    return ti.Vector([
        ti.exp(-absorption.x * dist),
        ti.exp(-absorption.y * dist),
        ti.exp(-absorption.z * dist)
    ]) * color

@ti.func
def checker_color(p):
    grid_scale = 2.0
    ix = ti.cast(ti.floor(p.x * grid_scale), ti.i32)
    iz = ti.cast(ti.floor(p.z * grid_scale), ti.i32)

    col = ti.Vector([0.0, 0.0, 0.0])
    if (ix + iz) % 2 == 0:
        col = ti.Vector([0.25, 0.25, 0.25])
    else:
        col = ti.Vector([0.85, 0.85, 0.85])
    return col

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

    # 玻璃球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0.0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.98, 0.99, 1.0])
        hit_mat = MAT_GLASS

    # 镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0.0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.95])
        hit_mat = MAT_MIRROR

    # 地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0.0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        hit_c = checker_color(p)

    return min_t, hit_n, hit_c, hit_mat

@ti.func
def is_in_shadow(p, light_p):
    to_light = light_p - p
    dist = to_light.norm()
    shadow = False
    if dist > 1e-8:
        ldir = to_light / dist
        t, _, _, _ = scene_intersect(p, ldir)
        if 0.0 < t < dist:
            shadow = True
    return shadow

@ti.func
def shade_diffuse(p, n, albedo, light_p):
    ambient = 0.12 * albedo
    col = ambient

    if not is_in_shadow(p + n * EPS, light_p):
        ldir = normalize(light_p - p)
        ndotl = ti.max(0.0, n.dot(ldir))
        diffuse = albedo * ndotl

        view_dir = normalize(ti.Vector([0.0, 1.0, 5.0]) - p)
        half_dir = normalize(ldir + view_dir)
        ndoth = ti.max(0.0, n.dot(half_dir))
        spec = ti.Vector([1.0, 1.0, 1.0]) * (ndoth ** 32.0) * 0.28

        dist2 = (light_p - p).dot(light_p - p)
        atten = 1.0 / (1.0 + 0.03 * dist2)

        col = ambient + atten * (diffuse + spec)

    return col

@ti.func
def background(rd):
    # 渐变天空，让玻璃更容易显形
    t = 0.5 * (rd.y + 1.0)
    return (1.0 - t) * ti.Vector([0.08, 0.12, 0.18]) + t * ti.Vector([0.55, 0.7, 0.9])

@ti.func
def trace_ray(ro, rd, light_p):
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    for bounce in range(max_bounces[None]):
        t, N, obj_color, mat_id = scene_intersect(ro, rd)

        if t > INF * 0.5:
            final_color += throughput * background(rd)
            break

        p = ro + rd * t

        if mat_id == MAT_DIFFUSE:
            c = shade_diffuse(p, N, obj_color, light_p)
            final_color += throughput * c
            break

        elif mat_id == MAT_MIRROR:
            ro = p + N * EPS
            rd = normalize(reflect(rd, N))
            throughput *= 0.8 * obj_color

        elif mat_id == MAT_GLASS:
            front_face = rd.dot(N) < 0.0
            outward_n = N if front_face else -N

            eta_i = 1.0
            eta_t = IOR_GLASS
            if not front_face:
                eta_i = IOR_GLASS
                eta_t = 1.0

            eta = eta_i / eta_t
            reflect_dir = normalize(reflect(rd, outward_n))
            refr_dir, tir = refract(rd, outward_n, eta)

            cos_theta = ti.max(0.0, -rd.dot(outward_n))
            kr = fresnel_schlick(cos_theta, IOR_GLASS)

            # 玻璃更真实：保留反射与折射两种路径的“单路径近似混合”
            # 并加入轻微吸收
            dist_inside = 2.0 if front_face else 1.0
            glass_tint = ti.Vector([0.98, 0.99, 1.0])
            absorption_tint = beer_lambert(glass_tint, dist_inside)

            if tir:
                ro = p + outward_n * EPS
                rd = reflect_dir
                throughput *= 0.98
            else:
                # 按 Fresnel 倾向更“合理”地选择反射/折射
                if kr > 0.5:
                    ro = p + outward_n * EPS
                    rd = reflect_dir
                    throughput *= kr * 0.98
                else:
                    if front_face:
                        ro = p - outward_n * EPS
                    else:
                        ro = p + outward_n * EPS
                    rd = normalize(refr_dir)
                    throughput *= (1.0 - kr) * absorption_tint

    return final_color

# -----------------------------
# 渲染
# -----------------------------
@ti.kernel
def render():
    light_p = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    cam_pos = ti.Vector([0.0, 1.0, 5.0])

    fov = 45.0
    aspect = res_x / res_y
    scale = ti.tan(fov * 0.5 * math.pi / 180.0)

    forward = normalize(ti.Vector([0.0, -0.2, -1.0]))
    right = normalize(forward.cross(ti.Vector([0.0, 1.0, 0.0])))
    up = right.cross(forward)

    for i, j in pixels:
        accum = ti.Vector([0.0, 0.0, 0.0])

        for s in range(spp[None]):
            rx = ti.random(ti.f32)
            ry = ti.random(ti.f32)

            u = (2.0 * ((i + rx) / res_x) - 1.0) * aspect * scale
            # 关键修正：这里不再把 y 反过来，保持与窗口显示一致
            v = (2.0 * ((j + ry) / res_y) - 1.0) * scale

            rd = normalize(forward + u * right + v * up)
            accum += trace_ray(cam_pos, rd, light_p)

        pixels[i, j] = ti.math.clamp(accum / spp[None], 0.0, 1.0)

# -----------------------------
# 主程序
# -----------------------------
def main():
    window = ti.ui.Window("Whitted Ray Tracing + Glass + AA", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 5
    spp[None] = 16

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.28):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 8)
            spp[None] = gui.slider_int('Samples / Pixel', spp[None], 1, 64)

        window.show()

if __name__ == '__main__':
    main()