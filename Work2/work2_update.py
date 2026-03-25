import taichi as ti
import math

ti.init(arch=ti.cpu)

WIDTH, HEIGHT = 700, 700

#立方体顶点（中心原点，边长2）
cube_vertices = [
    (-1.0, -1.0, -1.0),  # 0
    ( 1.0, -1.0, -1.0),  # 1
    ( 1.0,  1.0, -1.0),  # 2
    (-1.0,  1.0, -1.0),  # 3
    (-1.0, -1.0,  1.0),  # 4
    ( 1.0, -1.0,  1.0),  # 5
    ( 1.0,  1.0,  1.0),  # 6
    (-1.0,  1.0,  1.0),  # 7
]

#12条边（顶点索引对）
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  #z = -1 面
    (4, 5), (5, 6), (6, 7), (7, 4),  #z = +1 面
    (0, 4), (1, 5), (2, 6), (3, 7)   #垂直边
]

def rotation_y(angle_deg):
    rad = angle_deg * math.pi / 180.0
    c = math.cos(rad); s = math.sin(rad)
    return ti.Matrix([[  c, 0.0,   s, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [ -s, 0.0,   c, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dt=ti.f32)

def get_model_matrix(angle):
    #只绕Y轴旋转（竖直轴）
    return rotation_y(angle)

def get_view_matrix_lookat(eye_pos, center=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    #构造 look-at 视图矩阵，使相机从侧面看向原点
    eye_v = ti.Vector([eye_pos[0], eye_pos[1], eye_pos[2]], dt=ti.f32)
    center_v = ti.Vector([center[0], center[1], center[2]], dt=ti.f32)
    up_v = ti.Vector([up[0], up[1], up[2]], dt=ti.f32)

    z = (eye_v - center_v).normalized()    #相机朝向的反向（camera space Z）
    x = up_v.cross(z).normalized()         #camera right
    y = z.cross(x)                         #camera up

    #矩阵布局与列向量乘法兼容（与原来的平移矩阵格式一致）
    view = ti.Matrix([[x[0], x[1], x[2], -x.dot(eye_v)],
                      [y[0], y[1], y[2], -y.dot(eye_v)],
                      [z[0], z[1], z[2], -z.dot(eye_v)],
                      [0.0,  0.0,  0.0,        1.0]], dt=ti.f32)
    return view

def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    fov_rad = eye_fov * math.pi / 180.0
    n = -zNear
    f = -zFar

    t = math.tan(fov_rad / 2.0) * abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    persp = ti.Matrix([[n,   0.0,      0.0,     0.0],
                       [0.0, n,        0.0,     0.0],
                       [0.0, 0.0,  n + f,  -n * f],
                       [0.0, 0.0,      1.0,     0.0]], dt=ti.f32)

    ortho_trans = ti.Matrix([[1.0, 0.0, 0.0, -(r + l) / 2.0],
                             [0.0, 1.0, 0.0, -(t + b) / 2.0],
                             [0.0, 0.0, 1.0, -(n + f) / 2.0],
                             [0.0, 0.0, 0.0, 1.0]], dt=ti.f32)

    ortho_scale = ti.Matrix([[2.0 / (r - l), 0.0, 0.0, 0.0],
                             [0.0, 2.0 / (t - b), 0.0, 0.0],
                             [0.0, 0.0, 2.0 / (n - f), 0.0],
                             [0.0, 0.0, 0.0, 1.0]], dt=ti.f32)

    ortho = ortho_scale @ ortho_trans
    proj = ortho @ persp
    return proj

# GUI init
gui = ti.GUI("3D Cube (Side Camera & Y-rotation)", res=(WIDTH, HEIGHT))

#相机位置：往右上方侧面移动，看向原点
eye_pos = (3.0, 1.5, 5.0)   #侧面斜视效果更明显
fov = 60.0
aspect = WIDTH / HEIGHT
zNear = 0.1
zFar = 50.0

angle = 0.0
angle_step = 10.0

while gui.running:
    for e in gui.get_events():
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
        if e.type == ti.GUI.PRESS:
            if e.key in ('a', 'A'):
                angle += angle_step
            elif e.key in ('d', 'D'):
                angle -= angle_step

    model = get_model_matrix(angle)
    view = get_view_matrix_lookat(eye_pos, center=(0.0,0.0,0.0))
    proj = get_projection_matrix(fov, aspect, zNear, zFar)
    MVP = proj @ view @ model

    gui.clear(0x000000)
    for edge in cube_edges:
        i0, i1 = edge
        v0 = cube_vertices[i0]
        v1 = cube_vertices[i1]

        vh0 = ti.Vector([v0[0], v0[1], v0[2], 1.0], dt=ti.f32)
        vh1 = ti.Vector([v1[0], v1[1], v1[2], 1.0], dt=ti.f32)

        clip0 = MVP @ vh0
        clip1 = MVP @ vh1

        #透视除法
        ndc0 = ti.Vector([clip0[0] / clip0[3], clip0[1] / clip0[3], clip0[2] / clip0[3]], dt=ti.f32)
        ndc1 = ti.Vector([clip1[0] / clip1[3], clip1[1] / clip1[3], clip1[2] / clip1[3]], dt=ti.f32)

        u0 = (ndc0[0] + 1.0) * 0.5
        v0c = (ndc0[1] + 1.0) * 0.5
        u1 = (ndc1[0] + 1.0) * 0.5
        v1c = (ndc1[1] + 1.0) * 0.5

        gui.line(begin=(u0, v0c), end=(u1, v1c), color=0xFFFFFF, radius=2)

    gui.text(f"angle = {angle:.1f} deg  |  eye={eye_pos}", pos=(0.02, 0.02), color=0xFFFFFF)
    gui.show()