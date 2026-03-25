import taichi as ti 
import math

ti.init(arch=ti.cpu) #必备初始化

WIDTH, HEIGHT = 700, 700 #窗口分辨率
vertices = [
    (2.0, 0.0, -2.0),
    (0.0, 2.0, -2.0),
    (-2.0, 0.0, -2.0)
]

def get_model_matrix(angle):
    """模型变换矩阵：绕Z轴旋转"""
    rad = angle * math.pi/180.0
    c = math.cos(rad)
    s = math.sin(rad)
    #4*4旋转矩阵
    model = ti.Matrix([[c, -s, 0.0, 0.0],
                      [s, c, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dt=ti.f32) #dt=ti.f32指定元素为32位浮点数
    return model

def get_view_matrix(eye_pos):
    """视图变换矩阵：将相机平移到原点"""
    ex, ey, ez = eye_pos
    view = ti.Matrix([[1.0, 0.0, 0.0, -ex],
                      [0.0, 1.0, 0.0, -ey],
                      [0.0, 0.0, 1.0, -ez],
                      [0.0, 0.0, 0.0, 1.0]], dt=ti.f32) #平移矩阵
    return view

def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    """透视投影矩阵：视场角（Y轴方向，角度制）、屏幕长宽比、近截面距离和远截面距离"""
    fov_rad = eye_fov * math.pi/180.0 #视角转为弧度
    n = -zNear
    f = -zFar #相机看向-Z轴，为负值

    t = math.tan(fov_rad/2.0) * abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r #计算近截面上的各个边界
    
    #1)透视到正交的矩阵(persp -> ortho)
    persp = ti.Matrix([[n,   0.0,      0.0,     0.0],
                       [0.0, n,        0.0,     0.0],
                       [0.0, 0.0,  n + f,  -n * f],
                       [0.0, 0.0,      1.0,     0.0]], dt=ti.f32)

    #2)构造正交投影：先平移到中心，再缩放到[-1,1]
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

#GUI初始化
gui = ti.GUI("3D Transformation", res=(WIDTH, HEIGHT))

#相机与投影参数
eye_pos = (0.0, 0.0, 5.0) #相机放在 z=+5，看向-Z
fov = 45.0
aspect = WIDTH / HEIGHT
zNear = 0.1
zFar = 50.0

angle = 0.0  #初始角度（度）

#主循环
while gui.running:
    for e in gui.get_events():
        #退出
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
        #按D顺时针旋转，按A逆时针旋转
        if e.key == 'a' or e.key == 'A':
            angle += 5.0
        if e.key == 'd' or e.key == 'D':
            angle -= 5.0

    #计算 MVP
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(fov, aspect, zNear, zFar)
    MVP = proj @ view @ model  #列向量规范下的右乘顺序

    #将三角形顶点从世界坐标 -> 裁剪坐标 -> NDC -> 屏幕归一化 (0..1)
    ndc_coords = []
    for v in vertices:
        x, y, z = v
        v_h = ti.Vector([x, y, z, 1.0], dt=ti.f32)
        v_clip = MVP @ v_h #裁剪坐标 (x, y, z, w)
        #透视除法
        v_ndc = ti.Vector([v_clip[0] / v_clip[3],
                           v_clip[1] / v_clip[3],
                           v_clip[2] / v_clip[3]], dt=ti.f32)
        #NDC -> 归一化屏幕坐标 [0,1]
        u = (v_ndc[0] + 1.0) * 0.5
        vcoord = (v_ndc[1] + 1.0) * 0.5 #GUI 的 y 向上
        ndc_coords.append((u, vcoord))

    #清屏并画线框三角形（用归一化坐标）
    gui.clear(0x000000)
    #三条边，分别绘制不同颜色便于观察
    gui.line(begin=ndc_coords[0], end=ndc_coords[1], color=0xff0000, radius=3)
    gui.line(begin=ndc_coords[1], end=ndc_coords[2], color=0x009b48, radius=3)
    gui.line(begin=ndc_coords[2], end=ndc_coords[0], color=0x0082ff, radius=3)

    gui.text(f"angle = {angle:.1f} deg", pos=(0.02, 0.02), color=0xFFFFFF)
    gui.show()

#3D立方体如何进化？