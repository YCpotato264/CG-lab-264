# 实验八：基于 SMPL 的 LBS 蒙皮过程可视化实验报告

姓名：杨芸菲
学号：202411081095
专业：计算机科学与技术

## 目录
1. [项目概述](#项目概述)
2. [实验目标](#实验目标)
3. [实验原理](#实验原理)
   - [3.1 理解 LBS](#31-理解-lbs)
   - [3.2 五个核心对象](#32-五个核心对象)
4. [代码逻辑](#代码逻辑)
   - [4.1 环境与依赖准备](#41-环境与依赖准备)
   - [4.2 旧版 SMPL 文件兼容处理](#42-旧版-smpl-文件兼容处理)
   - [4.3 可视化辅助函数](#43-可视化辅助函数)
   - [4.4 构造示例形状参数与姿态参数](#44-构造示例形状参数与姿态参数)
   - [4.5 手写 LBS 计算流程](#45-手写-lbs-计算流程)
   - [4.6 与官方 forward 结果对比](#46-与官方-forward-结果对比)
   - [4.7 图像与结果保存](#47-图像与结果保存)
5. [图像展示](#图像展示)
6. [实现功能](#实现功能)
7. [实验结果说明](#实验结果说明)
8. [总结](#总结)

---

## 项目概述

本实验基于 **SMPL 参数化人体模型**，对 **LBS（Linear Blend Skinning，线性混合蒙皮）** 的完整过程进行拆解、计算与可视化展示。

与“只调用模型得到最终顶点”的方式不同，本实验将 SMPL 内部的关键中间量单独提取出来，包括：

- 模板网格 `v_template`
- 形状变形后网格 `v_shaped`
- 由形状网格回归得到的关节 `J`
- 姿态校正后网格 `v_posed`
- 经 LBS 后的最终顶点 `verts`

并进一步将其保存为多张对比图，用于观察 SMPL 从“静态模板”到“最终姿态人体”的完整生成过程。

该实验的核心价值在于：

1. 帮助理解 SMPL 模型中各参数的物理意义；
2. 理解 `lbs()` 函数内部的计算路径；
3. 将抽象的数学过程转化为可视化结果；
4. 验证手写 LBS 与官方实现的一致性。

---

## 实验目标

本实验旨在完成以下三个方面的学习目标：

### 1. 理解参数化人体模型中各核心对象之间的关系
包括：

- 模板网格
- 形状参数 `β`
- 姿态参数 `θ`
- 关节回归器 `J_regressor`
- 蒙皮权重 `lbs_weights`

### 2. 理解 LBS 的四个阶段

#### （a）模板网格与蒙皮权重
$$
\bar{T}, \mathcal{W}
$$

#### （b）加入形状参数后的网格与关节
$$
T_{shape} = \bar{T} + B_S(\beta)
$$
$$
J(\beta) = \mathcal{J}(T_{shape})
$$

#### （c）加入姿态相关校正后的网格
$$
T_P(\beta,\theta) = \bar{T} + B_S(\beta) + B_P(\theta)
$$

#### （d）经过 LBS 后的最终姿态结果
$$
v_i' = \sum_{k=1}^{K} w_{ik} \, G_k(\theta, J(\beta))
\begin{bmatrix}
v_i^{posed} \\
1
\end{bmatrix}
$$

### 3. 学会调用 SMPL 模型并提取中间量可视化
通过代码直接展示官方 `lbs()` 中的关键阶段，帮助理解每一步到底做了什么。

---

## 实验原理

### 3.1 理解 LBS

LBS 的核心思想是：  
**一个顶点不会只跟随一个骨骼，而是同时受到多个关节的影响，并按照权重进行加权变换。**

这是一种非常经典的人体动画方法，广泛应用于角色建模和动画系统中。

---

### （a）模板网格与蒙皮权重

初始状态是模板人体网格 $\bar{T}$，通常处于 T-pose。

同时，每个顶点都带有一组对各关节的影响权重 $\mathcal{W}$。如果某个顶点更靠近手臂，那么它通常会更受肩、肘、腕等关节影响。

这一步的重点不是“动起来”，而是理解：

- 网格还没根据人物体型改变；
- 网格也还没根据姿态弯曲；
- 但每个顶点已经知道“将来应该主要跟着哪些骨骼走”。

在 `lbs()` 实现中，最终每个顶点的 $4 \times 4$ 变换矩阵，就是由这些 `lbs_weights` 对各关节变换矩阵加权得到的。

---

### （b）加入形状参数：$B_S(\beta)$

形状参数 $\beta$ 控制“这个人长什么样”，例如高矮、胖瘦、肩宽、腿长等。

形状校正后，得到：

$$
T_{shape} = \bar{T} + B_S(\beta)
$$

然后再根据这个已经改变了体型的网格，利用关节回归器得到关节位置：

$$
J(\beta) = \mathcal{J}(T_{shape})
$$

实现思路为：

$$
v_{shaped} = v_{template} + \text{blend\_shapes}(\beta, shapedirs)
$$

以及：

$$
J = \text{vertices2joints}(J_{regressor}, v_{shaped})
$$

也就是说，关节位置不是固定常数，而是由形状后的网格回归出来的。

---

### （c）加入姿态相关校正：$B_P(\theta)$

蒙皮并非把骨骼旋转一下，皮肤跟着转这么简单。  
人体在弯曲时，肩膀、肘部、膝盖附近会出现额外几何变化，仅靠骨骼刚体旋转无法表达。

因此，SMPL 在进入真正的 LBS 前，还会加入一项 pose blend shape：

$$
T_P(\beta,\theta) = \bar{T} + B_S(\beta) + B_P(\theta)
$$

实现思路是先把姿态参数转成旋转矩阵，再构造：

$$
pose\_feature = R(\theta) - I
$$

随后通过 `posedirs` 线性映射得到 `pose_offsets`，并加到 `v_shaped` 上，形成 `v_posed`：

- `rot_mats = batch_rodrigues(...)`
- `pose_feature = (rot_mats[:, 1:, :, :] - ident).view(...)`
- `pose_offsets = torch.matmul(pose_feature, posedirs).view(...)`
- `v_posed = v_shaped + pose_offsets`

---

### （d）线性混合蒙皮：$W(\cdot)$

经过上述步骤后：

- 已经考虑形状的关节位置 $J(\beta)$
- 已经考虑姿态校正的顶点 $T_P(\beta,\theta)$
- 每个顶点对各关节的权重 $\mathcal{W}$

于是进入真正的 LBS：

$$
v_i' = \sum_{k=1}^{K} w_{ik} \, G_k(\theta, J(\beta))
\begin{bmatrix}
v_i^{posed} \\
1
\end{bmatrix}
$$

其中：

- $v_i^{posed}$ 是第 $i$ 个经过 shape + pose 矫正的顶点；
- $w_{ik}$ 是顶点 $i$ 受第 $k$ 个关节影响的权重；
- $G_k$ 是第 $k$ 个关节在运动学链上的全局刚体变换。

对应到代码中：

- `J_transformed, A = batch_rigid_transform(...)`
- `W = lbs_weights.unsqueeze(...).expand(...)`
- `T = torch.matmul(W, A.view(...)).view(..., 4, 4)`
- `v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))`
- `verts = v_homo[:, :, :3, 0]`

也就是说，每个顶点最终不是只跟着一个关节走，而是跟着多个关节做加权平均后的变换。这也是 “Linear Blend Skinning” 名字的来源。

---

### 3.2 五个核心对象

本实验中明确区分了以下五个核心变量：

1. **`v_template`**：模板顶点  
2. **`v_shaped`**：加了形状形变后的顶点  
3. **`J`**：由 `v_shaped` 回归出的关节  
4. **`v_posed`**：加了姿态校正后的顶点  
5. **`verts`**：完成 LBS 之后的最终顶点  

这五个对象构成了 SMPL 内部从静态形状到动态人体的完整生成链条。

---

## 代码逻辑

---

### 4.1 环境与依赖准备

代码中导入了以下库：

- `numpy`、`torch`：数值计算与张量运算
- `matplotlib`：结果可视化
- `smplx`：SMPL 模型加载与前向计算
- `smplx.lbs`：LBS 相关底层函数

并设置了：

```python
matplotlib.use("Agg")
```

这表示使用无界面绘图后端，适合在服务器或命令行环境中直接保存图片。

---

### 4.2 旧版 SMPL 文件兼容处理

很多旧版 SMPL `.pkl` 文件内部依赖 `chumpy` 保存的对象。  
代码中定义了：

- `_ChumpyArrayShim`
- `install_chumpy_pickle_shim()`

其作用是：

- 在不安装 `chumpy` 的情况下，仍能读取老版本 SMPL 模型文件；
- 通过 mock 的方式兼容 pickle 反序列化过程。

这一步提高了代码在不同环境下的可运行性。

---

### 4.3 可视化辅助函数

代码定义了一系列辅助函数，用于完成三维网格展示：

- `set_axes_equal()`：保证三维坐标轴比例一致
- `smpl_to_plot_coords()`：将 SMPL 坐标转换为绘图坐标
- `draw_mesh()`：绘制带颜色的三维人体网格
- `save_single_figure()`：保存单张结果图
- `save_comparison_grid()`：保存四阶段对比图
- `save_all_joint_weights_figure()`：保存所有关节权重图

这些函数的意义在于：  
**把抽象的 LBS 过程拆成可观察的阶段图像。**

---

### 4.4 构造示例形状参数与姿态参数

#### 形状参数 `betas`

```python
betas = build_demo_shape(device, dtype, num_betas=args.num_betas)
```

代码中人为设置几个非零的 shape 参数：

- `betas[0, 0] = 2.0`
- `betas[0, 1] = -1.2`
- `betas[0, 2] = 0.8`

这样可以让人体体型变化更明显，便于观察形状空间对网格的影响。

#### 姿态参数 `global_orient` 和 `body_pose`

```python
global_orient, body_pose = build_demo_pose(device, dtype)
```

代码人为对多个关节设置轴角旋转，例如：

- 左右肩膀
- 左右肘部
- 左右髋部
- 左右膝部

这样可以让人体呈现较明显的弯曲状态，从而验证 pose blend shapes 与最终 LBS 的效果。

---

### 4.5 手写 LBS 计算流程

这是本实验最关键的部分。  
函数 `compute_manual_lbs()` 手动复现了 SMPL 中的核心计算过程。

#### 第一步：模板网格

```python
v_template = model.v_template
```

若模板不是 batch 形式，则扩展为 `[1, V, 3]`。

---

#### 第二步：形状形变

```python
shapedirs = model.shapedirs[:, :, :betas.shape[1]]
v_shaped = v_template + blend_shapes(betas, shapedirs)
```

这一步对应：

$$
v_{shaped} = v_{template} + B_S(\beta)
$$

---

#### 第三步：关节回归

```python
J = vertices2joints(model.J_regressor, v_shaped)
```

这一步对应：

$$
J(\beta) = \mathcal{J}(T_{shape})
$$

说明关节位置来自形状后的网格，而不是固定写死的。

---

#### 第四步：姿态表示转旋转矩阵

```python
full_pose = torch.cat([global_orient, body_pose], dim=1)
rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)
```

这里将轴角表示转换为旋转矩阵，便于后续姿态特征构造。

---

#### 第五步：姿态校正项

```python
pose_feature = (rot_mats[:, 1:, :, :] - ident).view(1, -1)
pose_offsets = torch.matmul(pose_feature, posedirs).view(1, -1, 3)
v_posed = v_shaped + pose_offsets
```

这一步对应：

$$
T_P(\beta,\theta) = \bar{T} + B_S(\beta) + B_P(\theta)
$$

即对肩、肘、膝等弯曲关节附近的非刚性形变进行补偿。

---

#### 第六步：刚体变换与 LBS

```python
J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents, dtype=dtype)
W = model.lbs_weights.unsqueeze(0).expand(1, -1, -1)
T = torch.matmul(W, A.view(1, num_joints, 16)).view(1, -1, 4, 4)
```

这里先计算每个关节的全局变换矩阵，再利用 `lbs_weights` 得到每个顶点的混合变换矩阵。

最终：

```python
v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
verts = v_homo[:, :, :3, 0]
```

得到完成蒙皮后的最终顶点。

---

### 4.6 与官方 forward 结果对比

为了验证手写 LBS 的正确性，代码通过：

```python
compare_with_official_forward(...)
```

将手写计算得到的 `manual_verts` 与 SMPL 官方 `forward()` 输出进行对比，统计：

- 平均绝对误差 `mean_err`
- 最大绝对误差 `max_err`

如果误差非常小，说明手写过程与官方实现一致性较高。

---

### 4.7 图像与结果保存

代码将结果保存到输出目录中，包括：

- `stage_a_template_weights.png`
- `stage_b_shaped_joints.png`
- `stage_c_pose_offsets.png`
- `stage_d_lbs_result.png`
- `comparison_grid.png`
- `all_joint_weights.png`
- `summary.txt`

这使得实验结果既可以图像展示，也可以作为文本摘要保存，便于写实验报告与复查。

---

## 图像展示
![stage_a_template_weights](lbs_lab\outputs\stage_a_template_weights.png)
![stage_b_shaped_joints](lbs_lab\outputs\stage_b_shaped_joints.png)
![stage_c_pose_offsets](lbs_lab\outputs\stage_c_pose_offsets.png)
![stage_d_lbs_result](lbs_lab\outputs\stage_d_lbs_result.png)
![comparison_grid](lbs_lab\outputs\comparison_grid.png)
![all_joint_weights](lbs_lab\outputs\all_joint_weights.png)

---

## 实现功能

本实验实现了以下功能：

### 1. SMPL 模型加载
- 成功加载 `SMPL_NEUTRAL.pkl`
- 兼容旧版 `chumpy` 序列化对象

### 2. 手写 LBS 流程复现
- 复现了 SMPL 的关键中间步骤
- 显式计算 `v_template`、`v_shaped`、`J`、`v_posed`、`verts`

### 3. 四阶段可视化
- 模板网格与权重
- 形状校正后的网格与关节
- 姿态校正后的网格
- 最终 LBS 结果

### 4. 关节权重可视化
- 按某个关节的权重给模型上色
- 也可显示所有关节的主导权重分布

### 5. 与官方 forward 对比验证
- 计算手写结果与官方输出误差
- 验证实现正确性

### 6. 自动保存结果
- 保存多张图片
- 保存误差摘要到 `summary.txt`

---

## 实验结果说明

### 1. 模板网格与权重图
模板网格通常呈现标准 T-pose，整体结构未发生形变。  
通过颜色映射可以看出：

- 靠近目标关节的顶点权重更高；
- 不同关节主导的区域颜色不同；
- 蒙皮权重在身体不同部位分布不均匀，符合人体结构逻辑。

---

### 2. 形状变形后的网格与关节
加入 `betas` 后，人体体型会发生变化，例如：

- 躯干宽度变化
- 四肢粗细变化
- 头部和身体比例变化

同时，由 `v_shaped` 回归出来的关节位置也会随体型变化而发生偏移，说明 SMPL 的关节并非固定，而是与形状参数相关。

---

### 3. 姿态校正后的网格
在加入姿态 blend shapes 后，关节弯曲区域会出现额外几何变化，例如：

- 肘部弯曲处更自然
- 膝盖弯曲处更贴合人体结构
- 肩部、髋部局部形变更合理

如果没有这一步，仅靠骨骼旋转会显得非常“机械”，不符合真实皮肤形变。

---

### 4. 最终 LBS 结果
经过 LBS 后，人体网格形成最终姿态。  
可以观察到：

- 身体整体姿势与设定的关节旋转一致；
- 顶点受到多个关节权重共同影响；
- 结果保持了较强的拓扑连续性；
- 网格没有出现明显断裂或塌陷。

---

### 5. 数值一致性验证
实验最后输出了手写 LBS 与官方 forward 的误差：

- `manual_vs_official_mean_abs_error`
- `manual_vs_official_max_abs_error`

如果这两个值较小，说明：

1. 手写计算过程正确；
2. 对 SMPL 内部机制的理解是准确的；
3. 可视化与模型推导结果吻合。

---

## 总结

本实验围绕 SMPL 模型的 LBS 蒙皮过程展开，通过手动拆解官方 `lbs()` 的关键步骤，实现了从模板网格、形状变形、姿态校正到最终蒙皮结果的完整可视化。

实验不仅帮助理解了：

- `v_template`
- `v_shaped`
- `J`
- `v_posed`
- `verts`

这五个核心对象之间的关系，也明确了 SMPL 中：

- 形状参数如何改变人体体型；
- 姿态参数如何影响局部几何；
- 关节回归器如何从网格中推导骨架；
- 蒙皮权重如何把关节运动传播到每个顶点。

通过将手写 LBS 与官方输出对比，还验证了实现的正确性。  
总体而言，本实验加深了对参数化人体模型、线性混合蒙皮和人体动画生成机制的理解。

---

## 附录：实验中关键公式

### 1. 形状变形
$$
v_{shaped} = v_{template} + B_S(\beta)
$$

### 2. 关节回归
$$
J(\beta) = \mathcal{J}(v_{shaped})
$$

### 3. 姿态校正
$$
v_{posed} = v_{shaped} + B_P(\theta)
$$

### 4. LBS 结果
$$
v_i' = \sum_{k=1}^{K} w_{ik} \, G_k(\theta, J(\beta))
\begin{bmatrix}
v_i^{posed} \\
1
\end{bmatrix}
$$

---

## 附录：五个核心对象对照表

| 名称 | 代码变量 | 含义 |
|------|----------|------|
| 模板顶点 | `v_template` | T-pose 下的标准人体网格 |
| 形状后顶点 | `v_shaped` | 加入形状参数后的顶点 |
| 关节 | `J` | 由形状后网格回归得到的关节位置 |
| 姿态后顶点 | `v_posed` | 加入姿态补偿后的顶点 |
| 最终顶点 | `verts` | LBS 后的最终人体姿态网格 |
