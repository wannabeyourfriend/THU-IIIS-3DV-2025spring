# 3D Vision Computing - Part I Notes

> **Note Taker:** Alex
>
> **Contact:** `wang-zx23@mails.tsinghua.edu.cn`
>
> **Instructor:** Li Yi
>
> **Reference:** Li Yi's 3DV lecture & Hao Su's ML-meets-geometry lecture

---

## Table of Contents

- [Introduction](#0-introduction)
- [Geometry: Curves & Surfaces](#1-geometry-curves--surfaces)
- [Representations: Mesh & Point Cloud](#2-representations-mesh--point-cloud)
- [Transformation](#3-transformation)

---

## 0 Introduction

Geometry understanding is crucial in Robotics, Augmented Reality, Autonomous Driving, and Medical Image Processing. Through geometry understanding, robots can obtain **a priori knowledge of the 3D world**.

### Key Topics

- **Geometry theories** → Curves, Surface, Rotation, etc.
- **Sensing: Computer Representation of Geometries** → Mesh, Point, etc.
- **Sensing: 3D reconstruction** → Reconstruction from a single image
- **Geometry Processing** → Local geometric property estimation, Surface reconstruction
- **Recognition** → Object classification, Object detection, 6D pose estimation, Segmentation, Human pose estimation
- **Relationship Analysis** → Shape correspondences

---

## 1 Geometry: Curves & Surfaces

> This chapter mainly focuses on the basic concepts, definitions, and fundamentals of geometry.

### 1.1 Curves

#### 1.1.1 Parameterization

**Definition**

A parameterized curve is a map from a 1-dimensional region to $\mathbb{R}^n$.

- **2D curve:** $\gamma(t) = (x(t), y(t))$
  - **Intuition:** A particle moving in space with position $\gamma(t)$ at time $t$
- **3D curve:** $\gamma(t) = (x(t), y(t), z(t))| \mathbb{R} \to \mathbb{R}^3: t \to p(t)$
- **Example:** $p(t) = r(\cos(t), \sin(t)), \quad t \in [0,2\pi)$

**Applications**

Bezier Curves, Splines:

$$
s(t) = \sum_{i = 0}^n \mathbf{p}_iB_i^n(t)
$$

A curve is like a one-dimensional "manifold" - a set of points that locally looks like a line. However, when a cusp occurs, things become extremely complex.

**Key Concepts:**

- **Tangent Vector:** $\gamma'(t) = (x'(t), y'(t)) \in \mathbb{R}^2$
  - $\gamma'(t)$ indicates the direction of movement
  - $\|\gamma'(t)\|$ indicates the speed of movement

- **Arc Length:** $\int_a^b ||\gamma'(t)|| dt$

- **Parameterization by Arc Length:**
  - $s(t) = \int_{t_0}^t ||\gamma'(t)||dt$
  - $t(s)$ = inverse function of $s(t)$
  - $\hat{\gamma}(s) = \gamma(t(s))$

#### 1.1.2 2D Curves

**Theorem:** Define Tangent vector $T(s) = \gamma'(s)$, $\implies \|T(s)\| \equiv 1$

**Proof:** By definition of arc length parameterization.

**Normal Vector:** Define Normal vector $N(s)$ where $J$ is the rotation matrix of $90^{\circ}$ in 2D space:

$$
J =
\begin{bmatrix}
0 & -1\\
1 & 0
\end{bmatrix}
$$

We have $N(s) := JT(s)$.

**Frenet Equation:**

$$
\frac{d}{ds} \begin{bmatrix} T(s) \\ N(s) \end{bmatrix} := \begin{bmatrix} 0 & k(s) \\ -k(s) & 0 \end{bmatrix} \begin{bmatrix} T(s) \\ N(s) \end{bmatrix}
$$

**$\mathbb{R}^2$ Curve Theorem**

Radius of Curvature is defined as $\kappa(s) = \frac{1}{R}$, where $R$ is the radius of curvature. The geometric meaning indicates how much the normal changes in the direction tangent to the curve. Curvature $\kappa(s)$ **characterizes a planar curve up to rigid motion**, which is always positive.

#### 1.1.3 3D Curves

**Osculating Plane**

The plane determined by $T(s)$ and $N(s)$. We define the Binormal Vector $B(s) = T(s) \times N(s)$.

**Curvature $\kappa$ & Torsion $\tau$**

**Definition:**
$<N'(s), T(s)> = -\kappa(s) \quad <N'(s), B(s)> = \tau(s)$

**Theorem:**
- $T'(s) = \kappa(s)N(s)$
- $N'(s) = -\kappa(s)T(s) + \tau(s)B(s)$
- $B'(s) = -\tau(s)N(s)$

**Geometric Meaning:**

- Curvature indicates how much the **normal** changes in the direction **tangent** to the curve (in-plane motion)
- Torsion indicates how much the normal changes in the direction **orthogonal** to the osculating plane of the curve (out-of-plane motion)
- Curvature is always **positive** but torsion can be **negative**

**Frenet Frame:**

$$
\frac{d}{ds} \begin{pmatrix} T \\ N \\ B \end{pmatrix} = \begin{pmatrix} 0 & \kappa & 0 \\ -\kappa & 0 & \tau \\ 0 & -\tau & 0 \end{pmatrix} \begin{pmatrix} T \\ N \\ B \end{pmatrix}
$$

**$\mathbb{R}^3$ Curve Theorem**

Curvature $\kappa(s)$ and torsion $\tau(s)$ characterize a 3D curve up to rigid motion.

#### 1.1.4 Geometric Meaning

A curve is defined as a **map** from an **interval** to $\mathbb{R}^n$. The **tangent vector** to the curve describes the **direction of motion along the curve**. Both **curvature** and **torsion** are measures that **describe the change in the normal direction of the curve**.

### 1.2 Surfaces

#### 1.2.1 Surface Parametrization

**$f: U \to \mathbb{R}^3$**

- A parameterized surface is a map from a two-dimensional region $U \subset \mathbb{R}^2$ to $\mathbb{R}^n$
- The set of points $f(U)$ is called the image of the parameterization

**Saddle Example:**

$$
U := \{(u, v) \in \mathbb{R}^2 : u^2 + v^2 \leq 1\}\\
f(u, v) = [u, v, u^2 - v^2]^T
$$

#### 1.2.2 Differentiable Manifold

**Properties:**

- **Local Properties:** properties that can be discovered by local observation (points + neighborhoods)
- **Smoothness:** a continuous one-to-one mapping from local to global
- **Tangent Plane:** each point can have a tangent plane attached, containing all possible directions passing tangentially from that point, defined as $T_p(\mathbb{R}^3)$

**Differential of a Surface**

$Df_p: T_p(\mathbb{R}^2) \to T_{f(p)}(\mathbb{R}^3)$

Relates the movement of point in the domain and on the surface.

$$
df = \frac{\partial f}{\partial u} du + \frac{\partial f}{\partial v} dv
$$

$Df_p := \left[ \frac{\partial f}{\partial u}, \frac{\partial f}{\partial v} \right] \in \mathbb{R}^{3 \times 2}$

#### 1.2.3 Curvature

**Normal Vector $N_p$:**

$$
N(u, v) = \frac{f_u \times f_v}{\| f_u \times f_v \|}\\
\text{where } f_u = \frac{\partial f}{\partial u}, \quad f_v = \frac{\partial f}{\partial v}
$$

**Shape Operator $DN_p$:**

$$
DN_p := \begin{bmatrix} \frac{\partial N}{\partial u}, \frac{\partial N}{\partial v} \end{bmatrix} \in \mathbb{R}^{3 \times 2}
$$

**Curvature $\mathbf{\kappa}$:**

Vector $\mathbf{\kappa} = DN_p[\mu X] = \frac{DN_pX}{\|Df_pX\|}$

**Principal Curvatures:**

$$
\kappa_n := <\mathbf{T}, \kappa> = \frac{<Df_pX, DN_pX>}{\|Df_pX|\|^2}>
$$

**Principal Directions:**

- $\kappa_1 = \kappa_{\text{max}} = \max_{\phi} \kappa_n(\phi)$
- $\kappa_2 = \kappa_{\text{min}} = \min_{\phi} \kappa_n(\phi)$

**Theorem:** The principal directions are always orthogonal.

**Shape Operator:**

$$
\exists S \in \mathbb{R}^{2 \times 2} \quad \text{such that} \quad DN_p = Df_p S
$$

The principal directions are the eigenvectors of the shape operator $S$, and the principal curvatures are the eigenvalues of $S$.

#### 1.2.4 First Fundamental Form

**Definition:**

The first fundamental form $I_p$ is defined as the inner product in the tangent space $T_p(\mathbb{R}^3)$:

$$
I_p(X, Y) = \langle Df_p X, Df_p Y \rangle = X^T (Df_p^T Df_p)Y
$$

**Applications:**

- Determine curve length within the surface without referring to $f$
- Determine angles within the surface without referring to $f$
- Shape classification by isometry
- Geodesic distances
- Distance distribution descriptor

**Second Fundamental Form:**

$$
II(X, Y) = \langle DN_p X, Df_p Y \rangle
$$

**Theorem:** A smooth surface is determined up to rigid motion by its first and second fundamental forms.

#### 1.2.6 Gaussian and Mean Curvature

**Definition:**

- **Gaussian Curvature:** $K := \kappa_1 \kappa_2$
- **Mean Curvature:** $H := \frac{1}{2} (\kappa_1 + \kappa_2)$

**Gauss's Theorema Egregium:**

The Gaussian curvature of an embedded smooth surface in $\mathbb{R}^3$ is invariant under local isometries.

---

## 2 Representations: Mesh & Point Cloud

> This chapter mainly focuses on 3D representations, including mesh, point cloud, and implicit representation methods.

### 2.1 Meshes

#### 2.1.1 Formulation

Mesh formulation can be seen as **manifold condition** plus a set of:

- $V=\{v_1,v_2,...,v_n\} \subset \mathbb{R}^3$ (Vertices)
- $E=\{e_1,e_2,...,e_k\} \subseteq V \times V$ (Edges)
- $F=\{f_1,f_2,...,f_m\} \subseteq V \times V \times V$ (Faces)

**Manifold condition of discrete mesh:**

1. Each edge is incident to one or two faces
2. Faces incident to a vertex form a closed or open fan

Polygonal meshes are piece-wise linear approximations of smooth surfaces. Good triangulation is important (manifold, equi-length).

#### 2.1.2 Storage

**Triangle List:**

- **STL format:** Used in CAD
- Each face is stored with 3 positions
- No connectivity information

**Indexed Face Set:**

- **Formats:** OBJ, OFF, WRL
- **Storage:**
  - Vertex: Position
  - Face: Vertex indices
  - Convention: Save vertices in counterclockwise order for normal computation

#### 2.1.3 Normals

Normal can be computed using various methods, including the right-hand rule and cross products. By indicating normal continuity, surfaces can be divided into:
- **Orientable:** Surfaces with consistent normal direction
- **Non-orientable:** Surfaces like the Möbius strip

#### 2.1.4 Curvatures

**Rusinkiewicz's Method:**

An effective approach for face curvature estimation:
- Assume a local frame at a small triangle
- Assume that normals are roughly parallel
- Solve for the shape operator $S$ using least squares

### 2.2 Point Cloud

#### 2.2.1 Representation

A point cloud is **a set of points** in 3D space, representing the surface of an object.

- **From the real world:**
  - 3D scanning techniques (LIDAR, Kinect, Stereo)
  - Challenges: Resolution, occlusion, noise, registration
- **From existing virtual shapes:**
  - Lightweight shape representation
  - Compact storage and easy to build algorithms

#### 2.2.2 Application-based Sampling

**For storage or analysis purposes:**
- Preserve surface information

**For learning data generation:**
- Minimize virtual-real domain gap

**Uniform Sampling:**

- Independent identically distributed (i.i.d.) samples by surface area
- Usually the easiest to implement
- Issue: Irregularly spaced sampling

**Farthest Point Sampling:**

- Goal: Sampled points are far away from each other
- NP-hard problem
- Greedy approximation method

**Iterative Furthest Point Sampling:**

1. Over-sample the shape by any fast method
2. Iteratively select $K$ points

**Implementation Issues:**

- Naive implementation complexity: $\mathcal{O}(KN)$
- Optimization techniques:
  - CPU: Vectorization (numpy, scipy.spatial.distance.cdist)
  - GPU: Shared memory, complexity reduced to $\mathcal{O}(K(N/M + \log M))$

#### 2.2.3 Voxel Downsampling

- Uses a regular voxel grid to downsample
- Allows higher parallelization
- Generates regularly spaced sampling
- Complexity: $\mathcal{O}(N)$

#### 2.2.4 Estimating Normals

**Plane-fitting:** Find the plane that best fits the neighborhood of a point of interest.

**Least-square Formulation:**

Assume the plane equation is $w^T(x - c) = 0$ with $\|w\| = 1$.

$$
\min_{w,c} \sum_i \|w^T(x_i - c)\|^2_2 \quad \text{subject to} \quad \|w\|^2 = 1
$$

**Solution:**
- Let $M = \sum_i (x_i - \bar{x})(x_i - \bar{x})^T$ and $\bar{x} = \frac{1}{n} \sum_i x_i$
- $w$ is the smallest eigenvector of $M$
- $c = w^T \bar{x}$

### 2.3 Implicit Representations

In explicit representations of geometry, all points are given directly, generally represented as $f: \mathbb{R}^2 \to \mathbb{R}^3; (u, v) \to (x, y, z)$. However, for tasks that distinguish between inside and outside of the surface, we can use implicit representations of geometry.

**Applications:**

- **Constructive Solid Geometry:** Combine implicit geometry via Boolean operations
- **Distance Functions:** Giving minimum distance (could be signed distance) from anywhere to object
- **Surface Blending:** Gradually blend surfaces together using distance functions

**Key Insight:** There are no "best" geometric representations!

---

## 3 Transformation

> This chapter focuses on the transformation and rotation of 3D objects.

### 3.1 Homogeneous Transformation

#### 3.1.1 Rigid Transformation

**Degrees of Freedom (DoF):** Degree of freedom, representing the number of independent parameters required to describe a transformation.

A rigid transformation can be described using a pair $(R_{s \rightarrow b}, \mathbf{t}_{s \rightarrow b})$, where:
- $R_{s \rightarrow b}$ is the rotation matrix
- $\mathbf{t}_{s \rightarrow b}$ is the translation vector

**Coordinate Transformation:**

$$
p^s = R_{s \rightarrow b}^s p^b + \mathbf{t}_{s \rightarrow b}^s
$$

The transformation is non-linear due to the translation component.

**Homogeneous Coordinates:**

To represent translations as linear transformations, we use homogeneous coordinates:

$$
\hat{x} = [x, 1]^T \in \mathbb{R}^4
$$

**Homogeneous Transformation Matrix:**

$$
T_{s \rightarrow b}^s = \begin{bmatrix}
R_{s \rightarrow b}^s & \mathbf{t}_{s \rightarrow b}^s \\
0 & 1
\end{bmatrix}
$$

**Linear Form of Coordinate Transformation:**

$$
\hat{x}^s = T_{s \rightarrow b}^s \hat{x}^b
$$

#### 3.1.2 Transformations

**Scaling:**

$$
S_s = \begin{bmatrix}
s_x & 0 & 0 & 0 \\
0 & s_y & 0 & 0 \\
0 & 0 & s_z & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

**Translation:**

$$
T_{\mathbf{t}} = \begin{bmatrix}
1 & 0 & 0 & t_x \\
0 & 1 & 0 & t_y \\
0 & 0 & 1 & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

**Rotation:**

$$
R_z(\theta) = \begin{bmatrix}
\cos \theta & -\sin \theta & 0 & 0 \\
\sin \theta & \cos \theta & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

### 3.2 Rotation

#### 3.2.1 Mathematics about Rotations

The set of rotations in $n$-dimensional space is defined by the Special Orthogonal Group $SO(n)$:

$$
SO(n) = \{ R \in \mathbb{R}^{n \times n} : \det(R) = 1, RR^T = I \}
$$

**Properties:**
- **Group:** Forms a group under matrix multiplication
- **Orthogonal:** Matrices satisfy $RR^T = I$
- **Special:** The determinant of each matrix is 1

**Special Cases:**
- $SO(2)$: 2D rotations, with 1 degree of freedom
- $SO(3)$: 3D rotations, with 3 degrees of freedom

**Topology:**
- $SO(2)$ has the same topology as a circle
- $SO(3)$ has a different topology from $(-1,1)^n$, which affects how rotations can be parameterized

#### 3.2.2 Parameterizing Rotation in Neural Networks

When using rotations in neural networks, ideal parameterizations should:

1. Map from $(-l, l)^n$ (as network output) to $SO(2)$
2. Be a differentiable bijection

**Challenges:**
- Input data points are close, but their corresponding $\theta$ predictions are far apart after convergence
- Special network designs are needed to handle these issues effectively

#### 3.2.3 Three Kinds of Representations

**Euler Angles:**

Represent 3D rotations using three angles about the principal axes $(x, y, z)$:

$$
R = R_z(\gamma) R_y(\beta) R_x(\alpha)
$$

**Advantages:** Intuitive
**Disadvantages:**
- Gimbal lock: Loss of a degree of freedom under certain conditions
- Non-uniqueness in representation

**Axis-Angle Representation:**

**Euler's Theorem:** Any rotation in $SO(3)$ can be represented as a rotation about a fixed axis $\hat{\omega} \in \mathbb{R}^3$ through a positive angle $\theta$

$$
\text{Rot}(\hat{\omega}, \theta) = e^{[\hat{\omega}]\theta} = I + [\hat{\omega}]\sin\theta + [\hat{\omega}]^2(1 - \cos\theta)
$$

where $[\mathbf{\omega}]$ is the skew-symmetric matrix:

$$
[\mathbf{\omega}] = \begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix}
$$

**Rotation Matrix to Axis-Angle:**

$$
\theta = \arccos\frac{1}{2}[\text{tr}(R) - 1]
$$

$$
[\hat{\omega}] = \frac{1}{2\sin\theta}(R - R^T) \quad \text{when } \text{tr}(R) \neq -1
$$

**Quaternion Representation:**

A quaternion $q$ is defined as:

$$
q = w + xi + yj + zk
$$

where $w$ is the real part and $(x, y, z)$ form the imaginary part. The imaginary units satisfy:
- $i^2 = j^2 = k^2 = ijk = -1$
- $ij = k = -ji$, $jk = i = -kj$, $ki = j = -ik$

**Quaternion to Rotation Matrix:**

$$
R(q) = \begin{bmatrix}
1 - 2y^2 - 2z^2 & 2xy - 2wz & 2xz + 2wy \\
2xy + 2wz & 1 - 2x^2 - 2z^2 & 2yz - 2wx \\
2xz - 2wy & 2yz + 2wx & 1 - 2x^2 - 2y^2
\end{bmatrix}
$$

**Axis-Angle to Quaternion:**

$$
q = [\cos(\theta/2), \sin(\theta/2) \hat{\omega}]
$$

**Comparison:**
- **Euler Angles:** Intuitive but suffer from gimbal lock
- **Axis-Angle:** Useful for geometric interpretation
- **Quaternions:** Compact, efficient, widely used in computer graphics and robotics

---

## Contributing

This repository contains lecture notes and materials from the 3D Vision Computing course. For questions or suggestions, please contact the note taker.

## License

Please refer to the course instructor for usage guidelines regarding these materials.

## Acknowledgments

- **Instructor:** Li Yi
- **Reference Materials:** Li Yi's 3DV lecture & Hao Su's ML-meets-geometry lecture

---

**Last Updated:** 2025
