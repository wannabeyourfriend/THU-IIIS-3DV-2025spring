# 3D Vision Computing

> Notes Taking: Alex
>
> Contact: `wang-zx23@mails.tsinghua.edu.cn`
>
> Instructor: Li Yi
>
> Reference: `Li Yi 's 3DV lecture & Hao Su's ML-meets-geometry lecture`

[TOC]

## 0 Introduction

Geometry understanding is very important in Robotics, Augmented Reality Autonomous driving and Medical Image Processing. From geometry understanding the robot can get **a priori knowledge of the 3D world**.

- Geometry theories $\to$ Curves, Surface, Rotation ···
- Sensing: Computer Representation of Geometries $\to$ Mesh, Point, ···
- Sensing: 3D reconstruction from a single image $\to$ 
- Geometry Processing: Local geometric property estimation, Surface reconstruction
- Recognition: Object classification, Object detection, 6D pose estimation, Segmentation,Human pose estimation
- Relationship Analysis: Shape correspondences

## 1 Geometry

> This Chapter mainly focus on the basic concepts, definition and of geometry.

### **1.1 Curves**

#### 1.1.1 Parameterization

##### Definition

A parameterized curve is a map from a 1-dimensional region to $R^n$ .

- 2d curve: $\gamma(t) = (x(t), y(t))$  
  Intuition: A particle moving in space with position $\gamma(t)$ at time $t$.

Use parameterized  methods to represent a curve. 

- 3d curve: $\gamma(t) = (x(t), y(t), z(t))| R \to R^3: t \to p(t)$

- $p(t) = r(cos(t), sin(t)), \quad t \in [0,2\pi)$

##### Application

Bezier Curves, Splines: 

![](assets/clipboard-image-1742101531.png)

$$
s(t) = \sum_{i = 0}^n \mathbf{p}_iB_i^n(t)
$$


A curve is just like One-dimensional “Manifold", Set of points that locally looks like a line. (however when a cusp occured things becomes extremely  complex)

- **Tangent Vector:**  
  $\gamma'(t) = (x'(t), y'(t)) \in \mathbb{R}^2$  
  Example: For $\gamma(t) = (\cos(t), \sin(t))$,  
  $\gamma'(t) = (-\sin(t), \cos(t))$  
  - $\gamma'(t)$ indicates the direction of movement.  
  - $\|\gamma'(t)\|$ indicates the speed of movement.
- **Arc length**
  $\int_a^b ||\gamma'(t)|| dt$
- **Parameterization by Arc Length**
  $s(t) = \int_{t_0}^t ||\gamma'(t)||dt$
  $t(s)$ = inverse function of $s(t)$ 
  $\hat{\gamma}(s) = \gamma(t(s))$



#### 1.1.2 2D

> Theorem

Define Tangent vector $T(s) = \gamma'(s)$, $ \implies $  $ \|T(s)\| \equiv 1 $  

##### $\|T(s)\| \equiv 1$

> Proof: By definition.

$ S(t) = \int_{t_0}^{t} \|\gamma'(t)\| dt $

$ \frac{ds}{dt} = \|\gamma'(t)\| $

$ T(s) = \|\gamma'(s)\| = \left\|\frac{d\gamma}{ds}\right\| = \left\|\frac{d \gamma}{dt}\right\| \cdot \left\|\frac{dt}{ds}\right\| = |\gamma'(t)| \|\frac{dt}{ds}\| $

$ t(s) = s^{-1}(t)\quad \frac{dt}{ds} = \frac{1}{\frac{ds}{dt}} = \frac{1}{\left\|\gamma'(t)\right\|}$

Thus, $ \|T(s)\| = \frac{\|\gamma'(t)\|}{\|\gamma'(t)\|} = 1 $

##### $N(s):= JT(s)$

Define Normal vector $ N(s)$ where $J$ is the rotation matrix of $90^{\circ}$ in 2D space.
$$
J = 
\begin{bmatrix}
0 & -1\\
1 & 0
\end{bmatrix}
$$
We have the definition of the normal vector: $N(s) := JT(s)$.

##### Frenet Equation

> Theorem

$$
\frac{d}{ds} \begin{bmatrix} T(s) \\ N(s) \end{bmatrix} := \begin{bmatrix} 0 & k(s) \\ -k(s) & 0 \end{bmatrix} \begin{bmatrix} T(s) \\ N(s) \end{bmatrix}
$$
> Proof: By $\|T(s)\| \equiv 1$ and $\frac{d}{dt}<u,v>=\frac{du}{dt}v + \frac{dv}{dt}u$

Now, let's derive the Frenet equations: We know that $T(s)$ is a unit tangent vector, meaning $|T(s)| = 1$, which implies that $\langle T(s), T(s) \rangle = 1$. When we differentiate $\langle T(s), T(s) \rangle = 1$ with respect to $s$, we get: $\langle \frac{dT}{ds}, T \rangle + \langle T, \frac{dT}{ds} \rangle = 0$ $\Rightarrow 2\langle \frac{dT}{ds}, T \rangle = 0$ $\Rightarrow \langle \frac{dT}{ds}, T \rangle = 0$ This shows that $\frac{dT}{ds}$ is orthogonal to $T$. Since $\frac{dT}{ds}$ is orthogonal to $T$, and in a 2D plane, the only orthogonal direction is along the normal vector $N$, we can write $\frac{dT}{ds} = \kappa(s)N(s)$, where $\kappa(s)$ is the curvature. For the normal vector $N(s) = JT(s)$, when we differentiate, we get: $\frac{dN}{ds} = J\frac{dT}{ds} = J(\kappa(s)N(s)) = \kappa(s)JN(s)$ Since $N(s) = JT(s)$, we have $JN(s) = J(JT(s)) = J^2T(s)$ . Computing $J^2$:
$$
J^2 =
\begin{bmatrix}
0 & -1\\
1 & 0
\end{bmatrix}
\begin{bmatrix}
0 & -1\\
1 & 0
\end{bmatrix} =
\begin{bmatrix}
-1 & 0\\
0 & -1
\end{bmatrix} = -I
$$

Therefore, $JN(s) = J^2T(s) = -T(s)$ . Substituting back: $\frac{dN}{ds} = \kappa(s)JN(s) = -\kappa(s)T(s)$

In summary, we have derived: $\frac{dT}{ds} = \kappa(s)N(s)$ $\frac{dN}{ds} = -\kappa(s)T(s)$ These equations can be expressed in matrix form:

$$
\frac{d}{ds} \begin{bmatrix} T(s) \\ N(s) \end{bmatrix} =
\begin{bmatrix}
0 & \kappa(s) \\
-\kappa(s) & 0
\end{bmatrix}
\begin{bmatrix} T(s) \\ N(s) \end{bmatrix}
$$

> Thoughts: Use the geometry self-coordinates to describe the shape of itself.

##### $\mathbb{R}^2$ Curve Theorem  

Radius of Curvature is defined as $\kappa(s) = \frac{1}{R}$  , $R$ is the radius of curvature. The geometry meaning indicated how much the normal changes in the direction tangent to the curve. Or curvature $\kappa(s)$ **characterizes a planar curve up to rigid motion**, which is always positive.

#### 1.1.3 3D

##### Osculating Plane 

The plane determined by $T(s)$ and $N(s)$. And we define the the Binormal Vector $B(s) = T(s) \times N(s)$ Curvature and Torsion

##### Curvature $\kappa$ & Torsion $\tau$

> Definition

$<N'(s), T(s)> = -\kappa(s) \quad <N'(s), B(s)> = \tau(s)$

> Theorem

$T'(s) = \kappa(s)N(s)$  $N'(s) = -\kappa(s)T(s) + \tau(s)B(s)$   $B'(s) = -\tau(s)N(s)$  

> Proof

For the first equation, we know that $T(s)$ is a unit vector, so $\|T(s)\| = 1$. Differentiating $\langle T(s), T(s) \rangle = 1$ with respect to $s$:
$\langle T'(s), T(s) \rangle + \langle T(s), T'(s) \rangle = 0$
$\Rightarrow 2\langle T'(s), T(s) \rangle = 0$
$\Rightarrow \langle T'(s), T(s) \rangle = 0$

This shows that $T'(s)$ is orthogonal to $T(s)$. Since $\{T, N, B\}$ forms an orthonormal basis, $T'(s)$ must lie in the plane spanned by $N$ and $B$:
$T'(s) = \alpha N(s) + \beta B(s)$

To find $\alpha$ and $\beta$, we compute:
$\langle T'(s), N(s) \rangle = \alpha \langle N(s), N(s) \rangle + \beta \langle B(s), N(s) \rangle = \alpha \cdot 1 + \beta \cdot 0 = \alpha$

By definition, $\alpha = \kappa(s)$. Also:
$\langle T'(s), B(s) \rangle = \alpha \langle N(s), B(s) \rangle + \beta \langle B(s), B(s) \rangle = \alpha \cdot 0 + \beta \cdot 1 = \beta$

Since $T$, $N$, and $B$ form a right-handed orthonormal basis, $\langle T'(s), B(s) \rangle = 0$, thus $\beta = 0$.
Therefore, $T'(s) = \kappa(s)N(s)$.

For the second equation, we know that $\{T, N, B\}$ is an orthonormal basis, so $N'(s)$ can be expressed as:
$N'(s) = a T(s) + b N(s) + c B(s)$

Since $\langle N(s), N(s) \rangle = 1$, differentiating gives:
$\langle N'(s), N(s) \rangle + \langle N(s), N'(s) \rangle = 0$
$\Rightarrow 2\langle N'(s), N(s) \rangle = 0$
$\Rightarrow b = 0$

From $\langle N(s), T(s) \rangle = 0$, differentiating:
$\langle N'(s), T(s) \rangle + \langle N(s), T'(s) \rangle = 0$
$\Rightarrow \langle N'(s), T(s) \rangle + \langle N(s), \kappa(s)N(s) \rangle = 0$
$\Rightarrow \langle N'(s), T(s) \rangle + \kappa(s) = 0$
$\Rightarrow a = -\kappa(s)$

By definition, $\langle N'(s), B(s) \rangle = \tau(s)$, thus $c = \tau(s)$.
Therefore, $N'(s) = -\kappa(s)T(s) + \tau(s)B(s)$.

For the third equation, since $B = T \times N$, differentiating:
$B'(s) = T'(s) \times N(s) + T(s) \times N'(s)$
$= \kappa(s)N(s) \times N(s) + T(s) \times (-\kappa(s)T(s) + \tau(s)B(s))$
$= 0 + (-\kappa(s))(T(s) \times T(s)) + \tau(s)(T(s) \times B(s))$
$= 0 + 0 + \tau(s)(T(s) \times B(s))$
$= 0 + 0 + \tau(s)(T(s) \times B(s))$

Since $\{T, N, B\}$ is a right-handed orthonormal basis, $T \times B = -N$. Thus:
$B'(s) = \tau(s)(-N(s)) = -\tau(s)N(s)$
> Thoughts

Curvature indicates how much the **normal** changes in the direction **tangent** to the curve. (Indicates in-plane motion.) Torsion indicates how much normal changes in the direction **orthogonal** to the osculating plane of the curve.(Indicates out-of-plane motion.) Curvature is always **positive** but torsion can be **negative**

##### Frenet Frame

> Theorem:

$$
\frac{d}{ds} \begin{pmatrix} T \\ N \\ B \end{pmatrix} = \begin{pmatrix} 0 & \kappa & 0 \\ -\kappa & 0 & \tau \\ 0 & -\tau & 0 \end{pmatrix} \begin{pmatrix} T \\ N \\ B \end{pmatrix}
$$

> Proof: By the relations above.

##### $\mathbb{R}^3$ Curve Theorem

Curvature $\kappa(s)$ and torsion $\tau(s)$ characterize a 3D curve up to rigid motion.

#### 1.1.4 Geometry Meaning

A curve is defined as a **map** from an **interval** to $\mathbb{R}^n $ The **tangent vecto**r to the curve describes the **direction of motion along the curve**. When the curve is parameterized by arc-length, the derivative of the tangent vector is the normal vector. Both **curvature** and **torsion** are measures that **describe the change in the normal direction of the curve**. Curvature quantifies how much the normal vector changes in the direction tangent to the curve, while **torsion** quantifies **how much the normal vector changes in the direction orthogonal to the osculating plane of the curve**. Curvature is always positive, indicating the rate of bending, whereas torsion can be negative, indicating twisting. Together, curvature and torsion uniquely describe the shape of a curve, up to rigid transformations. The tangent, normal, and binormal vectors together form a moving frame, known as the Frenet frame, which provides a local coordinate system that moves along the curve.

### **1.2 Surface** 

#### 1.2.1 Surface Parametrization

##### $f: U \to \mathbb{R}^3$

- A parameterized surface is a map from a two-dimensional region  $ U \subset \mathbb{R}^2 $ to  $ \mathbb{R}^n $.

![image-20250319113942255](assets/image-20250319113942255.png)

- The set of points $f(U)$ is called the image of the parameterization

###### Saddle Example

$$
U := \{(u, v) \in \mathbb{R}^2 : u^2 + v^2 \leq 1\}\\
f(u, v) = [u, v, u^2 - v^2]^T
$$

![image-20250319114325890](assets/image-20250319114325890.png)

#### 1.2.2 Differentiable Manifold

> Inspiration

- Things that can be discovered by local observation: point + neighborhood.

> Properties

- **Local Properties**: properties that can be discovered by local observation (points + neighborhoods).
- **Smoothness**: a continuous one-to-one mapping from local to global.
- **Tangent Plane**: each point can have a tangent plane attached to it, which contains all possible directions passing tangentially from that point, defined as $T_p(\mathbb{R}^3)$

##### $Df_p$

Differential of a Surface  $Df_p: T_p(\mathbb{R}^2) \to T_{f(p)}(\mathbb{R}^3)$

- Relate the movement of point in the domain and on the surface.

$df = \frac{\partial f}{\partial u} du + \frac{\partial f}{\partial v} dv$

- If the point $ p \in \mathbb{R}^2 $ is moving along the vector $ X = [u, v]^T $ with velocity $ \epsilon $, the motion of the point $ f(p) $ on the surface is:
    $
    \Delta f_p \approx \frac{\partial f}{\partial u} (\epsilon u) + \frac{\partial f}{\partial v} (\epsilon v) = \epsilon \left[ \frac{\partial f}{\partial u}, \frac{\partial f}{\partial v} \right] \begin{bmatrix} u \\\\ v \end{bmatrix} = \epsilon [Df_p]X
    $
  $
  Df_p := \left[ \frac{\partial f}{\partial u}, \frac{\partial f}{\partial v} \right] \in \mathbb{R}^{3 \times 2}
  $ is a linear mapping that maps tangent vectors in the parameter domain to tangent vectors in space, where  $X$ is the velocity in the 2D domain, and the $[Df_P]X$ is the velocity in the 3D space.

![image-20250319115030694](assets/image-20250319115030694.png)

> Thought

- Intuitively, the differential of a parameterized surface tells us how tangent vectors on the domain get mapped to tangent vectors in space. w.r.t, Maps a vector in the tangent space of the domain to the tangent space of the surface.
- Tells us the velocity of point in 3D when the parameter
  changes in 2D.
- Allows us to construct the bases of tangent plane.

###### Saddle Example-Continue

![image-20250319115554494](assets/image-20250319115554494.png)
$$
f(u, v) = [u, v, u^2 - v^2]^T\\
Df_p = \begin{bmatrix}
  \frac{\partial f_1}{\partial u} & \frac{\partial f_1}{\partial v} \\
  \frac{\partial f_2}{\partial u} & \frac{\partial f_2}{\partial v} \\
  \frac{\partial f_3}{\partial u} & \frac{\partial f_3}{\partial v}
  \end{bmatrix}
  = \begin{bmatrix}
  1 & 0 \\
  0 & 1 \\
  2u & -2v
  \end{bmatrix}
\\
  X := \frac{3}{4} \begin{bmatrix} 1 \\ -1 \end{bmatrix}
 \quad
  Df(X) = \frac{3}{4} \begin{bmatrix} 1 \\ -1 \\ 2(u + v) \end{bmatrix}
 
 \\
\text{e.g. for } (u, v) = (0,0) \quad 
Df(X) = \left[ \frac{3}{4}, -\frac{3}{4}, 0 \right]^T \\

\text{e.g. for }p = (u, v) = (1, 1), f(p) = (1,1,0) \quad 
\\
T_{f(p)}(\mathbb{R}^3) 
= \text{span of }\begin{bmatrix}
1 & 0\\
0 & 1\\
2 & -1
\end{bmatrix}
$$

#### 1.2.3 Curvature

##### $N_p$

> Definition

$$
N(u, v) = \frac{f_u \times f_v}{\| f_u \times f_v \|}\\
 \text{where} f_u = \frac{\partial f}{\partial u} \quad  f_v = \frac{\partial f}{\partial v}
$$

###### Cylinder Example

$$
f(u, v) := [\cos(u), \sin(u), u + v]^T \\
Df_{(u,v)} = \begin{bmatrix} -\sin(u) & 0 \\ \cos(u) & 0 \\ 1 & 1 \end{bmatrix}\\
N(u,v) = \begin{bmatrix} -\sin(u) \\ \cos(u) \\ 1 \end{bmatrix} \times \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} \cos(u) \\ \sin(u) \\ 0 \end{bmatrix}
$$

| Calculate Normal on a surface                                | Local change of normal                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20250319152056330](assets/image-20250319152056330.png) | ![image-20250319152234311](assets/image-20250319152234311.png) |

> Local change

Assume $q$ moves along a curve $\gamma $ parameterized by arclength $q = \gamma(s)$:, and the normal is $N(s)$ with unit norm. From $\frac{d}{ds}<N(s), N(s)> = 0$ . We know that the local change of normal is always in the tangent plane!

##### $DN_p$

![image-20250319153541423](assets/image-20250319153541423.png)
$$
dN = \frac{\partial N}{\partial u} du + \frac{\partial N}{\partial v} dv \\
\text{If point } p \in \mathbb{R}^2 \text{moves with velocity}\\
 X \text{ by }  \epsilon \text{, the movement of } N_p:
\\
\Delta N_p = \frac{\partial N}{\partial u} (\epsilon u) + \frac{\partial N}{\partial v} (\epsilon v) = \epsilon \begin{bmatrix} \frac{\partial N}{\partial u}, \frac{\partial N}{\partial v} \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = \epsilon [DN_p] X
\\
DN_p := \begin{bmatrix} \frac{\partial N}{\partial u}, \frac{\partial N}{\partial v} \end{bmatrix} \in \mathbb{R}^{3 \times 2} 
$$
Let $\|Df_p[\mu X]\| = 1$, $\mu = \frac{1}{\|Df_pX\|}$, thus $DN_p[\mu X] = \frac{DN_pX}{\|Df_pX\|}$. 

#####  $\mathbf{\kappa} $

> Definition

Vector $\mathbf{\kappa} = DN_p[\mu X] = \frac{DN_pX}{\|Df_pX\|}$

> Principal Curvatures

$$
\kappa_n := <\mathbf{T}, \kappa> = \frac{<Df_pX, DN_pX>}{\|Df_pX|\|^2}>
$$

![image-20250319155713365](assets/image-20250319155713365.png)

>Geodesic curvature

$$
\kappa_g := <\kappa, \mathbf{N} \times  \mathbf{T}>
$$



![image-20250319155800769](assets/image-20250319155800769.png)

###### Cylinder Example-Continue

| Calculte $\kappa_n$                                          | Cylinder                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $$Df_{p} = \begin{bmatrix} -\sin(u) & 0 \\ \cos(u) & 0 \\ 1 & 1 \end{bmatrix}\\N_p = \begin{bmatrix} -\sin(u) \\ \cos(u) \\ 1 \end{bmatrix} \times \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} \cos(u) \\ \sin(u) \\ 0 \end{bmatrix}\\DN_p =\begin{bmatrix}-\sin(u) & 0\\cos(u) & 0\\0 & 0\end{bmatrix} \\  \text{Thus }  X_1 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad \kappa_n(X_1) = \frac{\langle Df(X_1), DN(X_1) \rangle}{\|Df(X_1) \|^2} = 0,\\X_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}, \quad \kappa_n(X_2) = \frac{\langle Df(X_2), DN(X_2) \rangle}{\|Df(X_2) \|^2} = 1$$ | ![image-20250319160502656](assets/image-20250319160502656.png) |

##### $\kappa_1$ $\kappa_2$

> Definition

- The direction that bends fastest / slowest are principal directions, which are orthogonal to each other.

$$
\text{Maximum curvature }\kappa_1 = \kappa_{\text{max}} = \max_{\phi} \kappa_n(\phi) , \\
\quad \phi_1 \to \text{Principle directure 1}\\
\text{Minimun curvature } \kappa_2 = \kappa_{\text{min}} = \min_{\phi} \kappa_n(\phi)\\
\quad \phi_2 \to \text{Principle directure 2}\\
$$

| Visualization                                                | min curvature && max curvature                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20250319162006411](assets/image-20250319162006411.png) | ![image-20250319162134564](assets/image-20250319162134564.png) |

> Theorem

The principal directions are always orthogonal.

> Proof

Consider the shape operator (Weingarten mapping) $$S_p: T_p(M) \to T_p(M)$$, which can be expressed as:$$S_p(X) = -DN_p(X)$$ where $$DN_p$$ is the normal vector differential. The shape operator is self-adjoint, i.e., for any tangent vectors $X, Y \in T_p(M)$: $$\langle S_p(X), Y \rangle = \langle X, S_p(Y) \rangle$$. The principal curvatures $\kappa_1, \kappa_2$ are the eigenvalues of the shape operator $$S_p$$ and the corresponding principal directions $$\phi_1, \phi_2$$ are its eigenvectors: $$S_p(\phi_1) = \kappa_1 \phi_1$$ $$S_p(\phi_2) = \kappa_2 \phi_2$$. Since $$S_p$$ is self-concomitant, when $$\kappa_1 \neq \kappa_2$$, the corresponding eigenvectors are necessarily orthogonal. The proof is as follows: $$ \langle S_p(\phi_1), \phi_2 \rangle = \langle \kappa_1 \phi_1, \phi_2 \rangle = \kappa_1 \langle \phi_1, \phi_2 \rangle$$ Simultaneous: $$\langle \phi_1, S_p(\phi_2) \rangle = \langle \phi_1, \kappa_2 \phi_2 \rangle = \kappa_2 \langle \phi_1, \phi_2 \rangle$$ By the self-concomitant property: $$\langle S_p(\phi_1), \phi_2 \rangle = \langle \phi_1, S_p(\phi_2) \rangle$$, thus: $$ \kappa_1 \langle \phi_1, \phi_2 \rangle = \kappa_2 \langle \phi_1, \phi_2 \rangle$$ $$(\kappa_1 - \kappa_2) \langle \phi_1, \phi_2 \rangle = 0$$

| ---                                                          | ---                                                          | ---                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20250319163504822](assets/image-20250319163504822.png) | ![image-20250319163544497](assets/image-20250319163544497.png) | ![image-20250319163550290](assets/image-20250319163550290.png) |

> Theorem: Euler’s Theorem: 

Planes of principal curvature are orthogonal and independent of parameterization. 
$$
\kappa_n(\phi) = \kappa_1 \cos^2 \phi + \kappa_2 \sin^2 \phi
$$

##### Shape Operator

> Definition

- The shape operator $S$ is a linear map that relates the change in the normal vector to the change in the surface point. $DN_p(X) $and $ Df_p(X) $ are both in the tangent plane. Therefore, the column space of$DN_p$is a subspace of the column space of $Df_p$.

$$
\exists S \in \mathbb{R}^{2 \times 2} \quad \text{such that} \quad DN_p = Df_p S \\
\text{This implies: }
\forall X \in T_p(\mathbb{R}^2), \quad [DN_p]X = [Df_p]S X
$$

- Actually,  $S$ is the "Normal Change Prediction Operator",  When a point $p$ moves along a direction $SX$, the normal change vector $\vec d \in \mathbb{R}^3$. $S$ can represent some information about the normal of the surface. Actually，this linear map$S$ predicts the normal change when $p$ moves along any direction.

> Computation of Principal Directions

$S$ has some super cool properties:

- The principal directions are the eigenvectors of the shape operator $S$ 
- The principal curvatures are the eigenvalues of $S$
- Note: The shape operator $ S $ is a linear map that relates the change in the normal vector to the change in the surface point.

###### Cylinder-example-continue

$$
f(u, v) = [\cos(u), \sin(u), u + v]^T\\
Df = \begin{bmatrix}
-\sin(u) & 0 \\
\cos(u) & 0 \\
1 & 1
\end{bmatrix}

N = [\cos(u), \sin(u), 0]^T
\\
DN = \begin{bmatrix}
-\sin(u) & 0 \\
\cos(u) & 0 \\
0 & 0
\end{bmatrix}\\
X_1 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad \kappa_n(X_1) = 0\\
X_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}, \quad \kappa_n(X_2) = 1\\
\text{To verify the eigenvalues of} S:
DN_p = Df_p S \Rightarrow S = \begin{bmatrix} 1 & 0 \\ -1 & 0 \end{bmatrix}
$$

#### 1.2.4 First Fundamental Form

##### First Claim 

Curvature completely determines local surface geometry. However, it is insufficient to determine surface globally. See this below as an example：  $\exist f \text{and} f^*$ curvature value and directions are the same for any pair $(f(p), f^*(p)), \forall p \in U$.

![image-20250320102448638](assets/image-20250320102448638.png)

> Inspiration

Other than measuring how the surface bends, we should also measure length and angle.

##### Definition

The first fundamental form $ I_p $ is defined as the inner product in the tangent space $ T_p(\mathbb{R}^3) $.

$I_p(X, Y) = \langle Df_p X, Df_p Y \rangle$ where $ X, Y \in T_p(\mathbb{R}^2) $. $I_p(X, Y) = X^T (Df_p^T Df_p)Y$

 This form $I_p$ is dependent on both the surface $ f $ and the point $ p $.

- Arc-length by $ I(X, Y) $ : The arc-length of a curve on the surface can be determined using the first fundamental form.

  - **Velocity of a Point**:
    - Suppose a point $ p \in U $ moves with velocity $ X(t) $.
    - The curve on the surface is given by:
      $
      \gamma(t) = f(p(t)) = f(p_0 + \int_0^t X(t) dt)
      $
    - The derivative of the curve is:
      $
      \gamma'(t) = Df_p(t) [X(t)]
      $
    - The arc-length $ s(t) $ is:
      $$
      s(t) = \int_0^t \| \gamma'(t) \| dt \\
      = \int_0^t \sqrt{\langle Df_p(t) X(t), Df_p(t) X(t) \rangle }dt\\
      = \int_0^t \sqrt{I_p(t)(X(t), X(t))} dt
      $$

- With $I$ , we have completely determined curve length within the surface without referring to $f$

##### Local Isometric Surfaces Example

![image-20250320103111200](assets/image-20250320103111200.png)

Two surfaces $ M $ and $ M^* $ are locally isometric if there exist parameterizations $ f $ and $ f^* $ such that the first fundamental forms are equal.

$
f(u, v) = [u, v, 0]^T \quad \text{and} \quad f^*(u, v) = [\cos u, \sin u, v]^T
$
on $ U = \{(u, v) : u \in (0, 2\pi), v \in (0, 1)\} $.

Proof:

For the plane parameterization $f(u,v) = [u, v, 0]^T$:
$$Df_p = \begin{bmatrix} 
1 & 0 \\
0 & 1 \\
0 & 0
\end{bmatrix}$$

Computing the first fundamental form matrix:
$$Df_p^T Df_p = \begin{bmatrix} 
1 & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}
\begin{bmatrix} 
1 & 0 \\
0 & 1 \\
0 & 0
\end{bmatrix} = 
\begin{bmatrix} 
1 & 0 \\
0 & 1
\end{bmatrix}$$

For the cylinder parameterization $f^*(u,v) = [\cos u, \sin u, v]^T$:
$$Df^*_p = \begin{bmatrix} 
-\sin u & 0 \\
\cos u & 0 \\
0 & 1
\end{bmatrix}$$

Computing the first fundamental form matrix:
$$Df^*_p{}^T Df^*_p = \begin{bmatrix} 
-\sin u & \cos u & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} 
-\sin u & 0 \\
\cos u & 0 \\
0 & 1
\end{bmatrix} = 
\begin{bmatrix} 
\sin^2 u + \cos^2 u & 0 \\
0 & 1
\end{bmatrix} = 
\begin{bmatrix} 
1 & 0 \\
0 & 1
\end{bmatrix}$$

Since $\sin^2 u + \cos^2 u = 1$, we have:
$$Df_p^T Df_p = Df^*_p{}^T Df^*_p = 
\begin{bmatrix} 
1 & 0 \\
0 & 1
\end{bmatrix}$$

Therefore, the first fundamental forms of the plane and cylinder are identical:
$$I_p(X, Y) = X^T(Df_p^T Df_p)Y = X^T(Df^*_p{}^T Df^*_p)Y = I^*_p(X, Y)$$

This proves that the plane and cylinder are locally isometric. Intuitively, this makes sense because we can roll a plane into a cylinder without stretching or tearing, preserving all distances and angles.

Here are some applications of first form.

- Shape Classification by Isometry

  ![image-20250320110338366](assets/image-20250320110338366.png)

- Geodesic Distances

  ![image-20250320110508098](assets/image-20250320110508098.png)



- Distance Distribution Descriptor

  > Compute distribution of distances for point pairs by randomly picked on the surface

  ![image-20250320110619910](assets/image-20250320110619910.png)



- The angle between two vectors on the surface can be determined using the first fundamental form.
  $ \cos \phi = \frac{\langle Df_p X, Df_p Y \rangle}{\| Df_p X \| \| Df_p Y \|} = \frac{I(X, Y)}{\sqrt{I(X, X) I(Y, Y)}} $

With $I$, we have completely determined angles within the surface without referring to $f$

##### Second Fundamental Form

$$
II(X, Y) = \langle DN_p X, Df_p Y \rangle
$$

> Theorem

A smooth surface is determined up to rigid motion by its first and second fundamental forms.

#### 1.2.6 Gaussian and Mean Curvature

> Definition

- **Gaussian Curvature**:
  $
  K := \kappa_1 \kappa_2
  $
- **Mean Curvature**:
  $
  H := \frac{1}{2} (\kappa_1 + \kappa_2)
  $

> Theorem

Gaussian and mean curvature also fully describe local bending.

![image-20250320111946891](assets/image-20250320111946891.png)

> Gauss's Theorema Egregium

The Gaussian curvature of an embedded smooth surface in  $\mathbb{R}^3$is invariant under the local isometries.

> Thought

Locally Isometric Surfaces are invariant measured by Gaussian curvature. Gaussian curvatures are vulnerable to noises in practice and not informative. Needed for more robust surface analysis.

## 2 Representations

> This chapter mainly focuses on 3D representations, including mesh, point cloud and implicit representation methods.

## 3 Transformation

> This chapter focuses on the transformation and rotation of 3D objects.

## 4 Reconstruction

> This chapter focus on methods of reconstruct 3D information from pictures.

