# 3DV

> Alex
>
> `wang-zx23@mails.tsinghua.edu.cn`
>
> Reference: `Li Yi 's 3DV lecture & Hao Su's ML-meets-geometry lecture`

[TOC]

## Introduction

Geometry understanding is very important in Robotics, Augmented Reality Autonomous driving and Medical Image Processing. From geometry understanding the robot can get a priori knowledge of the 3D world.

- Geometry theories
- Computer Representation of Geometries
- Sensing: 3D reconstruction from a single image
- Geometry Processing: Local geometric property estimation, Surface reconstruction
- Recognition: Object classification, Object detection, 6D pose estimation, Segmentation,Human pose estimation
- Relationship Analysis: Shape correspondences, 

## Curves

### Definition of Curve
- **Parameterized Curve:**  
  \(\gamma(t) = (x(t), y(t))\)  
  Intuition: A particle moving in space with position \(\gamma(t)\) at time \(t\).


Use parameterized  methods to represent a curve. $\gamma(t) = (x(t), y(t), z(t))| R \to R^3: t \to p(t)$

eg: $p(t) = r(cos(t), sin(t)), \quad t \in [0,2\pi)$

Application: Bezier Curves, Splines: 

![](assets/clipboard-image-1742101531.png)

$$
s(t) = \sum_{i = 0}^n \mathbf{p}_iB_i^n(t)
$$


A curve is just like One-dimensional “Manifold", Set of points that locally looks like a line. (however when a cusp occured things becomeds com    	plex)

### Tangent Vector
- **Tangent Vector:**  
  \(\gamma'(t) = (x'(t), y'(t)) \in \mathbb{R}^2\)  
  Example: For \(\gamma(t) = (\cos(t), \sin(t))\),  
  \(\gamma'(t) = (-\sin(t), \cos(t))\)  
  - \(\gamma'(t)\) indicates the direction of movement.  
  - \(\|\gamma'(t)\|\) indicates the speed of movement.

- **Arc length**
    \(\int_a^b ||\gamma'(t)|| dt\)

- **Parameterization by Arc Length**
    \(s(t) = \int_{t_0}^t ||\gamma'(t)||dt\)
    \(t(s)\) = inverse function of \(s(t)\) 
    \(\hat{\gamma}(s) = \gamma(t(s))\)

### Moving Frame in 2D
- **Tangent and Normal Vectors:**  
  - Tangent vector \(T(s) = \gamma'(s)\), \( \implies \) （on board） \( \|T(s)\| \equiv 1 \)  
#### Derivation of \(\|T(s)\| \equiv 1\)

\[ S(t) = \int_{t_0}^{t} \|\gamma'(t)\| dt \]

\[ \frac{ds}{dt} = \|\gamma'(t)\| \]

\[ T(s) = \|\gamma'(s)\| = \left\|\frac{dr}{ds}\right\| = \left\|\frac{dr}{dt}\right\| \cdot \left\|\frac{dt}{ds}\right\| = \frac{1}{\left\|\gamma'(t)\right\|} \]

\[ t(s) = s'(t) \]

\[ \|T(s)\| = \frac{\|\gamma'(t)\|}{\|\gamma'(t)\|} = 1 \]
  - Normal vector \( N(s) := JT(s) \)
$$
J = 
\begin{bmatrix}
0 & -1\\
1 & 0
\end{bmatrix}
$$
- **Derivation:**  
  \(\frac{d}{ds} \langle u(s), v(s) \rangle = \langle \frac{du}{ds}, v \rangle + \langle u, \frac{dv}{ds} \rangle\)  
  - For \(T(s)\): \(\|T(s)\| \equiv 1\)  
  - For \(N(s)\): \(N'(s) = -\kappa(s)T(s)\)  
  \[
  \frac{d}{ds} \begin{pmatrix} T(s) \\ N(s) \end{pmatrix} := \begin{pmatrix} 0 & k(s) \\ -k(s) & 0 \end{pmatrix} \begin{pmatrix} T(s) \\ N(s) \end{pmatrix}
  \]
  proof:
1. \(\frac{dT}{ds} = kN\)
2. \(\frac{dN}{ds} = -kT\)
3. \(<T, N> = 0\)
4. \(<T, T> \equiv 1\), \(\Rightarrow <\frac{dT}{ds}, T> + <T, \frac{dT}{ds}> = 0\)
5. \(\langle T, N \rangle = 0\)
6. \(\langle N, N \rangle \equiv 1\), \(\Rightarrow \langle \frac{dN}{ds}, N \rangle \equiv 0 \Rightarrow N \perp T\)
7. \(\frac{dN}{ds} = \alpha \cdot T\)
8. \(\alpha = -k\)
  - Curvature \(\kappa(s)\) indicates how much the normal changes in the direction tangent to the curve.

### Radius of Curvature
- **Curvature:** \(\kappa(s)\)  
- **Radius of Curvature:**  
  \(\kappa(s) = \frac{1}{R}\)  
  - \(R\) is the radius of curvature.

### Invariance
- **Fundamental Theorem of Plane Curves:**  
  Curvature \(\kappa(s)\) characterizes a planar curve up to rigid motion.

### 3D Curves
- **Osculating Plane:**  
  The plane determined by \(T(s)\) and \(N(s)\).  
- **Binormal Vector:**  
  \(B(s) = T(s) \times N(s)\)  
  - Defines the osculating plane.

### Curvature and Torsion
- **Curvature:**  
  \(T'(s) = \kappa(s)N(s)\)  
  - Indicates in-plane motion.
- **Torsion:**  
  \(N'(s) = -\kappa(s)T(s) + \tau(s)B(s)\)  
  - Indicates out-of-plane motion.
- **Binormal Derivative:**  
  \(B'(s) = -\tau(s)N(s)\)  
  - Torsion \(\tau(s)\) can be negative.

### Frenet Frame
- **Frenet Frame:**  
  - Tangent \(T(s)\)  
  - Normal \(N(s)\)  
  - Binormal \(B(s)\)  
- **Fundamental Theorem of Space Curves:**  
  Curvature \(\kappa(s)\) and torsion \(\tau(s)\) characterize a 3D curve up to rigid motion.



Gaussion curvature and mean curvature

