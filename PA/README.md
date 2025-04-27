# 3 Surface Reconstruction [7pt]

**Problem 3. Deep Marching Tetrahedra (DMTet)** is a hybrid 3D representation that combines both implicit and explicit 3D surface representations. It represents a shape with a discrete SDF defined on vertices of a deformable tetrahedral grid. The SDF is converted to triangular mesh via a differentiable marching tetrahedra algorithm (MT), allowing explicit supervision on the extracted surface to be back-propagated to SDF and change mesh topology. In this problem, we aim at reconstructing meshes from point clouds via DMTet.

Given input point clouds or coarse voxels, DMTet optimizes both the SDF values and the 3D deformation of the tetrahedra grid's vertices so that the resulting mesh, obtained via marching tetrahedra, can approximate the input shape. At its core, DMTet employs a deformable tetrahedral grid representation that encodes a discretized SDF, along with a differentiable MT algorithm. Similar to the Marching Cubes (MC) algorithm covered in class, MT is an iso-surfacing method designed to extract an explicit mesh representation from an implicit SDF. However, instead of using a regular grid of cubes, MT partitions space into tetrahedra. This tetrahedral representation helps MT avoid many issues present in MC such as the topological ambiguities.

MT algorithm converts the SDF, encoded in a tetrahedral grid, into a triangular mesh by determining the surface topology within each tetrahedron based on the signs of its vertices. Since each tetrahedron has four vertices, there are a total of $2^4 = 16$ possible sign configurations. Considering rotational symmetry, these configurations reduce to three unique cases, each with an unambiguous surface topology:

- Case 1: All four vertices have the same sign → No surface is formed.
- Case 2: Three vertices are negative, while the fourth is positive → A
  triangle separates the positive vertex from the other three.
- Case 3: Two vertices are positive and two are negative → A quadrilateral
  separates them, which can be further subdivided into two triangles

![image-20250426123447546](assets/image-20250426123447546.png)

$v_{ab} \text{ is determined by } v_{ab} = \frac{v_a \c1 dot s(v_b) - v_b \cdot s(v_a)}{s(v_b) - s(v_a)}.$ Note that flipping the sign would not change the surface topology.

DMTet solves the surface reconstruction problem by optimizing the deformation and SDF value of the tetrahedra grid vertices. “Deep” comes from its utilizing neural networks to parameterize the deformation and SDFs. Specifically, for a tetrahedra grid $T = \{V \in \mathbb{R}^{N_v \times 3}, T \in \mathbb{N}^{N_t \times 4}\}$, where $N_v$ is the number of vertices and $N_t$ is the number of tetrahedra, it parameterize the deformation $d(v)$ and the sign $s(v)$ of a vertex $v$ via neural networks $\phi$ with parameters $\theta$: $d(v), s(v) = \phi_\theta(v, T)$. By defining losses between the extracted surface and the input point cloud, the network can be trained in an end-to-end manner, leveraging the differentiability of the MT algorithm.

In this problem, we aim to implement DMTet and use it to reconstruct surfaces from input point clouds. The original DMTet paper also introduces a tetrahedron subdivision technique, which adaptively refines the tetrahedral grid in specific areas to better capture fine details, such as the thin structures of a mouse’s tail. However, you do not need to implement this technique for this homework.

In this problem, you are required to implement the neural networks, the marching tetrahedra algorithm, and the training pipeline for DMTet. While the original implementation of DMTet is not open-sourced, you can find useful references in other GitHub repositories, such as [GET3D](https://github.com/GET3D/GET3D). Feel free to explore and learn from these resources, but direct copy-and-paste is not allowed.

Requirements are detailed in the following:

1. **Network implementation.** Implement the following two types of networks to parameterize the SDF and deformation of the tetrahedral grid:

   - **MLP.** A fully connected neural network that takes vertex positions as input and outputs the SDF and deformation: $d(v), s(v) = \text{MLP}_\theta(v)$. Instead of directly using the raw $(x, y, z)$ coordinates, apply a Fourier-based positional encoding to transform the input into a higher-dimensional space. This helps prevent over-smoothed reconstructions, similar to NeRF.

   - **3D Convolution Network.** A 3DConv network that accepts vertex position and outputs SDF and deformation, i.e., $d(v), s(v) = 3DConv_\theta(v)$. Similarly, instead of using the original vertex coordinates, you may add a positional encoding layer.

2. **Marching tetrahedra.** MT algorithm can be implemented following the process described above. In GET3D, a precomputed look-up table is used to efficiently determine edges by encoding and retrieving them from the table. There are no strict requirements on how you implement this process—as long as your function is correct, you will receive full credit for this part. Additionally, while the `kaolin` package provides a built-in marching tetrahedra function, you may only use it to verify the correctness of your implementation. Directly using it in your homework is not allowed.

3. **Reconstruction loss.** MT is differentiable. After extracting the mesh, we can optimize the network by defining losses between the mesh and the original point cloud and backward the gradient. Implement the Chamfer Distance loss, which measures the distance between points sampled from the surface of the extracted triangular mesh and the input point cloud. The process of sampling points from the mesh surface should be differentiable. You may either implement the sampling yourself or use a function from an existing Python package, e.g., `kaolin.ops.mesh.sample_points`. In addition to the CD loss, add a Laplacian regularization term to improve the smoothness of the reconstructed surface. This results in the following final loss function:
   $$
   \mathcal{L} = \mathcal{L}_{CD} + \lambda_{reg} \mathcal{L}_{reg}.
   $$
   Try to vary the parameter $\lambda_{reg}$ and analyze the influence of $\mathcal{L}_{reg}$ on the final reconstruction mesh. One reference implementation of $\mathcal{L}_{reg}$ is as follows:

   ```python
   def laplace_regularizer_const(mesh_verts, mesh_faces):
       term = torch.zeros_like(mesh_verts)
       norm = torch.zeros_like(mesh_verts[..., 0:1])
       v0 = mesh_verts[mesh_faces[:, 0], :]
       v1 = mesh_verts[mesh_faces[:, 1], :]
       v2 = mesh_verts[mesh_faces[:, 2], :]
       term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
       term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
       term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))
       two = torch.ones_like(v0) * 2.0
       norm.scatter_add_(0, mesh_faces[:, 0:1], two)
       norm.scatter_add_(0, mesh_faces[:, 1:2], two)
       norm.scatter_add_(0, mesh_faces[:, 2:3], two)
       term = term / torch.clamp(norm, min=1.0)
       return torch.mean(term**2)

You can find the point clouds for this problem in the folder Problem3-pcs. You are expected to experiment with tetrahedra grids of various resolutions and analyze their corresponding results. The folder Problem3.tets contains six tetrahedra grids in the format `{grid_res}_compress.npz`. For your experiments, please consider at least four of these grids. Each tetrahedra grid file is a dictionary that describes the vertex locations and the tetrahedra structure. Specifically:

- The value of vertices is a NumPy array with shape $N_v \times 3$, where each row corresponds to the coordinates of a vertex.
- The value of tets is a NumPy array with shape $N_t \times 4$, where each row contains the indices of the vertices that define a tetrahedron.

For this problem, please submit the following materials:

- **Report:**
  - Visualizations of the reconstructed meshes and the corresponding sampled point clouds.
  - Comparisons and analysis of the results.
- **Sampled Points:** Save the sampled points in formats such as `.npy`, `.ply`, etc.
- **Reconstructed Meshes:** Save the reconstructed meshes in formats such as `.obj`, `.ply`, etc.
- **Source code:** Include all the source code used for your experiments and implementations.

### File structure
.
├── Problem3
│   ├── data
│   │   ├── pcs
│   │   │   ├── apple_pts.ply
│   │   │   ├── ...
│   │   └── tets
│   │       ├── 64_compress.npz
│   │       ├── ...
│   ├── src
│   │   ├── test_pipeline.py
│   │   ├── train.py
│   │   ├── utils.py
│   │   └── model.py
│   ├── outputs
│   │   ├── ...
│   ├── configs
│       └── config.yaml
├── requirements.txt
│   └── README.md
└── assets
    └── image-20250426123447546.png