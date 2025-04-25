# 1 Volume Rendering [6pt]

**Problem 1.** In this exercise, you will recap what you have learned in Lecture 6, and implement a simple volume rendering method. You will then utilize this method to render multi-view images from a pre-trained NeRF.

We provide a codebase in Problem1 folder and what you need to do is to fill in some core codes of volume rendering.

**Setup.** First install Miniconda or Anaconda. Then you can set up the codebase with the following commands:

```bash
conda env create -f environment.yaml
conda activate hw2_nerf
```
**Code structure.** There are five main components of our codebase:

- The camera: `pytorch3d.CameraBase`
- The ray data structure: `RayBundle` in `ray_utils.py`
- The scene: `SDFVolume` and `NeuralRadianceField` in `implicit.py`
- The sampling routine: `StratifiedSampler` in `sampler.py`
- The renderer: `VolumeRenderer` in `renderer.py`

`StratifiedSampler` provides a simple ray marching method that samples multiple points along a ray traveling through the scene. The sampler and the renderer jointly make up a rendering pipeline. Like traditional graphics pipelines, this rendering procedure is independent of the scene and camera. The scene, sampler and renderer are all packaged together in the `Model` class in `main.py`, where the forward method executes a rendering process with a scene and a sampling strategy as input.

To perform volume rendering, you need to implement the following procedures:

- Ray sampling from cameras. You will complement some functions in `ray_utils.py` to generate rays in the world coordinate system from a particular camera.

- Point sampling along rays. You will complement the `StratifiedSampler` class to generate sample points along each ray.

- Rendering. You will complement the `VolumeRendering` class to evaluate a volume function at each sample point along a ray, and then aggregate the evaluations to form a rendering result.

1. **[Programming Assignment] Ray sampling.** Please look at the `render_images` function in `main.py`. The function enumerates each predefined camera view and renders an RGB image from the camera. The first step for the rendering is to acquire all the pixels from an image and generate corresponding camera rays in the world coordinate system:

```python
xy_grid = get_pixels_from_image(image_size, camera) # TODO (1): implement in ray_utils.py
ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera) # TODO (1): implement in ray_utils.py
```

You need to implement `get_pixels_from_image` and `get_rays_from_pixels` in `ray_utils.py`. The `get_pixels_from_image` function generates pixel coordinates in the range $[-1, 1]$. The `get_rays_from_pixels` function generates rays from pixels and transforms the rays from the camera space to the world space.

You can run the code for this problem by the following command:

```bash
python main.py --config-name=box
```

After you have implemented these functions, please verify that your output matches the TA's output by visualizing `xy_grid` and `ray_bundle` in `render_images`. The visualizations of grid and ray should look like `ta_images/grid_vis.png` and `ta_images/ray_vis.png`:

2. **[Programming Assignment] Point sampling.** The second step for the rendering is to sample multiple 3D points along a ray with a uniform sampling strategy. You need to implement the forward method of `StratifiedSampler` in `sampler.py` with the following routine:

(a) Uniformly generate a set of distances between [near, far].

(b) Use these distances to compute point offsets from ray origins `RayBundle.origins` along ray directions `RayBundle.directions`.

(c) Store the sampled distances and points in `RayBundle.sample_points` and `RayBundle.sample_lengths`.

After you have finished this method, you can visualize the result by first filling out the relevant codes in `render_images` and then running the command in Problem 1-1. The visualization of sample points should look like `ta_images/point_vis.png`:

4. **[Programming Assignment] Rendering.** Let us come back to the implementation of volume rendering. The final step for the rendering is to integrate emissions along a ray to form a color observation at a pixel. You need to complement the forward method of `VolumeRenderer` in `renderer.py`. Two functions `_compute_weights` and `_aggregate` are used in the forward method. The `_compute_weights` function computes the weight $ w_i = T(x_0, x_i)(1 - e^{-\sigma_i \Delta l_i}) $ for each sample point, where $ x_0 $ is the ray origin, $ x_{i \geq 1} $ is the sample point, $ \sigma $ is density, $ \Delta t $ is the length of current ray segment, and $ T(x_0, x_i) = T(x_0, x_{i-1})e^{-\sigma_{i-1}\Delta l_{i-1}} $ is the transmittance. Note that $ T(x_0, x_1) = 1 $. The `_aggregate` function aggregates emissions by a weighted sum:

$$ L(x, \omega) = \sum_{i=1}^{n} w_i L_e(x_i, \omega), $$

where $ \omega $ is the ray direction, $ L_e $ is the emission, and $ L $ is the final rendered color.

You will also render a depth map in addition to color from a volume.

After you have implemented the method, please finish the `render_images` function and run the command in Problem 1-1. Your rendering result will be written to `images/render_cube.gif`:

Up to now, you have finished this problem. In addition to the previously rendered cube that is represented by a very simple signed distance field, TA also provides a pre-trained NeRF of a Lego model. Run the following command:

```bash
python main.py --config-name=nerf_lego_render
```

Your rendering result will be written to `images/render_nerf.gif`, it should look like:

Please attach all of your rendering results to your submission file.

Note: We provide the NeRF training code under `train_nerf` in `main.py`. Based on your implementation of volume rendering, you can train a simple NeRF network by the following command:

```bash
python main.py --config-name=nerf_lego
```

Feel free to try it!
```