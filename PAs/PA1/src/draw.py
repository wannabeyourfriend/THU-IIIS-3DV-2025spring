# 在导入部分添加
import numpy as np
import open3d as o3d
import json
import os

class ParameterLoader:
    @staticmethod
    def load_from_file(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Init File {file_path} isn't exist, use default params")
            return ParameterLoader.get_default_params()
    @staticmethod
    def get_default_params():
        return {
            "ellipsoid": {
                "a": 1.0,
                "b": 1.0,
                "c": 0.5,
                "u_range": [-np.pi, np.pi],
                "v_range": [0, np.pi]
            },
            "curve": {
                "u_start": np.pi/4,
                "v_start": np.pi/6,
                "t_range": [-1, 1]
            },
            "visualization": {
                "window_name": "Ellipsoid with Curve",
                "background_color": [1, 1, 1],
                "curve_color": [0, 1, 0],
                "mesh_color": [0.8, 0.8, 0.8],
                "curve_thickness": 0.5,
                "show_wireframe": False,
                "show_axis_scale": True,
                "axis_scale_length": 0.1,
                "show_scale": False,
                "coordinate_size": 0.5
            }
        }
    @staticmethod
    def save_to_file(params, file_path):
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            else:
                return obj
        
        with open(file_path, 'w') as f:
            json.dump(convert_numpy(params), f, indent=4)
    

class Ellipsoid:
    def __init__(self, a=1, b=1, c=0.5, u_range=(-np.pi, np.pi), v_range=(0, np.pi)):
        self.a = a
        self.b = b
        self.c = c
        self.u_range = u_range
        self.v_range = v_range
        self.mesh = None
        self.wireframe = None

    def generate_surface(self, resolution=(50, 25)):
        u = np.linspace(self.u_range[0], self.u_range[1], resolution[0])
        v = np.linspace(self.v_range[0], self.v_range[1], resolution[1])
        u_grid, v_grid = np.meshgrid(u, v)
        x = self.a * np.cos(u_grid) * np.sin(v_grid)
        y = self.b * np.sin(u_grid) * np.sin(v_grid)
        z = self.c * np.cos(v_grid)
        vertices = []
        for i in range(len(v)):
            for j in range(len(u)):
                vertices.append([x[i, j], y[i, j], z[i, j]])
        vertices = np.array(vertices)
        triangles = []
        for i in range(len(v)-1):
            for j in range(len(u)-1):
                idx1 = i * len(u) + j
                idx2 = i * len(u) + (j + 1)
                idx3 = (i + 1) * len(u) + j
                idx4 = (i + 1) * len(u) + (j + 1)
                triangles.append([idx1, idx2, idx3])
                triangles.append([idx2, idx4, idx3])
        triangles = np.array(triangles)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        self.mesh = mesh
        self.wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        return mesh
    def get_wireframe(self):
        if self.mesh is None:
            self.generate_surface()
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
        wireframe.paint_uniform_color([0.5, 0.5, 0.5])  
        return wireframe


class Curve:
    def __init__(self, ellipsoid, u_start=np.pi/4, v_start=np.pi/6, t_range=(-1, 1)):
        self.ellipsoid = ellipsoid
        self.u_start = u_start
        self.v_start = v_start
        self.t_range = t_range
        self.curve_points = None
        self.line_set = None

    def generate_curve(self, num_points=100):
        t_values = np.linspace(self.t_range[0], self.t_range[1], num_points)
        curve_points = []

        for t in t_values:
            u_t = self.u_start + t
            v_t = self.v_start
            x_t = self.ellipsoid.a * np.cos(u_t) * np.sin(v_t)
            y_t = self.ellipsoid.b * np.sin(u_t) * np.sin(v_t)
            z_t = self.ellipsoid.c * np.cos(v_t)
            curve_points.append([x_t, y_t, z_t])

        self.curve_points = np.array(curve_points)
        return self.curve_points

    def create_line_set(self, color=[1, 0, 0], thickness=5.0):
        if self.curve_points is None:
            self.generate_curve()

        line_indices = [[i, i + 1] for i in range(len(self.curve_points) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(self.curve_points)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(line_indices))])
        
        self.line_set = line_set
        return line_set
    
    def create_thick_tube(self, radius=0.02, resolution=20, color=[1, 0, 0]):
        if self.curve_points is None:
            self.generate_curve()
        
        cylinders = []
        for i in range(len(self.curve_points) - 1):
            p1 = self.curve_points[i]
            p2 = self.curve_points[i + 1]
            
            direction = p2 - p1
            length = np.linalg.norm(direction)
            direction = direction / length
            
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius, 
                height=length, 
                resolution=resolution
            )
            
            z_axis = np.array([0, 0, 1])
            if np.allclose(direction, z_axis) or np.allclose(direction, -z_axis):
                rotation_axis = np.array([1, 0, 0])
                angle = 0 if np.allclose(direction, z_axis) else np.pi
            else:
                rotation_axis = np.cross(z_axis, direction)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.dot(z_axis, direction))
            
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
            
            cylinder.rotate(R, center=[0, 0, 0])
            cylinder.translate(p1 + direction * length / 2)
            cylinder.paint_uniform_color(color)
            cylinders.append(cylinder)
        
        if cylinders:
            combined_mesh = cylinders[0]
            for i in range(1, len(cylinders)):
                combined_mesh += cylinders[i]
            return combined_mesh
        return None


class ScaleBar:
    
    @staticmethod
    def create_scale_bar(length=1.0, position=[-1.5, -1.5, -1.0], color=[0, 0, 0]):
        line_points = [
            position,
            [position[0] + length, position[1], position[2]]
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_set.colors = o3d.utility.Vector3dVector([color])
        
        tick_points = []
        tick_lines = []
        tick_length = 0.05 * length
        
        tick_points.append(position)
        tick_points.append([position[0], position[1] - tick_length, position[2]])
        tick_lines.append([0, 1])
        
        tick_points.append([position[0] + length, position[1], position[2]])
        tick_points.append([position[0] + length, position[1] - tick_length, position[2]])
        tick_lines.append([2, 3])
        
        tick_set = o3d.geometry.LineSet()
        tick_set.points = o3d.utility.Vector3dVector(tick_points)
        tick_set.lines = o3d.utility.Vector2iVector(tick_lines)
        tick_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(tick_lines))])
        
        return line_set, tick_set


class DifferentialMap:
    def __init__(self, ellipsoid, u, v):
        self.ellipsoid = ellipsoid
        self.u = u
        self.v = v
        
    def compute_dfp(self):
        """Calculate Df_p @ (u,v)"""
        a, b, c = self.ellipsoid.a, self.ellipsoid.b, self.ellipsoid.c
        u, v = self.u, self.v
        
        dfp = np.array([
            [-a * np.sin(u) * np.sin(v), a * np.cos(u) * np.cos(v)],
            [b * np.cos(u) * np.sin(v), b * np.sin(u) * np.cos(v)],
            [0, -c * np.sin(v)]
        ])
        
        return dfp
    
    def apply_dfp_to_vector(self, v_direction):
        """Apply Df_p to a specific direction vector v"""
        dfp = self.compute_dfp()
        # Ensure v_direction is a numpy array and normalize it
        v_direction = np.array(v_direction)
        v_direction = v_direction / np.linalg.norm(v_direction)
        
        # Apply the differential map to the direction vector
        result = dfp @ v_direction
        return result
    
    def create_differential_vector(self, v_direction, scale=0.2, color=[1, 0, 0]):
        """Create a visualization of Df_p(v) for a specific direction v"""
        # Calculate the base point on the ellipsoid
        x = self.ellipsoid.a * np.cos(self.u) * np.sin(self.v)
        y = self.ellipsoid.b * np.sin(self.u) * np.sin(self.v)
        z = self.ellipsoid.c * np.cos(self.v)
        base_point = np.array([x, y, z])
        
        # Apply the differential map to the direction vector
        direction = self.apply_dfp_to_vector(v_direction)
        
        # Scale the direction vector
        direction = direction / np.linalg.norm(direction) * scale
        
        # Create cylinder for arrow shaft
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.01,
            height=np.linalg.norm(direction)
        )
        
        # Create cone for arrow head
        cone = o3d.geometry.TriangleMesh.create_cone(
            radius=0.02,
            height=0.05
        )
        
        # Calculate rotation to align with direction
        z_axis = np.array([0, 0, 1])
        if not np.allclose(direction, z_axis) and not np.allclose(direction, -z_axis):
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.dot(z_axis, direction/np.linalg.norm(direction)))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
            
            cylinder.rotate(R, center=[0, 0, 0])
            cone.rotate(R, center=[0, 0, 0])
        
        # Position the arrow
        cylinder.translate(base_point + direction/2)
        cone.translate(base_point + direction)
        
        # Set color
        cylinder.paint_uniform_color(color)
        cone.paint_uniform_color(color)
        
        return [cylinder, cone]
    
    def create_differential_vectors(self, scale=0.2, directions=None):
        """Create visualizations for multiple direction vectors"""
        if directions is None:
            # Default: standard basis vectors
            directions = [
                ([1, 0], [1, 0, 0]),  # u direction, red color
                ([0, 1], [0, 0, 1])   # v direction, blue color
            ]
        
        arrows = []
        for v_dir, color in directions:
            arrow_parts = self.create_differential_vector(v_dir, scale, color)
            arrows.extend(arrow_parts)
        
        return arrows

class Visualizer:
    def __init__(self, params=None):
        if params is None:
            params = ParameterLoader.get_default_params()
        
        self.params = params
        self.vis = o3d.visualization.Visualizer()
        
        ellipsoid_params = params["ellipsoid"]
        self.ellipsoid = Ellipsoid(
            a=ellipsoid_params["a"],
            b=ellipsoid_params["b"],
            c=ellipsoid_params["c"],
            u_range=tuple(ellipsoid_params["u_range"]),
            v_range=tuple(ellipsoid_params["v_range"])
        )
        
        curve_params = params["curve"]
        self.curve = Curve(
            ellipsoid=self.ellipsoid,
            u_start=curve_params["u_start"],
            v_start=curve_params["v_start"],
            t_range=tuple(curve_params["t_range"])
        )
        
        self.vis_params = params["visualization"]

    def _create_axis_scale(self, axis_length=1.0, num_ticks=5):
        scales = []
        scale_length = self.vis_params.get("axis_scale_length", 0.1)
        for axis in range(3):
            for i in range(num_ticks):
                pos = i * axis_length / (num_ticks - 1)
                if axis == 0:  
                    p1 = [pos, 0, 0]
                    p2 = [pos, -scale_length, 0]
                elif axis == 1: 
                    p1 = [0, pos, 0]
                    p2 = [-scale_length, pos, 0]
                else: 
                    p1 = [0, 0, pos]
                    p2 = [-scale_length, 0, pos]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector([p1, p2])
                line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
                line_set.paint_uniform_color([0, 0, 0])
                scales.append(line_set)
        return scales
    
    
    def initialize(self):
        self.vis.create_window(window_name=self.vis_params["window_name"])
        
        mesh = self.ellipsoid.generate_surface()
        mesh.paint_uniform_color(self.vis_params["mesh_color"])
        wireframe = self.ellipsoid.get_wireframe()
        thick_curve = self.curve.create_thick_tube(
            radius=self.vis_params["curve_thickness"] / 100,
            color=self.vis_params["curve_color"]
        )
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.vis_params["coordinate_size"], 
            origin=[0, 0, 0]
        )
        self.vis.add_geometry(mesh)
        self.vis.add_geometry(wireframe)
        self.vis.add_geometry(thick_curve)
        self.vis.add_geometry(coordinate_frame)
        self.add_differential_map(directions=[
    ([1, 0], [1, 0, 0]),     # u direction in red
    ([0, 1], [0, 0, 1]),     # v direction in blue
    ([1, 1], [0, 1, 0]),     # diagonal direction in green
    ([2, -1], [1, 0.5, 0])   # another direction in orange
])
        if self.vis_params.get("show_axis_scale", True):
            axis_scales = self._create_axis_scale(
                axis_length=self.vis_params["coordinate_size"]
            )
            for scale in axis_scales:
                self.vis.add_geometry(scale)
                
        if self.vis_params["show_scale"]:
            scale_length = max(self.ellipsoid.a, self.ellipsoid.b, self.ellipsoid.c)
            scale_bar, scale_ticks = ScaleBar.create_scale_bar(
                length=scale_length,
                position=[-scale_length*1.5, -scale_length*1.5, -scale_length]
            )
            self.vis.add_geometry(scale_bar)
            self.vis.add_geometry(scale_ticks)
        opt = self.vis.get_render_option()
        opt.background_color = np.array(self.vis_params["background_color"])
        opt.point_size = self.vis_params["curve_thickness"]
        opt.mesh_show_wireframe = self.vis_params["show_wireframe"]


    def run(self):
        self.vis.run()
        self.vis.destroy_window()
    
    def capture_image(self, filename="ellipsoid_with_curve.png"):
        self.vis.capture_screen_image(filename, True)
        print(f"Image has been saved as {filename}")
        
    def add_differential_map(self, u=None, v=None, directions=None):
        """Add visualization of differential map
        
        Args:
            u, v: Parameters for the point on the ellipsoid
            directions: List of tuples (v_direction, color) where:
                           - v_direction is a vector in the tangent space
                           - color is the RGB color for the arrow
        """
        if u is None:
            u = self.params["curve"]["u_start"]
        if v is None:
            v = self.params["curve"]["v_start"]
            
        diff_map = DifferentialMap(self.ellipsoid, u, v)
        arrows = diff_map.create_differential_vectors(directions=directions)
        
        for arrow in arrows:
            self.vis.add_geometry(arrow)

if __name__ == "__main__":
    config_file = "e:\\project\\3DV\\PAs\\Homework 1\\ellipsoid_config.json"
    
    if not os.path.exists(config_file):
        default_params = ParameterLoader.get_default_params()
        ParameterLoader.save_to_file(default_params, config_file)
        print(f"Has init default params file {config_file}")
        params = default_params
    else:
        params = ParameterLoader.load_from_file(config_file)
    visualizer = Visualizer(params)
    visualizer.initialize()
    visualizer.run()