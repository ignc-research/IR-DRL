from .helpers.urdf_wall_generator import UrdfWallGenerator

class ShapeGenerator:
    def generate_shape(self, name, geometry, color=None):
      color = [.5, .5, .5, 1] if color is None else color
      return f"""
        <robot name="{name}">
          <link name="base_link">
            <visual>
              <!-- visual origin is defined w.r.t. link local coordinate system -->
              <origin xyz="0 0 0" rpy="0 0 0" />
              <geometry>
                {geometry}
              </geometry>
              <material name="white">
                <color rgba="{" ".join(str(x) for x in color)}"/>
              </material>
            </visual>
            <collision>
              <!-- collision origin is defined w.r.t. link local coordinate system -->
              <origin xyz="0 0 0" rpy="0 0 0" />
              <geometry>
                {geometry}
              </geometry>
            </collision>
          </link>
        </robot>
      """

    def generate_sphere(self, radius, color=None, **args):
      return self.generate_shape("sphere", f"<sphere radius=\"{radius}\"/>", color)
    
    def generate_box(self, scale, color=None, **args):
       return self.generate_shape("box", f"<box size=\"{scale[0]} {scale[1]} {scale[2]}\"/>", color)
    
    def generate_cylinder(self, radius, height, color=None, **args):
       return self.generate_shape("cylinder", f"<cylinder radius=\"{radius}\" length=\"{height}\"/>", color)
