import pybullet as p

class UrdfWallGenerator:
    def __init__(self) -> None:
        self.segments = []

    def add_wall(self, w, h, d, pos_x, pos_y, pos_z):
        self.segments.append({
            "w": w,
            "h": h,
            "d": d,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "pos_z": pos_z
        })

    def get_urdf(self):
        output = f"""
        <robot name="maze">
            <link name="base_link">
               
            </link>
        """

        
        for i, segment in enumerate(self.segments):
            output += f"""
                <link name="link_{i}">
                    <visual>
                        <origin rpy="0 0 0" xyz="{segment["pos_x"]} {segment["pos_y"]} {segment["pos_z"]}"/>
                        <geometry>
                            <box size="{segment["w"]} {segment["h"]} {segment["d"]}"/>
                        </geometry>
                        <material name="white">
                            <color rgba="1 1 1 1"/>
                        </material>
                    </visual>
                    <collision>
                        <origin rpy="0 0 0" xyz="{segment["pos_x"]} {segment["pos_y"]} {segment["pos_z"]}"/>
                        <geometry>
                            <box size="{segment["w"]} {segment["h"]} {segment["d"]}"/>
                        </geometry>
                    </collision>
                </link>
                <joint name="joint_{i}" type="fixed">
                    <parent link="base_link"/>
                    <child link="link_{i}"/>
                    <origin xyz="0 0 0" rpy="0 0 0"/>
                </joint>
                """
        output += "</robot>"

        return output