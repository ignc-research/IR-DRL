class UrdfWallGenerator:
    def __init__(self, color=[.5, .5, .5, 1]) -> None:
        self.segments = []
        self.color = color

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
                <inertial>
                    <origin rpy="0 0 0" xyz="0 0 0"/>
                    <mass value="0"/>
                    <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                </inertial> 
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
                            <color rgba="{" ".join(str(x) for x in self.color)}"/>
                        </material>
                    </visual>
                    <collision>
                        <origin rpy="0 0 0" xyz="{segment["pos_x"]} {segment["pos_y"]} {segment["pos_z"]}"/>
                        <geometry>
                            <box size="{segment["w"]} {segment["h"]} {segment["d"]}"/>
                        </geometry>
                    </collision>
                    <inertial>
                        <origin rpy="0 0 0" xyz="{segment["w"]/2} {segment["h"]/2} {segment["d"]/2}"/>
                        <mass value="1"/>
                        <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                    </inertial>
                </link>
                <joint name="joint_{i}" type="fixed">
                    <parent link="base_link"/>
                    <child link="link_{i}"/>
                    <origin xyz="0 0 0" rpy="0 0 0"/>
                </joint>
                """
        output += "</robot>"

        return output