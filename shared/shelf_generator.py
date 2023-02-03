from .helpers.urdf_wall_generator import UrdfWallGenerator

class ShelfGenerator:
    def __init__(self, params) -> None:
        self.params = params

    def has_el_prev_row(self, grid, row_idx, cell_idx):
        return row_idx > 0 and grid[row_idx - 1][cell_idx] == 1

    def has_el_next_row(self, grid, row_idx, cell_idx):
        return row_idx < len(grid) - 1 and grid[row_idx + 1][cell_idx] == 1

    def has_el_prev_col(self, grid, row_idx, cell_idx):
        return cell_idx > 0 and grid[row_idx][cell_idx - 1] == 1

    def has_el_next_col(self, grid, row_idx, cell_idx):
        return cell_idx < len(grid[row_idx]) - 1 and grid[row_idx][cell_idx + 1]

    def generate(self):
        rows = self.params["rows"]
        columns = self.params["cols"]
        element_size = self.params["element_size"]
        shelf_depth = self.params["shelf_depth"]
        wall_thickness = self.params["wall_thickness"]

        xy_offset = element_size / 2
        wall_offset = wall_thickness

        urdf_wall_generator = UrdfWallGenerator()
        for row_idx in range(rows):
            for col_idx in range(columns):
                urdf_wall_generator.add_wall(element_size + wall_offset, wall_thickness, shelf_depth, wall_offset + xy_offset + col_idx * element_size, wall_offset + row_idx * element_size, shelf_depth / 2)
                urdf_wall_generator.add_wall(wall_thickness, element_size + wall_offset, shelf_depth, wall_offset + col_idx * element_size, wall_offset + xy_offset + row_idx * element_size, shelf_depth / 2)

                # closing walls
                if col_idx == columns - 1:
                    urdf_wall_generator.add_wall(wall_thickness, element_size  + wall_offset, shelf_depth, wall_offset + (col_idx + 1) * element_size, wall_offset + xy_offset + row_idx * element_size, shelf_depth / 2)
                if row_idx == rows - 1:
                    urdf_wall_generator.add_wall(element_size + wall_offset, wall_thickness, shelf_depth, wall_offset + xy_offset + col_idx * element_size, wall_offset + (row_idx + 1) * element_size, shelf_depth / 2)
        
        return urdf_wall_generator.get_urdf()