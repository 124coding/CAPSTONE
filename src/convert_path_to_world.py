def grid_to_world(grid_x, grid_y, origin_x, origin_y, resolution):
    wx = origin_x + grid_x * resolution
    wy = origin_y + grid_y * resolution
    return wx, wy

def convert_path_to_world(path, origin_x, origin_y, resolution):
    return [grid_to_world(x, y, origin_x, origin_y, resolution) for (x, y) in path]