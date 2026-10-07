"""Reading wall positions out of the maze text files."""

#: Side of a maze tile, in pixels. Entity positions are given in pixels, so
#: dividing by this value yields grid coordinates.
CELL_SIZE = 16


def is_wall(cell):
    """True when a maze cell is a wall tile.

    Walls are encoded as digits in the maze files; dots and letters mark
    corridors, pellet spots and portals.
    """
    return cell.isdigit()


def get_wall_pixel_positions(maze_path):
    """Return the top-left pixel coordinates of every wall tile of a maze."""
    wall_pixels = []
    with open(maze_path, "r") as f:
        lines = f.readlines()

    for y, line in enumerate(lines):
        tokens = line.strip().split()
        for x, cell in enumerate(tokens):
            if is_wall(cell):
                wall_pixels.append((x * CELL_SIZE, y * CELL_SIZE))
    return wall_pixels
