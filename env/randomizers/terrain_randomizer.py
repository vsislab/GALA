"""Generates a random terrain when resetting aliengo gym environment"""
import enum
import itertools
from math import pi, tan, sqrt, ceil
import numpy as np
import noise
from os.path import dirname, join
from typing import Dict

from .base_randomizer import BaseRandomizer

_GRID_LENGTH = 15
_GRID_WIDTH = 10
_MAX_SAMPLE_SIZE = 30
_MIN_BLOCK_DISTANCE = 0.5
_MAX_BLOCK_LENGTH = _MIN_BLOCK_DISTANCE
_MIN_BLOCK_LENGTH = _MAX_BLOCK_LENGTH / 2
_MAX_BLOCK_HEIGHT = 0.05
_MIN_BLOCK_HEIGHT = _MAX_BLOCK_HEIGHT / 2


class PoissonDisc2D(object):
    """Generates 2D points using Poisson disk sampling method.

    Implements the algorithm described in:
      http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
    Unlike the uniform sampling method that creates small clusters of points,
    Poisson disk method enforces the minimum distance between points and is more
    suitable for generating a spatial distribution of non-overlapping objects.
    """

    def __init__(self, grid_length, grid_width, min_radius, max_sample_size):
        """Initializes the algorithm.

        Args:
          grid_length: The length of the bounding square in which points are
            sampled.
          grid_width: The width of the bounding square in which points are
            sampled.
          min_radius: The minimum distance between any pair of points.
          max_sample_size: The maximum number of sample points around a active site.
            See details in the algorithm description.
        """
        self._cell_length = min_radius / sqrt(2)
        self._grid_length = grid_length
        self._grid_width = grid_width
        self._grid_size_x = int(grid_length / self._cell_length) + 1
        self._grid_size_y = int(grid_width / self._cell_length) + 1
        self._min_radius = min_radius
        self._max_sample_size = max_sample_size

        # Flattern the 2D grid as an 1D array. The grid is used for fast nearest
        # point searching.
        self._grid = [None] * self._grid_size_x * self._grid_size_y

        # Generate the first sample point and set it as an active site.
        first_sample = np.array(np.random.random_sample(2)) * [grid_length, grid_width]
        self._active_list = [first_sample]

        # Also store the sample point in the grid.
        self._grid[self._point_to_index_1d(first_sample)] = first_sample

    def _point_to_index_1d(self, point):
        """Computes the index of a point in the grid array.

        Args:
          point: A 2D point described by its coordinates (x, y).

        Returns:
          The index of the point within the self._grid array.
        """
        return self._index_2d_to_1d(self._point_to_index_2d(point))

    def _point_to_index_2d(self, point):
        """Computes the 2D index (aka cell ID) of a point in the grid.

        Args:
          point: A 2D point (list) described by its coordinates (x, y).

        Returns:
          x_index: The x index of the cell the point belongs to.
          y_index: The y index of the cell the point belongs to.
        """
        x_index = int(point[0] / self._cell_length)
        y_index = int(point[1] / self._cell_length)
        return x_index, y_index

    def _index_2d_to_1d(self, index2d):
        """Converts the 2D index to the 1D position in the grid array.

        Args:
          index2d: The 2D index of a point (aka the cell ID) in the grid.

        Returns:
          The 1D position of the cell within the self._grid array.
        """
        return index2d[0] + index2d[1] * self._grid_size_x

    def _is_in_grid(self, point):
        """Checks if the point is inside the grid boundary.

        Args:
          point: A 2D point (list) described by its coordinates (x, y).

        Returns:
          Whether the point is inside the grid.
        """
        return (0 <= point[0] < self._grid_length) and (0 <= point[1] < self._grid_width)

    def _is_in_range(self, index2d):
        """Checks if the cell ID is within the grid.

        Args:
          index2d: The 2D index of a point (aka the cell ID) in the grid.

        Returns:
          Whether the cell (2D index) is inside the grid.
        """

        return (0 <= index2d[0] < self._grid_size_x) and (0 <= index2d[1] < self._grid_size_y)

    def _is_close_to_existing_points(self, point):
        """Checks if the point is close to any already sampled (and stored) points.

        Args:
          point: A 2D point (list) described by its coordinates (x, y).

        Returns:
          True iff the distance of the point to any existing points is smaller than
          the min_radius
        """
        px, py = self._point_to_index_2d(point)
        # Now we can check nearby cells for existing points
        for neighbor_cell in itertools.product(range(px - 1, px + 2), range(py - 1, py + 2)):

            if not self._is_in_range(neighbor_cell):
                continue

            maybe_a_point = self._grid[self._index_2d_to_1d(neighbor_cell)]
            if maybe_a_point is not None and np.linalg.norm(maybe_a_point - point) < self._min_radius:
                return True

        return False

    def sample(self):
        """Samples new points around some existing point.

        Removes the sampling base point and also stores the new jksampled points if
        they are far enough from all existing points.
        """
        active_point = self._active_list.pop()
        for _ in range(self._max_sample_size):
            # Generate random points near the current active_point between the radius
            random_radius = np.random.uniform(self._min_radius, 2 * self._min_radius)
            random_angle = np.random.uniform(0, 2 * pi)

            # The sampled 2D points near the active point
            sample = random_radius * np.array([np.cos(random_angle),
                                               np.sin(random_angle)]) + active_point

            if not self._is_in_grid(sample):
                continue

            if self._is_close_to_existing_points(sample):
                continue

            self._active_list.append(sample)
            self._grid[self._point_to_index_1d(sample)] = sample

    def generate(self):
        """Generates the Poisson disc distribution of 2D points.

        Although the while loop looks scary, the algorithm is in fact O(N), where N
        is the number of cells within the grid. When we sample around a base point
        (in some base cell), new points will not be pushed into the base cell
        because of the minimum distance constraint. Once the current base point is
        removed, all future searches cannot start from within the same base cell.

        Returns:
          All sampled points. The points are inside the quare [0, grid_length] x [0,
          grid_width]
        """

        while self._active_list:
            self.sample()

        all_sites = []
        for p in self._grid:
            if p is not None:
                all_sites.append(p)

        return all_sites


class TerrainType(enum.Enum):
    """The randomized terrain types we can use in the gym env."""
    Flat = 0
    Box = 1
    Random = 2
    Slope = 3
    Hill = 4
    Step = 5
    Stair = 6
    Hole = 7
    # BLOCK = 2
    # MESH = 4
    # HEIGHTFIELD = 5


class TerrainParam:
    def __init__(self, type: TerrainType = TerrainType.Flat, size: tuple = (30., 30.), **specials):
        self.type = type
        self.size = size
        self.specials = specials
        self.pixel_size = None

    @classmethod
    def Template(cls, type: str = 'Flat', **specials):
        if type == 'Flat':
            return TerrainParam(type=TerrainType.Flat, size=(50., 50.))
        elif type == 'Box':
            return TerrainParam(type=TerrainType.Box, size=(30., 15.), **specials)
        elif type == 'Random':
            return TerrainParam(type=TerrainType.Random, size=(30., 30.), **specials)
        elif type == 'Slope':
            return TerrainParam(type=TerrainType.Slope, size=(30., 30.), **specials)
        elif type == 'Hill':
            return TerrainParam(type=TerrainType.Hill, size=(40., 40.), **specials)
        elif type == 'Step':
            return TerrainParam(type=TerrainType.Step, size=(15., 15.), **specials)
        elif type == 'Stair':
            return TerrainParam(type=TerrainType.Stair, size=(15., 15.), **specials)
        else:
            raise NameError(f'Not exist terrain {type}')

    def __repr__(self):
        return str({
            'type': self.type.name,
            'size': self.size,
            'specials': self.specials,
            'pixel_size': self.pixel_size
        })


class TerrainInstance:
    def __init__(self, client, id: int, position: tuple = (0., 0., 0.), param: TerrainParam = None, height_field: np.ndarray = None):
        self.client = client
        self.id = id
        self._position = position
        self.param = param
        self.height_field = height_field
        info = client.getDynamicsInfo(id, -1)
        self._friction, self._restitution = info[1], info[5]
        if height_field is not None:
            x_mid, y_mid = int(height_field.shape[0] / 2), int(height_field.shape[1] / 2)
            width = ceil(0.3 / self.param.pixel_size)
            x_slice, y_slice = slice(x_mid - width, x_mid + width), slice(y_mid - width, y_mid + width)
            x, y, z = position
            self.place_position = np.array([x, y, height_field[x_slice, y_slice].max()])
            self.start_position = np.array([x - param.size[0] / 2, y - param.size[1] / 2, height_field[0, 0]])
        else:
            self.place_position = position
            self.start_position = None
        self._raise_error = False

    def clip(self, position, threshold: float = 1.5):
        if self.param is not None:
            x_size, y_size = self.param.size
            bound = np.array([x_size / 2, y_size / 2]) - threshold
            xy_offset = np.clip((position - self.position)[:2], -bound, bound)
            position[:2] = xy_offset + self.position[:2]
        return position

    def in_terrain(self, position, threshold: float = 1.5):
        if self.param is not None:
            return np.all(np.abs(position - self.position)[:2] <= np.array(self.param.size) / 2 - threshold)
        return True

    def get_height(self, xy):
        if self.height_field is None:
            return self.position[-1]
        pixel_size = self.param.pixel_size
        xy = (xy - self.start_position[:2]) / pixel_size
        xy = tuple(np.round(xy).astype(int))
        z = self.height_field[xy]
        # try:
        #     z = self.height_field[xy]
        # except IndexError as e:
        #     print('terrain position:', self.position)
        #     print('terrain param:', self.param)
        #     print('xy:', xy)
        #     raise e
        return z

    @property
    def position(self):
        return self._position

    # @position.setter
    # def position(self, v):
    #     v = np.asarray(v)
    #     self._position = v
    #     _, orientation = self.client.getBasePositionAndOrientation(self.id)
    #     self.client.resetBasePositionAndOrientation(self.id, v, orientation)

    @property
    def friction(self):
        return self._friction

    @friction.setter
    def friction(self, v):
        self._friction = v
        self.client.changeDynamics(self.id, -1, lateralFriction=v)

    @property
    def restitution(self):
        return self._restitution

    @restitution.setter
    def restitution(self, v):
        self._restitution = v
        self.client.changeDynamics(self.id, -1, restitution=v)


class TerrainRandomizer(BaseRandomizer):
    """Generate an uneven terrain in the gym env."""
    assets_dir = join(dirname(__file__), '../assets')

    def __init__(self, terrain_param: TerrainParam = TerrainParam()):
        super(TerrainRandomizer, self).__init__()
        self.terrain_randomizer_dict = {
            TerrainType.Flat: self._generate_flat_terrain,
            TerrainType.Random: self._generate_random_terrain,
            TerrainType.Box: self._generate_box_terrain,
            TerrainType.Slope: self._generate_slope_terrain,
            TerrainType.Hill: self._generate_hill_terrain,
            TerrainType.Step: self._generate_step_terrain,
            TerrainType.Stair: self._generate_stair_terrain,
            TerrainType.Hole: self._generate_hole_terrain,
        }
        self._terrain_param = terrain_param
        self.terrain_param_updated = True
        self.terrain: TerrainInstance
        self.pixel_size = None

    def _randomize_step(self, env):
        pass

    def _randomize_env(self, env):
        """Choose the terrain for the current env."""
        if self.terrain_param_updated:
            self.terrain_param_updated = False
            terrain_param = self._terrain_param
            height_field = self.terrain_randomizer_dict[terrain_param.type](size=terrain_param.size, **terrain_param.specials)
            # height_field -= (np.max(height_field) + np.min(height_field)) / 2.
            terrain_param.pixel_size = self.pixel_size
            terrain_param.size = tuple(terrain_param.pixel_size * np.asarray(height_field.shape))
            terrain_position = (0., terrain_param.size[1] / 2, (np.max(height_field) + np.min(height_field)) / 2.)
            terrain_id = self.create_terrain_by_height_field(env, height_field, position=terrain_position)
            self.terrain = TerrainInstance(env.client, terrain_id, position=terrain_position, param=terrain_param, height_field=height_field)
            if self.terrain.param.type == TerrainType.Box:
                s1, s2 = self.terrain.param.size
                box_height = self.terrain.param.specials['box_height']
                shape_id = env.client.createCollisionShape(env.client.GEOM_BOX, halfExtents=np.asarray([s1 / 2, s2, 0.1]) / 2)
                env.client.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=shape_id,
                                           basePosition=self.terrain.place_position + np.array([-s1 / 4 + 0.01 * np.sign(box_height), 0, -max(box_height, 0) - 0.051]),
                                           baseOrientation=[0.0, 0.0, 0.0, 1])
                env.client.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=shape_id,
                                           basePosition=self.terrain.place_position + np.array([s1 / 4 + 0.01 * np.sign(box_height), 0, min(box_height, 0) - 0.051]),
                                           baseOrientation=[0.0, 0.0, 0.0, 1])
            elif self.terrain.param.type == TerrainType.Flat:
                s1, s2 = self.terrain.param.size
                shape_id = env.client.createCollisionShape(env.client.GEOM_BOX, halfExtents=np.asarray([s1, s2, 0.1]) / 2)
                box_id = env.client.createMultiBody(baseMass=0,
                                                    baseCollisionShapeIndex=shape_id,
                                                    basePosition=self.terrain.place_position - np.asarray([0, 0, 0.051]),
                                                    baseOrientation=[0.0, 0.0, 0.0, 1])

    def _generate_flat_terrain(self, size):
        self.pixel_size = 0.1
        x_sample, y_sample = int(size[0] / self.pixel_size), int(size[1] / self.pixel_size)
        height_field = np.zeros((x_sample, y_sample))
        return height_field

    def _generate_random_terrain(self, size, step_height=0.05):
        self.pixel_size = 0.1
        x_sample, y_sample = int(size[0] / self.pixel_size), int(size[1] / self.pixel_size)
        height_field = np.zeros((x_sample, y_sample))
        x_mid, y_mid = int(x_sample / 2), int(y_sample / 2)
        for i in range(x_mid):
            for j in range(y_mid):
                height_field[2 * i:2 * (i + 1), 2 * j:2 * (j + 1)] = np.random.uniform(0, step_height)
        # self._flatten_placement_position(height_field)
        return height_field

    def _generate_box_terrain(self, size, box_height=0.2):
        self.pixel_size = 0.02
        x_sample, y_sample = int(size[0] / self.pixel_size), int(size[1] / self.pixel_size)
        height_field = np.zeros((x_sample, y_sample))
        # mid0, mid1 = int(0.5 * x_sample - 0.5 * step_width / self.pixel_size), int(0.5 * x_sample + 0.5 * step_width / self.pixel_size)
        if box_height >= 0:
            height_field[x_sample // 2:] = box_height
        else:
            height_field[:x_sample // 2] = -box_height
        return height_field

    def _generate_slope_terrain(self, size, slope: float = 15):
        """
        param slope: degree
        """
        self.pixel_size = 0.1
        # slope = (1, -1)[np.random.randint(0, 2)] * slope
        step_start = 0
        step_height = tan(slope / 180 * pi) * self.pixel_size
        x_sample, y_sample = int(size[0] / self.pixel_size), int(size[1] / self.pixel_size)
        mid0 = int(2 / self.pixel_size)
        mid1 = int(0.5 * x_sample - 0.4 / self.pixel_size)
        mid2 = int(0.5 * x_sample + 0.4 / self.pixel_size)
        mid3 = int(x_sample - 2 / self.pixel_size)
        assert mid0 < mid1 < mid2 < mid3
        height_field = np.zeros((x_sample, y_sample))
        for i in range(mid0, mid1):
            height_field[i] = step_start
            step_start += step_height
        height_field[mid1:mid2] = step_start
        for i in range(mid2, mid3):
            height_field[i] = step_start
            step_start += step_height
        height_field[mid3:x_sample] = step_start
        return height_field

    def _generate_hill_terrain(self, size, frequency=0.3, amplitude=0., roughness=0.1):
        self.pixel_size = 0.1
        x_sample, y_sample = int(size[0] / self.pixel_size), int(size[1] / self.pixel_size)
        height_field = np.zeros((x_sample, y_sample))
        for i in range(x_sample):
            for j in range(y_sample):
                height_field[i, j] = noise.pnoise2(i / x_sample, j / y_sample,
                                                   octaves=5,
                                                   persistence=frequency,
                                                   lacunarity=3)
        height_field = (height_field + 1.) * amplitude + np.random.uniform(0, 1, height_field.shape) * roughness
        # self._flatten_placement_position(height_field)
        return height_field

    def _generate_step_terrain(self, size, step_width=0.3, step_height=0.05):
        self.pixel_size = 0.02
        x_sample, y_sample = int(size[0] / self.pixel_size), int(size[1] / self.pixel_size)
        height_field = np.zeros((x_sample, y_sample))
        grid_width = int(step_width / self.pixel_size)
        for i in np.arange(0, x_sample, grid_width):
            for j in np.arange(0, y_sample, grid_width):
                height_field[i:i + grid_width, j:j + grid_width] = np.random.rand() * step_height
        # self._flatten_placement_position(height_field)
        return height_field

    def _generate_stair_terrain(self, size, step_width=0.2, step_height=0.05):
        self.pixel_size = 0.02
        x_sample, y_sample = int(size[0] / self.pixel_size), int(size[1] / self.pixel_size)
        height_field = np.zeros((x_sample, y_sample))
        step_start = 0
        N = int(step_width / self.pixel_size)
        mid0, mid1 = int(0.5 * x_sample - 0.4 / self.pixel_size), int(0.5 * x_sample + 0.4 / self.pixel_size)
        for i in np.arange(mid0, 0, -N)[::-1]:
            height_field[max(i - N, 0):i] = step_start
            # if np.random.choice((True, False)): height_field[i] = step_start - 0.01
            step_start += step_height
        height_field[mid0:mid1] = step_start
        for i in np.arange(mid1, x_sample, N):
            step_start += step_height
            height_field[i:min(i + N, x_sample)] = step_start
            # if np.random.choice((True, False)): height_field[i] = step_start - 0.01
        return height_field

    def _generate_hole_terrain(self, size):
        self.pixel_size = 0.02
        x_sample, y_sample = int(size[0] / self.pixel_size), int(size[1] / self.pixel_size)
        height_field = np.zeros((x_sample, y_sample))
        hole_positions = PoissonDisc2D(*size, 0.4, 4).generate()
        hole_coords = np.floor(np.asarray(hole_positions) / self.pixel_size).astype(np.int)
        for coord in hole_coords:
            x, y = coord
            a, b, c, d = [np.random.randint(2, 4) for _ in range(4)]
            height_field[max(0, x - a):min(x_sample, x + b), max(0, y - c):min(y_sample, y + d)] = -1.
        # self._flatten_placement_position(height_field)
        return height_field

    def _flatten_placement_position(self, height_field):
        x_sample, y_sample = height_field.shape
        x_mid, y_mid = int(x_sample / 2), int(y_sample / 2)
        width = ceil(0.3 / self.pixel_size)
        x_slice, y_slice = slice(x_mid - width, x_mid + width + 1), slice(y_mid - width, y_mid + width + 1)
        height = np.mean(height_field[x_slice, y_slice])
        height_field[x_slice, y_slice] = height

    def create_terrain_by_height_field(self, env, height_field, position=(0., 0., 0.)):
        terrain_shape = env.client.createCollisionShape(
            shapeType=env.client.GEOM_HEIGHTFIELD,
            # flags=env.client.GEOM_CONCAVE_INTERNAL_EDGE,
            meshScale=[self.pixel_size, self.pixel_size, 1],
            heightfieldTextureScaling=(height_field.shape[0] - 1) / 2,
            heightfieldData=height_field.transpose().flatten(),  # by column
            numHeightfieldRows=height_field.shape[0],
            numHeightfieldColumns=height_field.shape[1])
        terrain_id = env.client.createMultiBody(baseMass=0,
                                                baseCollisionShapeIndex=terrain_shape,
                                                basePosition=position)
        texture_id = env.client.loadTexture("%s/terrain/grey.png" % self.assets_dir)
        env.client.changeVisualShape(terrain_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id)  # todo
        env.client.changeVisualShape(terrain_id, -1, rgbaColor=[2.9, 2.9, 2.9, 1])  # todo
        return terrain_id

    @property
    def terrain_param(self):
        return self._terrain_param

    @terrain_param.setter
    def terrain_param(self, terrain_specials: Dict):
        self.terrain_param_updated = True
        for k, v in terrain_specials.items():
            self._terrain_param.specials[k] = v


if __name__ == '__main__':
    poisson_disc = PoissonDisc2D(_GRID_LENGTH, _GRID_WIDTH, _MIN_BLOCK_DISTANCE, _MAX_SAMPLE_SIZE)
    block_centers = poisson_disc.generate()
    pass
