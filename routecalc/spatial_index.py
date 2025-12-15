import numpy as np
from scipy.spatial import KDTree
from pyproj import Transformer
from .models import Point


class PointSpatialIndex:
    _instance = None
    _tree = None
    _point_ids = []
    _transformer = Transformer.from_crs("EPSG:4326", "EPSG:32720",
                                        always_xy=True)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PointSpatialIndex, cls).__new__(cls)
            cls._instance._build_index()
        return cls._instance

    def _build_index(self):
        points_data = list(Point.objects.values('id', 'x_coord', 'y_coord'))
        if not points_data:
            self._tree = None
            self._point_ids = []
            return
        ids = [p['id'] for p in points_data]
        xs = [p['x_coord'] for p in points_data]
        ys = [p['y_coord'] for p in points_data]
        projected_x, projected_y = self._transformer.transform(xs, ys)
        coords = np.column_stack((projected_x, projected_y))
        self._tree = KDTree(coords)
        self._point_ids = np.array(ids)
        print(f"Index built with {len(ids)} points.")

    def query_radius(self, target_point_obj, radius_meters=50.0):
        if self._tree is None:
            return []
        target_x, target_y = self._transformer.transform(
            target_point_obj.x_coord,
            target_point_obj.y_coord
        )
        indices = self._tree.query_ball_point([target_x, target_y],
                                              radius_meters)
        found_ids = self._point_ids[indices].tolist()
        return found_ids

    def query(self, target_point_obj):
        if self._tree is None:
            return []
        target_x, target_y = self._transformer.transform(
            target_point_obj.x_coord,
            target_point_obj.y_coord
        )
        _, index = self._tree.query([target_x, target_y], k=1)
        found_id = self._point_ids[index]
        return found_id
