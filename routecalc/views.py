from .models import Line, Point, Path, Route
from rest_framework import viewsets
from .serializers import LineSerializer, PointSerializer
from .serializers import PathSerializer, RouteSerializer
import numpy
from scipy.spatial import KDTree
from django.db.models import QuerySet


class PathTrace:
    def __init__(
        self,
        paths: list[Path] = [],
        length: float = 0,
        transfers: int = 0
    ):
        self.paths = paths
        self.length = length
        self.transfers = transfers

    def __repr__(self):
        return f"{{{self.paths}, {self.length}, {self.transfers} }}"


def ClosestPoint(points: QuerySet[Point], point: Point) -> Point:
    nparray = numpy.array(points)
    tree = KDTree(nparray)
    _, index = tree.query(point)
    return points[int(index)]


def DistanceBetween(point_a: Point, point_b: Point) -> float:
    return numpy.linalg.norm(point_a.__array__() - point_b.__array__())


def LengthOfPath(path: list[Path]) -> float:
    length: float = 0
    for p in path:
        length += DistanceBetween(p.origin, p.destination)
    return length


def TransfersOfPath(path: list[Path]) -> int:
    transfers: int = 0
    currentLine: Line | None = None
    for p in path:
        if currentLine is not None and currentLine is not p.line:
            transfers += 1
        currentLine = p.line
    return transfers


def Step(
    origin: Point,
    destination: Point,
    traversed: list[Path],
    currentLine: Line | None,
    last: Point | None,
    pathsTaken: list[Path]
):
    if origin == destination:
        pathsTaken.append(
            PathTrace(
                traversed.copy(),
                LengthOfPath(traversed),
                TransfersOfPath(traversed)
                )
            )
        return
    for path in origin.paths.all():
        if path in traversed:
            return
        dest: Point = path.destination
        traversed.append(path)
        Step(dest, destination, traversed, path.line, origin, pathsTaken)
        traversed.pop()


def CalculatePaths(origin: Point, destination: Point):
    lastOrigin = origin
    lastDestination = destination
    pathsTaken: list[PathTrace] = []
    if (lastOrigin is not None and lastDestination is not None):
        Step(lastOrigin, lastDestination, [], None, None, pathsTaken)
    return pathsTaken


class LineViewSet(viewsets.ModelViewSet):
    queryset = Line.objects.all()
    serializer_class = LineSerializer


class PointViewSet(viewsets.ModelViewSet):
    queryset = Point.objects.all()
    serializer_class = PointSerializer


class PathViewSet(viewsets.ModelViewSet):
    queryset = Path.objects.all()
    serializer_class = PathSerializer


class RouteViewSet(viewsets.ModelViewSet):
    queryset = Route.objects.all()
    serializer_class = RouteSerializer
