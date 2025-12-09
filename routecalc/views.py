from django.shortcuts import get_object_or_404
from rest_framework.response import Response
from .models import Line, Point, Step, Route
from rest_framework import viewsets
from rest_framework.views import APIView
from .serializers import LineSerializer, PointSerializer
from .serializers import StepSerializer, RouteSerializer, StepTraceSerializer
import numpy
from scipy.spatial import KDTree
from django.db.models import QuerySet


class StepTrace:
    def __init__(
        self,
        steps: list[Step] = [],
        length: float = 0,
        transfers: int = 0
    ):
        self.steps = steps
        self.length = length
        self.transfers = transfers

    def __repr__(self):
        return f"{{{self.steps}, {self.length}, {self.transfers} }}"


def ClosestPoint(points: QuerySet[Point], point: Point) -> Point:
    nparray = numpy.array(points)
    tree = KDTree(nparray)
    _, index = tree.query(point)
    return points[int(index)]


def DistanceBetween(point_a: Point, point_b: Point) -> float:
    return numpy.linalg.norm(point_a.__array__() - point_b.__array__())


def LengthOfPath(path: list[Step]) -> float:
    length: float = 0
    for p in path:
        length += DistanceBetween(p.point, p.next.point)
    return length


def TransfersOfPath(path: list[Step]) -> int:
    transfers: int = 0
    currentLine: Line | None = None
    for p in path:
        if currentLine is not None and currentLine is not p.route.line:
            transfers += 1
        currentLine = p.route.line
    return transfers


def TakeStep(
    origin: Point,
    destination: Point,
    traversed: list[Step],
    stepsTaken: list[Step]
):
    MAX_ALLOWED_TRANSFERS = 10
    MAX_ACCEPTABLE_DISTANCE = 50
    if stepsTaken.__len__() > 0:
        return
    if TransfersOfPath(traversed) > MAX_ALLOWED_TRANSFERS:
        return
    if origin is None:
        return
    if origin == destination:
        stepsTaken.append(
            StepTrace(
                traversed.copy(),
                LengthOfPath(traversed),
                TransfersOfPath(traversed)
            )
        )
        return
    for step in origin.steps.all():
        if step in traversed:
            return
        dest: Point = step.next.point if step.next is not None else None
        traversed.append(step)
        TakeStep(dest, destination, traversed, stepsTaken)
        traversed.pop()


def CalculatePaths(origin: Point, destination: Point):
    lastOrigin = origin
    lastDestination = destination
    stepsTaken: list[StepTrace] = []
    if (lastOrigin is not None and lastDestination is not None):
        TakeStep(lastOrigin, lastDestination, [], stepsTaken)
    return stepsTaken


def RenderRoute(route: Route):
    steps = []
    currentStep: Step = route.first
    while currentStep.next is not None:
        steps.append(currentStep.point)
        currentStep = currentStep.next
    return steps


def IsWithinRange(route: Route, point: Point, radius: float):
    steps = route.step_set.all()
    points = [step.point for step in steps]
    return DistanceBetween(point, ClosestPoint(points, point)) <= radius


class LineViewSet(viewsets.ModelViewSet):
    queryset = Line.objects.all()
    serializer_class = LineSerializer


class PointViewSet(viewsets.ModelViewSet):
    queryset = Point.objects.all()
    serializer_class = PointSerializer


class StepViewSet(viewsets.ModelViewSet):
    queryset = Step.objects.all()
    serializer_class = StepSerializer


class RouteViewSet(viewsets.ModelViewSet):
    queryset = Route.objects.all()
    serializer_class = RouteSerializer


class LineRoutesView(APIView):
    def get(self, request, line_id, *args, **kwargs):
        line = get_object_or_404(Line, id=line_id)
        renderedRoutes = []
        routes = Route.objects.filter(line=line).all()
        for route in routes:
            jsonroute = PointSerializer(RenderRoute(route), many=True)
            renderedRoutes.append({
                "id": route.id,
                "lineName": route.line.name,
                "isReturn": route.isReturn,
                "distance": route.distance,
                "time": route.time,
                "path": jsonroute.data
            })
        return Response(renderedRoutes)


class CloseRoutesView(APIView):
    def get(self, request, x_coord, y_coord, radius, *args, **kwargs):
        x_coord = float(x_coord.replace(',', '.'))
        y_coord = float(y_coord.replace(',', '.'))
        radius = float(radius.replace(',', '.'))
        renderedRoutes = []
        routes = Route.objects.all()
        for route in routes:
            if not IsWithinRange(
                route,
                Point(x_coord=x_coord, y_coord=y_coord),
                radius
            ):
                continue
            jsonroute = PointSerializer(RenderRoute(route), many=True)
            renderedRoutes.append({
                "id": route.id,
                "lineName": route.line.name,
                "isReturn": route.isReturn,
                "distance": route.distance,
                "time": route.time,
                "path": jsonroute.data
            })
        return Response(renderedRoutes)


class BestRoutesView(APIView):
    def get(self, request, o_x, o_y, d_x, d_y, *args, **kwargs):
        o_x = float(o_x.replace(',', '.'))
        o_y = float(o_y.replace(',', '.'))
        d_x = float(d_x.replace(',', '.'))
        d_y = float(d_y.replace(',', '.'))
        origin = Point(x_coord=o_x, y_coord=o_y)
        destination = Point(x_coord=d_x, y_coord=d_y)
        points = Point.objects.filter(
            steps__isnull=False).distinct().all()
        originmatch = ClosestPoint(points, origin)
        destinationmatch = ClosestPoint(points, destination)
        paths = CalculatePaths(originmatch, destinationmatch)
        response = StepTraceSerializer(paths, many=True)
        return Response(response.data)
