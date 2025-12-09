import heapq
from django.shortcuts import get_object_or_404
from rest_framework.response import Response
from .models import Line, Point, Step, Route
from rest_framework import viewsets
from rest_framework.views import APIView
from .serializers import LineSerializer, PointSerializer
from .serializers import StepSerializer, RouteSerializer
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


def calculatePaths(
    start_point_id: int,
    end_point_id: int,
    K: int = 3,
    max_path_length: int = 100,
    switch_cost: float = 0.0,
) -> list[tuple[list[Step], float]]:
    all_steps = Step.objects.select_related('point', 'next__point').all()
    steps_by_point = {}
    paths_found_at_step = {step.id: 0 for step in all_steps}
    entry_count = 0
    pq = []
    for step in all_steps:
        point_id = step.point.id
        if point_id not in steps_by_point:
            steps_by_point[point_id] = []
        steps_by_point[point_id].append(step)
    start_steps = steps_by_point.get(start_point_id, [])
    if not start_steps:
        return []
    for step in start_steps:
        heapq.heappush(pq, (0.0, entry_count, [step]))
        entry_count += 1
    k_results = []
    while pq:
        current_dist, _, current_path = heapq.heappop(pq)
        current_step = current_path[-1]
        current_step_id = current_step.id
        if len(current_path) > max_path_length:
            continue
        if current_step.point.id == end_point_id:
            k_results.append((current_path, current_dist))
            if len(k_results) >= K:
                break
        paths_found_at_step[current_step_id] += 1
        if paths_found_at_step[current_step_id] > K * 2:
            continue
        if current_step.next:
            neighbor = current_step.next
            weight = current_step.distance_to_next_step()
            new_distance = current_dist + weight
            new_path = current_path + [neighbor]
            if neighbor not in current_path:
                heapq.heappush(pq, (new_distance, entry_count, new_path))
                entry_count += 1
        for switch_neighbor in steps_by_point.get(current_step.point.id, []):
            if switch_neighbor.id == current_step_id:
                continue
            new_distance = current_dist + switch_cost
            new_path = current_path + [switch_neighbor]
            if switch_neighbor not in current_path:
                heapq.heappush(pq, (new_distance, entry_count, new_path))
                entry_count += 1
    return k_results


def convertBestPathsToResponse(bestPaths):
    result = []
    for tup in bestPaths:
        currentRoute = None
        l, dis = tup
        l: list[Step] = l
        segments = []
        currentSegment = {}
        currentSteps = []
        for step in l:
            if currentRoute is None:
                currentRoute = step.route
                currentSegment["route"] = RouteSerializer(step.route).data
            if currentRoute != step.route:
                currentSegment["path"] = currentSteps
                segments.append(currentSegment)
                currentSegment = {"route": RouteSerializer(step.route).data}
                currentSteps = []
                currentRoute = step.route
            currentSteps.append(PointSerializer(step.point).data)
        result.append({"distance": dis, "segments": segments})
    return result


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
                "lineColor": route.line.color,
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
        o = ClosestPoint(points, origin)
        d = ClosestPoint(points, destination)
        result = calculatePaths(o.id, d.id)
        renderedResult = convertBestPathsToResponse(result)
        print(renderedResult)
        return Response(renderedResult)
