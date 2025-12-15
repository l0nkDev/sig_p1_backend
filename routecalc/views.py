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
from .spatial_index import PointSpatialIndex


def ClosestPoint(points: QuerySet[Point], point: Point) -> Point:
    nparray = numpy.array(points)
    tree = KDTree(nparray)
    _, index = tree.query(point.__array__())
    return points[int(index)]


def ClosestPoints(points_qs, target_point, radius=50.0) -> list:
    points_list = list(points_qs)
    if not points_list:
        return []
    coords = numpy.array([p.__array__() for p in points_list])
    tree = KDTree(coords)
    target_coords = target_point.__array__()
    indices = tree.query_ball_point(target_coords, radius)
    nearby_points = [points_list[i] for i in indices]
    return nearby_points


def DistanceBetween(point_a: Point, point_b: Point) -> float:
    return numpy.linalg.norm(point_a.__array__() - point_b.__array__())


def reconstruct_path(end_step_id, predecessors, step_map):
    path = []
    current_id = end_step_id
    while current_id is not None:
        step = step_map[current_id]
        path.append(step)
        current_id = predecessors.get(current_id)
    path.reverse()
    return path


def find_best_path(start_steps, end_point_ids, steps_by_point, step_map,
                   switch_cost, penalized_edges, start_costs, end_costs):
    pq = []
    distances = {step_id: (float('inf'), float('inf'))
                 for step_id in step_map.keys()}
    predecessors = {}
    entry_count = 0
    best_result = (None, float('inf'), None, None)
    for step in start_steps:
        walk_cost = start_costs.get(step.point.id, 0.0)
        cost = (walk_cost, 0)
        distances[step.id] = cost
        heapq.heappush(pq, (walk_cost, 0, entry_count, step.id))
        entry_count += 1
    EDGE_PENALTY = 100000.0
    while pq:
        current_dist, current_switches, _, current_step_id = heapq.heappop(pq)
        if current_dist > best_result[1]:
            continue
        current_step = step_map[current_step_id]
        if (current_dist > distances[current_step_id][0] and
                current_switches > distances[current_step_id][1]):
            continue
        if current_step.point.id in end_point_ids:
            final_walk = end_costs.get(current_step.point.id, 0.0)
            total_dist = current_dist + final_walk
            if total_dist < best_result[1]:
                path = reconstruct_path(current_step_id, predecessors,
                                        step_map)
                best_result = (path, total_dist, path[0].point.id,
                               current_step.point.id)
        if current_step.next:
            neighbor = current_step.next
            weight = current_step.distance_to_next_step()
            edge_key = (current_step_id, neighbor.id)
            if edge_key in penalized_edges:
                weight += EDGE_PENALTY
            new_dist = current_dist + weight
            new_switches = current_switches
            new_cost = (new_dist, new_switches)
            if new_cost < distances[neighbor.id]:
                distances[neighbor.id] = new_cost
                predecessors[neighbor.id] = current_step_id
                heapq.heappush(pq, (new_dist, new_switches,
                               entry_count, neighbor.id))
                entry_count += 1
        for switch_neighbor in steps_by_point.get(current_step.point.id, []):
            if switch_neighbor.id == current_step_id:
                continue
            new_dist = current_dist + switch_cost
            new_switches = current_switches + 1
            new_cost = (new_dist, new_switches)
            if new_cost < distances[switch_neighbor.id]:
                distances[switch_neighbor.id] = new_cost
                predecessors[switch_neighbor.id] = current_step_id
                heapq.heappush(pq, (new_dist, new_switches,
                               entry_count, switch_neighbor.id))
                entry_count += 1
    return best_result


def calculatePaths(
    start_point_ids: list[int],
    end_point_ids: list[int],
    start_costs: dict,
    end_costs: dict,
    K: int = 3,
    switch_cost: float = 0.001,
) -> list[tuple[list, float]]:
    all_steps = list(Step.objects.select_related(
        'point', 'next__point', 'route').all())
    step_map = {step.id: step for step in all_steps}
    steps_by_point = {}
    for step in all_steps:
        point_id = step.point.id
        if point_id not in steps_by_point:
            steps_by_point[point_id] = []
        steps_by_point[point_id].append(step)
    start_steps = []
    for point_id in start_point_ids:
        start_steps.extend(steps_by_point.get(point_id, []))
    if not start_steps:
        return []
    end_point_set = set(end_point_ids)
    k_results = []
    penalized_edges = set()
    current_start_costs = start_costs.copy()
    current_end_costs = end_costs.copy()
    POINT_REUSE_PENALTY = 100000.0
    for _ in range(K * 2):
        path, search_score, s_id, e_id = find_best_path(
            start_steps,
            end_point_set,
            steps_by_point,
            step_map,
            switch_cost,
            penalized_edges,
            current_start_costs,
            current_end_costs
        )
        if not path:
            break
        path_ids = tuple(s.id for s in path)
        if any(tuple(s.id for s in res[0]) == path_ids for res in k_results):
            pass
        else:
            real_distance = start_costs.get(s_id, 0.0)
            for i in range(len(path) - 1):
                real_distance += path[i].distance_to_next_step()
            real_distance += end_costs.get(e_id, 0.0)
            k_results.append((path, real_distance))
            if s_id in current_start_costs:
                current_start_costs[s_id] += POINT_REUSE_PENALTY
            if e_id in current_end_costs:
                current_end_costs[e_id] += POINT_REUSE_PENALTY
        for i in range(len(path) - 1):
            source_step = path[i]
            dest_step = path[i+1]
            if source_step.route.id == dest_step.route.id:
                penalized_edges.add((source_step.id, dest_step.id))
        if len(k_results) >= K:
            break
    return k_results


def convertBestPathsToResponse(bestPaths, o_x, o_y, d_x, d_y):
    result = []
    for tup in bestPaths:
        currentRoute = None
        l, dis = tup
        l: list[Step] = l
        segments = []
        currentSegment = {}
        currentSteps = []
        walkingrouteOr = Route(
            id=999,
            line=Line(id=998, name="L000", color="#000000"),
            isReturn=False,
            distance=0,
            time=0
        )
        walkingrouteDes = Route(
            id=1001,
            line=Line(id=1000, name="L000", color="#000000"),
            isReturn=False,
            distance=0,
            time=0
        )
        originToPoint = {"route": RouteSerializer(walkingrouteOr).data,
                         "path": [PointSerializer(Point(x_coord=o_x,
                                                        y_coord=o_y)).data,
                                  PointSerializer(l[0].point).data]}
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
        currentSegment["path"] = currentSteps
        segments.append(currentSegment)
        currentSegment = {"route": RouteSerializer(step.route).data}
        currentSteps.append(PointSerializer(step.point).data)
        destinationToPoint = {"route": RouteSerializer(walkingrouteDes).data,
                              "path": [PointSerializer(l[-1].point).data,
                                       PointSerializer(
                                           Point(x_coord=d_x,
                                                 y_coord=d_y)).data]}
        result.append({
            "distance": dis,
            "segments": [originToPoint] + segments + [destinationToPoint]}
                      )
    return result


def RenderRoute(route: Route):
    steps = []
    currentStep: Step = route.first
    while currentStep.next is not None:
        steps.append(currentStep.point)
        currentStep = currentStep.next
    steps.append(currentStep.point)
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
        points = PointSpatialIndex()
        o_l = points.query_radius(origin, radius_meters=400.0)
        d_l = points.query_radius(destination, radius_meters=400.0)
        if not o_l:
            o = points.query(origin)
            o_l = [o]
        if not d_l:
            d = points.query(destination)
            d_l = [d]
        start_costs = {p.id: DistanceBetween(origin, p) for p in o_l}
        end_costs = {p.id: DistanceBetween(destination, p) for p in d_l}
        start_ids = [p.id for p in o_l]
        end_ids = [p.id for p in d_l]
        result = calculatePaths(start_ids, end_ids, start_costs, end_costs,
                                5, 200.0)
        renderedResult = convertBestPathsToResponse(result, o_x, o_y, d_x, d_y)
        return Response(renderedResult)
