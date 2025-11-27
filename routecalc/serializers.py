from rest_framework import serializers
from .models import Line, Route, Path, Point


class LineSerializer(serializers.ModelSerializer):
    class Meta:
        model = Line
        fields = ('name', 'routes')


class PointSerializer(serializers.ModelSerializer):
    class Meta:
        model = Point
        fields = ('x_coord', 'y_coord')


class PathSerializer(serializers.ModelSerializer):
    class Meta:
        model = Path
        fields = ('origin', 'destination', 'line')


class RouteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Route
        fields = ('line', 'direction', 'starting_point')
