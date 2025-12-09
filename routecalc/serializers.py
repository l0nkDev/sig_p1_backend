from rest_framework import serializers
from .models import Line, Route, Step, Point


class PointSerializer(serializers.ModelSerializer):
    class Meta:
        model = Point
        fields = ('x_coord', 'y_coord')


class SimpleLineSerializer(serializers.ModelSerializer):

    class Meta:
        model = Line
        fields = ('id', 'name')


class RouteSerializer(serializers.ModelSerializer):
    line = SimpleLineSerializer()

    class Meta:
        model = Route
        fields = ('id', 'isReturn', 'distance', 'time', 'line')


class StepSerializer(serializers.ModelSerializer):
    point = PointSerializer()
    route = RouteSerializer()

    class Meta:
        model = Step
        fields = ('route', 'point')


class LineSerializer(serializers.ModelSerializer):
    routes = RouteSerializer(many=True)

    class Meta:
        model = Line
        fields = ('id', 'name', 'routes')


class StepTraceSerializer(serializers.Serializer):
    steps = StepSerializer(many=True)
    length = serializers.FloatField()
    transfers = serializers.IntegerField()
