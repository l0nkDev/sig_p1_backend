from django.db import models
import numpy


class Line(models.Model):
    name = models.CharField("Nombre", max_length=5, primary_key=True)

    def __str__(self):
        return self.name


class Point(models.Model):
    x_coord = models.FloatField("Coordenada X")
    y_coord = models.FloatField("Coordenada Y")

    def __str__(self):
        return f"({self.x_coord}, {self.y_coord})"

    def __array__(self) -> numpy.ndarray:
        return numpy.array([self.x_coord, self.y_coord], dtype=float)


class Path(models.Model):
    origin = models.ForeignKey(
        Point,
        on_delete=models.CASCADE,
        related_name='paths'
        )
    destination = models.ForeignKey(
        Point,
        on_delete=models.CASCADE,
        )
    line = models.ForeignKey(
        Line,
        on_delete=models.CASCADE,
    )

    def __str__(self):
        return f"{self.origin.__str__()} -> {self.destination.__str__()}"


class Route(models.Model):
    line = models.ForeignKey(
        Line,
        on_delete=models.CASCADE,
        related_name="routes"
        )
    direction = models.BooleanField()
    starting_point = models.ForeignKey(Path, on_delete=models.CASCADE)
