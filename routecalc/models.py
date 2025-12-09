from django.db import models
import numpy
from pyproj import Transformer
import pyproj


class Line(models.Model):
    name = models.CharField("Nombre", max_length=5)

    def __str__(self):
        return self.name


class Point(models.Model):
    x_coord = models.FloatField("Coordenada X")
    y_coord = models.FloatField("Coordenada Y")

    def __str__(self):
        return f"({self.x_coord}, {self.y_coord})"

    def __array__(self) -> numpy.ndarray:
        transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:32720", always_xy=True)
        x_coord, y_coord = transformer.transform(self.x_coord, self.y_coord)
        return numpy.array([x_coord, y_coord], dtype=float)


class Route(models.Model):
    line = models.ForeignKey(
        Line,
        on_delete=models.CASCADE,
        related_name='routes'
    )
    isReturn = models.BooleanField()
    distance = models.FloatField()
    time = models.FloatField()
    first = models.ForeignKey(
        'Step',
        on_delete=models.CASCADE,
        related_name='routes'
    )


class Step(models.Model):
    route = models.ForeignKey(
        Route,
        on_delete=models.CASCADE,
    )
    point = models.ForeignKey(
        Point,
        on_delete=models.CASCADE,
        related_name="steps"
    )
    next = models.ForeignKey(
        'self',
        on_delete=models.CASCADE,
        related_name="steps",
        blank=True,
        null=True
    )

    def __str__(self):
        res = f"{self.point.__str__()} -> "
        res += f"{self.next.point.__str__() if self.next is not None else ''}"
        return res
