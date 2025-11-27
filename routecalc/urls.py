from django.urls import include, path
from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register(r"lines", views.LineViewSet)
router.register(r"points", views.PointViewSet)
router.register(r"paths", views.PathViewSet)
router.register(r"routes", views.RouteViewSet)

urlpatterns = [
    path('', include(router.urls))
]
