from django.urls import include, path
from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register(r"lines", views.LineViewSet)
router.register(r"points", views.PointViewSet)
router.register(r"steps", views.StepViewSet)
router.register(r"routes", views.RouteViewSet)

urlpatterns = [
    path('lines/<int:line_id>/routes',
         views.LineRoutesView.as_view(), name='line-routes'),
    path('routes/range/<str:x_coord>/<str:y_coord>/<str:radius>',
         views.CloseRoutesView.as_view(), name='close-routes'),
    path('routes/best/<str:o_x>/<str:o_y>/<str:d_x>/<str:d_y>',
         views.BestRoutesView.as_view(), name='best-routes'),
    path('', include(router.urls)),
]
