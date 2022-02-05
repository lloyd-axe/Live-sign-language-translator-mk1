from django.urls import path
from .views import *

urlpatterns = [
    path('', Home, name='test'),
    path('stream', StreamVideo, name='streamVideo'),
    path('update', update, name='update')
]
