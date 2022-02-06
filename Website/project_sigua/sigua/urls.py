from django.urls import path
from .views import *

urlpatterns = [
    path('', Home, name='home'),
    path('stream', StreamVideo, name='streamVideo'),
    path('update', Update, name='update'),
    path('about', About, name='about')
]
