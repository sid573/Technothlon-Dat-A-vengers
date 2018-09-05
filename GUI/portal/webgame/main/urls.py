"""webgame URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.Index, name='index'),
    path('start', views.Start, name='start'),
    path('createUsers', views.createUsers, name='createUsers'),
    path('view_1', views.View_1, name='start'),
    path('view_2', views.View_2, name='start'),
    path('view_3', views.View_3, name='start'),
    path('view_4', views.View_4, name='start'),
    path('view_5', views.View_5, name='start'),
    path('view_6', views.View_6, name='start'),
    path('view_7', views.View_7, name='start'),
    path('view_8', views.View_8, name='start'),
    path('view_9', views.View_9, name='start'),
    path('c_p', views.c_p, name='start'),
    path('c_p_revert', views.c_p_revert, name='start'),
    path('Test_acc', views.Test_acc, name='start'),
]
 