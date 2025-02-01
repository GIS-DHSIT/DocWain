from django.urls import path
from . import views

app_name = 'chat'  # Define the application namespace

urlpatterns = [
    path('', views.ChatView.as_view(), name='index'),
    path('upload/', views.upload_document, name='upload_document'),
]