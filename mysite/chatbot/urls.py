from django.urls import path
from . import views

app_name = 'chatbot'

urlpatterns = [
    path('', views.chat_ui, name='chat_ui'),
    path('api/send/', views.send_message, name='send_message'),
    path('api/history/', views.get_history, name='get_history'),
    path('api/upload/', views.upload_file, name='upload_file'),
]
