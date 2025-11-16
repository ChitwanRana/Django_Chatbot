"""
URL configuration for Chatbot project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
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
    # Redirect root to a new chat, or you can change it to a landing page
    path('', views.chat_page, name='home'), 
    
    # Page for a specific chat thread
    path('chat/<str:thread_id>/', views.chat_page, name='chat_page'),
    
    # Creates a new chat thread and redirects to it
    path('new/', views.new_chat, name='new_chat'),
    
    # Deletes a chat thread
    path('delete/<str:thread_id>/', views.delete_thread, name='delete_thread'),
    
    # SSE streaming endpoint for a specific thread
    path('stream/chat/<str:thread_id>/', views.chat_stream, name='chat_stream'),
    
    # PDF upload endpoint
    path('upload_pdf/', views.upload_pdf, name='upload_pdf'),
    
    # Auth
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
]


