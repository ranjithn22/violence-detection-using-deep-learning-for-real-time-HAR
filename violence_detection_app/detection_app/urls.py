from django.urls import path
from .views import index, register, render_home,login,approve_user,admin_dashboard,admin_register,admin_login,user_dashboard

urlpatterns = [
    path('', render_home, name='home'),  # Home page
    path('upload/', index, name='index'),  # Video upload and prediction
    path('register/', register, name='register'),  # Registration page
    path('login/', login, name='login'),
    path('approve_user/', approve_user, name='approve_user'),
    path('admin_dashboard/', admin_dashboard, name='admin_dashboard'),
    path('adminregister/', admin_register, name='admin_register'),
    path('adminlogin/', admin_login, name='admin_login'),
    path('dashboard/', user_dashboard, name='user_dashboard'),
]
