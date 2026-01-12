from django.shortcuts import render
from .models import TrainingSession

def session_list(request):
    # Show only active sessions to public
    sessions = TrainingSession.objects.filter(is_active=True)
    return render(request, 'feedback/public_session_list.html', {'sessions': sessions})
