from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/chat/$', consumers.ChatConsumer.as_asgi()),
]

# With this routing configuration in place, WebSocket connections
# to ws://yourdomain/ws/chat/ will be handled by ChatConsumer