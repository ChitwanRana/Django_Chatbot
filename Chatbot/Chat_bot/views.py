from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from .backend import process_user_query, retrieve_all_threads, build_vectorstore
import uuid, os

def generate_thread_id():
    return str(uuid.uuid4())

def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("chat_page")
        else:
            return render(request, "login.html", {"error": "Invalid credentials"})
    return render(request, "login.html")

def logout_view(request):
    logout(request)
    return redirect("login")

@login_required
def chat_page(request):
    if "thread_id" not in request.session:
        request.session["thread_id"] = generate_thread_id()
    threads = retrieve_all_threads()
    return render(request, "chat.html", {"threads": threads, "user": request.user})

@login_required
def chat_stream(request):
    """SSE streaming endpoint."""
    message = request.GET.get("message")
    thread_id = request.session["thread_id"]

    def event_stream():
        response = process_user_query(thread_id, message)
        yield f"data: {response}\n\n"

    return StreamingHttpResponse(event_stream(), content_type="text/event-stream")

@login_required
def upload_pdf(request):
    if request.method == "POST" and request.FILES["pdf"]:
        pdf_file = request.FILES["pdf"]
        path = f"uploaded_pdfs/{pdf_file.name}"
        os.makedirs("uploaded_pdfs", exist_ok=True)
        with open(path, "wb") as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)
        build_vectorstore(path)
        return JsonResponse({"status": "PDF uploaded and indexed!"})
    return JsonResponse({"error": "Invalid request"}, status=400)


def chat_view(request):
    return render(request, "chat.html")