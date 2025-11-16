from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from .backend import process_user_query, retrieve_all_threads, build_vectorstore, get_thread_history, delete_thread_history
import uuid
import os

def generate_thread_id():
    return str(uuid.uuid4())

def login_view(request):
    if request.user.is_authenticated:
        return redirect('new_chat')
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("new_chat")
        else:
            return render(request, "login.html", {"error": "Invalid credentials"})
    return render(request, "login.html")

def logout_view(request):
    logout(request)
    return redirect("login")

@login_required
def chat_page(request, thread_id=None):
    if thread_id is None:
        # If no thread is specified, redirect to start a new one
        return redirect('new_chat')

    all_user_threads = retrieve_all_threads() 
    thread_objects = [{"thread_id": tid} for tid in all_user_threads]
    
    # Get messages for the current thread
    messages = get_thread_history(thread_id)

    context = {
        "threads": thread_objects,
        "current_thread_id": thread_id,
        "messages": messages,
        "user": request.user,
    }
    return render(request, "chat.html", context)

@login_required
def new_chat(request):
    # Generate a new unique thread_id and redirect
    new_thread_id = str(uuid.uuid4())
    return redirect('chat_page', thread_id=new_thread_id)

@login_required
def delete_thread(request, thread_id):
    if request.method == "POST":
        delete_thread_history(thread_id)
    return redirect('new_chat')

@login_required
def chat_stream(request, thread_id):
    """SSE streaming endpoint."""
    message = request.GET.get("message")
    if not message:
        return StreamingHttpResponse(status=400)

    def event_stream():
        response_generator = process_user_query(thread_id, message)
        for chunk in response_generator:
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingHttpResponse(event_stream(), content_type="text/event-stream")

@login_required
def upload_pdf(request):
    if request.method == "POST" and request.FILES.get("pdf"):
        pdf_file = request.FILES["pdf"]
        
        # Ensure the directory exists
        upload_dir = "uploaded_pdfs"
        os.makedirs(upload_dir, exist_ok=True)
        
        path = os.path.join(upload_dir, pdf_file.name)
        
        with open(path, "wb") as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)
        
        # This rebuilds the vector store globally.
        build_vectorstore(path)
        return JsonResponse({"status": "PDF uploaded and indexed successfully!"})
    return JsonResponse({"error": "Invalid request. Please upload a PDF file."}, status=400)