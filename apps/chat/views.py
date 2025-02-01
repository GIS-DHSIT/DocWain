# apps/chat/views.py
from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .forms import DocumentForm
from .models import Document
from .services import DocumentService

class ChatView(TemplateView):
    template_name = 'chat/index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = DocumentForm()
        return context

@require_http_methods(["POST"])
def upload_document(request):
    files = request.FILES.getlist('file')
    if not files:
        return JsonResponse({
            'status': 'error',
            'message': 'No files were uploaded'
        }, status=400)

    saved_documents = []
    errors = []

    for file in files:
        form = DocumentForm(files={'file': file})
        if form.is_valid():
            try:
                document = form.save(commit=False)
                document.name = file.name
                document.save()
                saved_documents.append(document)
            except Exception as e:
                errors.append(f"Error saving {file.name}: {str(e)}")
        else:
            errors.append(f"Invalid file {file.name}: {form.errors['file'][0]}")

    if saved_documents:
        try:
            DocumentService.process_documents(saved_documents)
            return JsonResponse({
                'status': 'success',
                'message': f'Successfully uploaded {len(saved_documents)} documents',
                'documents': [{'id': doc.id, 'name': doc.name} for doc in saved_documents],
                'errors': errors if errors else None
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e),
                'documents': [{'id': doc.id, 'name': doc.name} for doc in saved_documents],
                'errors': errors + [str(e)]
            }, status=500)
    else:
        return JsonResponse({
            'status': 'error',
            'message': 'No files were successfully uploaded',
            'errors': errors
        }, status=400)