from django import forms
from .models import Document


class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ['file']

    def clean_file(self):
        file = self.cleaned_data.get('file')
        allowed_types = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.csv']

        if file:
            file_extension = file.name.lower()[file.name.rfind('.'):]
            if file_extension not in allowed_types:
                raise forms.ValidationError(f'File type {file_extension} is not supported. '
                                            f'Please upload files of type: {", ".join(allowed_types)}')
        return file