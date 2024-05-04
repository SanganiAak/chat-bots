import pdfplumber

def transcribe_pdf_files(uploaded_files):
    document_texts = []
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            full_text = ''
            for page in pdf.pages:
                full_text += page.extract_text() + '\n'
            document_texts.append(full_text)
    return document_texts
