import io
import PyPDF2
import pytesseract
import pdf2image
from urllib.request import urlopen

def read_pdf(filepath_or_url, is_scanned=False, is_url=False):
        
    def get_text_doc(fobj):
        pdf_bytes = io.BytesIO(fobj.read())
        pdfdoc = PyPDF2.PdfFileReader(pdf_bytes)
        num_pages = pdfdoc.numPages
        text = []

        for page_num in range(num_pages):
            page_obj = pdfdoc.getPage(page_num)
            text.append(page_obj.extractText())

        text = '\n\n'.join(text)
        print(text)
        return text
    
    def get_text_imgs(fobj):
        images = pdf2image.convert_from_bytes(fobj.read(), grayscale=True)
        det_text = []

        for page_num, page in enumerate(images):
            det_text.append(pytesseract.image_to_string(page,config='--psm 3'))
            print('+'*50)
            print(page_num)
            print('+'*50)
            print(det_text[page_num])
            print()
        return '\n\n'.join(det_text)

    if "http" in filepath_or_url or is_url:
        if not filepath_or_url.startswith("http"):
            filepath_or_url = "https://" + filepath_or_url.strip(':/')
        with urlopen(filepath_or_url) as f:
            if not is_scanned:
                text = get_text_doc(f)
                if len(text) <= 20:
                    is_scanned = True
            if is_scanned:
                text = get_text_imgs(f)
    else:
        with open(filepath_or_url,"rb") as f:
            if not is_scanned:
                text = get_text_doc(f)
                if len(text) <= 20:
                    is_scanned = True
            if is_scanned:
                text = get_text_imgs(f)
    
    return text



def read_docai_outputs(user_id, output_path, filename):
    import os
    from langchain.schema import Document
    from google.cloud import documentai

    
    folder_path = os.path.join(output_path, filename)
    if not os.path.isdir(folder_path):
        return []
    
    documents = []
    npages, nwords, nchars = (0, 0, 0)
    for f in os.listdir(folder_path):
        with open(os.path.join(folder_path, f)) as fd:
            data = fd.read()
        # Read JSON File as bytes object and convert to Document Object
        print(f"Fetching {f}")
        gs_document = documentai.Document.from_json(
            data, ignore_unknown_fields=True
        )
        npages += len(gs_document.pages)
        nwords += len(gs_document.text.split(' ')) + 1
        nchars += len(gs_document.text)
        for page in gs_document.pages:
            page_content = ""
            for ts in page.layout.text_anchor.text_segments:
                page_content += gs_document.text[ts.start_index:ts.end_index] + '\n'
            documents.append(
                Document(
                    page_content = page_content.strip('\n'),
                    metadata = {
                        "page": page.page_number,
                        "source": f"{filename}:  Pg. {page.page_number}",
                        "filename": filename,
                        "user_id": f"{user_id}"
                    }
                )
            )
    meta_content = f"{filename} has {npages} pages, {nwords} words and {nchars} characters."
    meta_doc = Document(
        page_content=meta_content,
        metadata = {
            "page": 0,
            "source": filename,
            "filename": filename,
            "user_id": f"{user_id}"
        }
    )
    return documents + [meta_doc]