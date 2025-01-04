from PyPDF2 import PdfReader

class ChunkHelpers:
    # Utility function: Extract paragraphs from a PDF
    @staticmethod
    def extract_paragraphs_from_pdf(pdf_file):
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Split text into paragraphs based on single newlines (adjust logic if needed)
        paragraphs = text.split("\n")
        return paragraphs

    # Utility function: Process PDF and generate chunks based on paragraphs
    @staticmethod
    def process_pdf_to_chunks(pdf_file, max_chunk_length=500):
        paragraphs = ChunkHelpers.extract_paragraphs_from_pdf(pdf_file)
        chunks = []
        current_chunk = []

        # Combine paragraphs into chunks of a reasonable size
        for paragraph in paragraphs:
            # If adding this paragraph exceeds max chunk length, start a new chunk
            if len(" ".join(current_chunk)) + len(paragraph) > max_chunk_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [paragraph]  # Start new chunk with the current paragraph
            else:
                current_chunk.append(paragraph)  # Add paragraph to the current chunk

        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
