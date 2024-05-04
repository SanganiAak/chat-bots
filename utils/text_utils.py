def custom_text_splitter(text, lines_per_chunk):
    chunks = []
    lines = str(text).split('\n')
    non_empty_lines = [line for line in lines if line.strip()]  # Remove blank lines
    for i in range(0, len(non_empty_lines), lines_per_chunk):
        chunk_lines = non_empty_lines[i:i+lines_per_chunk]
        chunk = '\n'.join(chunk_lines)
        chunks.append(chunk)
    return chunks
