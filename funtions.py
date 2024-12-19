def slice_text(text, chunk_size=1000, chunk_overlap=200):
    slices = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        slice = text[start:end]
        slices.append(slice)
        start = end - chunk_overlap 
    return slices

