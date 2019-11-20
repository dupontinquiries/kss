#insert into distr()

file_size = 1 # file_size(file_name_chunk)
if file_size >= (10 ** 9):
    if verbose:
        print('file {0} is large ({1})\n(Future Capability) Keeping the chunked clip as \"cc\"' \
            .format(file_name_chunk, file_size))
