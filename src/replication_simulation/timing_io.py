"""Helpers for replication timing tables."""

def sanitize_chr_dataframe(df):
    """
    Takes a pandas DataFrame and checks whether the first column header looks like 'Chr', 'chrom', etc.
    If so, the header row is discarded and column names are reset to integers: 0, 1, 2, ...
    """
    chr_like = ['chr', 'chrom', 'chromosome']
    
    # Get the first column header, normalized
    first_col = str(df.columns[0]).strip().lower()
    
    if any(first_col.startswith(prefix) for prefix in chr_like):
        # Header is present, so remove the first row and reset index
        df = df.drop(index=0).reset_index(drop=True)

    # Rename all columns to integers
    df.columns = range(df.shape[1])
    
    return df
