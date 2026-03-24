from astropy.io import fits 

def ingest_fits(file_path):
    """Ingests a FITS file and returns its data and header.

    Parameters:
    file_path (str): The path to the FITS file.

    Returns:
    tuple: A tuple containing the data and header of the FITS file.
    """
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    return data, header