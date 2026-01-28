import ingest


import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Ingest a FITS file and display its header information.')
    parser.add_argument('file_path', type=str, help='Path to the FITS file to be ingested.')
    args = parser.parse_args()

    data, header = ingest.ingest_fits(args.file_path)
    print("Header Information:")
    print(header)


    plt.imshow(data, cmap='gray')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()