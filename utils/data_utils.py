import os
# from renishawWiRE import WDFReader
import numpy as np
import shutil
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from config import (
    PROCESSED_DATA_DIR,
    TRAIN_DATA_DIR,
    TEST_DATA_DIR,
    DATA_DIR,
    DATASET_DIR
)

# def read_wdf_file(file_name, data_dir=PROCESSED_DATA_DIR):
#   """
#   Reads a WDF file based on a file name and returns the data.
#   Parameters:
#   - file_name: The name to search for in data dir.
#   - data_dir: Directory where the WDF files are located. Default is PROCESSED_DATA_DIR.
  
#   Returns:
#   - data: The WDFReader object containing the data from the file.
#   """
  
#   # Construct the file path
#   file_path = os.path.join(data_dir, file_name)
  
#   # Check if the file exists
#   if not os.path.exists(file_path):
#       raise FileNotFoundError(f"The file {file_path} does not exist.")
  
#   # Read the WDF file
#   data = WDFReader(file_path)
  
#   return data




# def read_data(id, xmin=400, xmax=1800, data_dir=PROCESSED_DATA_DIR):
#     """Get an ID and a data directory and return the data from the WDF file.

#     Args:
#         id (str): ID of the data to be read, from metadata file.
#         xmin (int): Minimum wavenumber for interpolation.
#         xmax (int): Maximum wavenumber for interpolation.
#         data_dir (str): Data directory.
        
#     Returns:
#         xdata (numpy array): x-axis data from the WDF file, in range of xmin to xmax.
#         ydata (numpy array): y-axis data from the WDF file, combined from all files starting with id.
#     """

#     # Define the common xdata range
#     xdata = np.linspace(xmin, xmax, (xmax - xmin) + 1)

#     # Initialize a list to store all interpolated ydata
#     combined_ydata = []

#     # Open all WDF files starting with the given ID
#     for file in os.listdir(data_dir):
        
#         if file.startswith(id) and file.endswith('.wdf'):
#             # Read the WDF file
#             data = read_wdf_file(file, data_dir)
#             print('Reading file:', file)  
            
#             # Flatten the 2D grid of spectra to a 1D array
#             spectra_2d = data.spectra.reshape(-1, data.spectra.shape[-1])
            
#             # Interpolate all spectra at once using vectorized operations
#             interp_func = interp1d(data.xdata, spectra_2d, kind='linear', axis=1, fill_value="extrapolate")
#             interpolated_spectra = interp_func(xdata)
#             combined_ydata.append(interpolated_spectra)

#     # Concatenate all interpolated spectra from different files
#     ydata = np.concatenate(combined_ydata, axis=0)

#     return xdata, ydata



def plot_data(xdata, ydata, title="Raman Spectrum", x_label="Wavenumber (cm-1)", y_label="Intensity (a.u.)"):
    """Plot the Raman spectrum from the given x and y data.
    
    Args:
        xdata (numpy array): x-axis data.
        ydata (numpy array): y-axis data.
        title (str): Title of the plot.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
    """
    
    # Create a new figure
    plt.figure(figsize=(12, 6))
    
    # Plot the data
    plt.plot(xdata, ydata.T)
    
    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(400, 1800)
    
    # Display the plot
    plt.show()

def create_data_structure(
        metadata, 
        data_dir=DATASET_DIR,
        train_dir=TRAIN_DATA_DIR,
        test_dir=TEST_DATA_DIR,
        test_size=0.2,
        random_state=42,
        clear_dirs=False):
    """
    Create a data directory structure for training and testing.

    Args:
        metadata (DataFrame): Metadata containing file IDs and types.
        data_dir (str): Directory containing the raw WDF files.
        clear_dirs (bool): Whether to clear train and test directories before processing.
    """

    # Clear directories if the option is enabled
    if clear_dirs:
        print("Clearing train and test directories...")
        for directory in [train_dir, test_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"Directory {directory} cleared.")
            os.makedirs(directory, exist_ok=True)

    # Get unique ID_Types
    id_types = metadata['ID_Type'].unique()

    # Create subdirectories for each ID_Type
    for id_type in id_types:
        os.makedirs(os.path.join(train_dir, id_type), exist_ok=True)
        os.makedirs(os.path.join(test_dir, id_type), exist_ok=True)


    # Process each file
    for _, row in metadata.iterrows():
        if row['ID_Type'] != "UNK":
            idx = row['Idx']
            id_type = row['ID_Type']
            file_id = f"{idx}_{id_type}"
            print(f"Processing {file_id} with ID_Type {id_type}...")

            # load from npy file
            file_path = os.path.join(data_dir, id_type, f"{idx}_{id_type}.npy")

            try:  
                ydata = np.load(file_path)
                print(f"Loaded {file_id} data with shape {ydata.shape}")
            except Exception as e:
                print(f"Error loading {file_id} data from {file_path}: {e}")

            # Split data into train and test
            y_train, y_test = train_test_split(ydata, test_size=test_size, random_state=random_state)
            # # Save train data
            np.save(os.path.join(train_dir, id_type, f"{idx}_{id_type}_train.npy"), y_train)

            # # Save test data
            np.save(os.path.join(test_dir, id_type, f"{idx}_{id_type}_test.npy"), y_test)

    print("Data processing and saving completed.")

def load_mean(id_type, 
              data_dir=DATA_DIR, 
              subset="train",
              xmin=400,
              xmax=1800):
    """
    Load the data from the processed data directory.

    Args:
        data_dir (str): Path to the processed data directory.
        subset (str): Subset to load. Either "train" or "test".

    Returns:
        xdata (numpy array): x-axis data.
        ydata (numpy array): y-axis data.
    """

    # Get the list of files
    files = os.listdir(os.path.join(data_dir, subset, id_type))
    # Initialize array to store y data
    ydata = []

    # Load the data
    for file in files:
        if file.endswith(".npy"):
            # Load the data
            data = np.load(os.path.join(data_dir, subset, id_type, file))
            
            # Append to the lists
            ydata.append(data.mean(axis=0))

    # Convert to numpy arrays
    ydata = np.array(ydata)
    xdata = np.linspace(xmin, xmax, (xmax - xmin) + 1)
    return xdata, ydata

# https://zhauniarovich.com/post/2022/2022-09-matplotlib-graphs-in-research-papers/
def save_fig(
        fig: plt.figure, 
        fig_name: str, 
        fig_dir: str, 
        fig_fmt: str,
        fig_size: tuple[float, float] = [6.4, 4], 
        save: bool = True, 
        dpi: int = 1200,
        transparent_png = True,
    ):
    """This procedure stores the generated matplotlib figure to the specified 
    directory with the specified name and format.

    Parameters
    ----------
    fig : [type]
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str
        Format of the figure, the format should be supported by matplotlib 
        (additional logic only for pdf and png formats)
    fig_size : Tuple[float, float]
        Size of the figure in inches, by default [6.4, 4] 
    save : bool, optional
        If the figure should be saved, by default True. Set it to False if you 
        do not want to override already produced figures.
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return
    
    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(
        fig_dir,
        '{}.{}'.format(fig_name, fig_fmt.lower())
    )
    if fig_fmt == 'pdf':
        metadata={
            'Creator' : '',
            'Producer': '',
            'CreationDate': None
        }
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(
            pth, 
            bbox_inches='tight',
            dpi=dpi,
        )
    else:
        try:
            fig.savefig(pth, bbox_inches='tight')
        except Exception as e:
            print("Cannot save figure: {}".format(e)) 