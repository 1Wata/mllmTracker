import numpy as np
import pandas as pd


# def load_text_numpy(path, delimiter, dtype):
#     if isinstance(delimiter, (tuple, list)):
#         for d in delimiter:
#             try:
#                 ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
#                 return ground_truth_rect
#             except:
#                 pass

#         raise Exception('Could not read file {}'.format(path))
#     else:
#         ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
#         return ground_truth_rect




def load_text_numpy(path, delimiter, dtype):
    """Helper function to load text file using numpy.loadtxt."""
    last_exception = None # Store the last exception

    # 1. Try the explicitly provided delimiters first
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                # print(f"Attempting to load {path} with explicit delimiter '{repr(d)}'") # Debug print
                # Use skiprows=0 to be explicit, although it's default
                # Use comments='%' or '#' if your files might have comment lines
                ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype, comments='#')
                # print(f"Successfully loaded {path} with explicit delimiter '{repr(d)}'") # Debug print
                return ground_truth_rect
            except Exception as e:
                # print(f"Failed with explicit delimiter '{repr(d)}': {e}") # Debug print
                last_exception = e # Store the exception
                pass # Continue to try the next delimiter
    else: # Handle single explicit delimiter case
         try:
            # print(f"Attempting to load {path} with single explicit delimiter '{repr(delimiter)}'") # Debug print
            ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype, comments='#')
            # print(f"Successfully loaded {path} with single explicit delimiter '{repr(delimiter)}'") # Debug print
            return ground_truth_rect
         except Exception as e:
            # print(f"Failed with single explicit delimiter '{repr(delimiter)}': {e}") # Debug print
            last_exception = e

    # 2. If explicit delimiters failed, try with delimiter=None (handles any whitespace)
    # print(f"Explicit delimiters failed for {path}. Trying with delimiter=None (any whitespace).") # Debug print
    try:
        # delimiter=None treats one or more whitespace characters as a single delimiter
        ground_truth_rect = np.loadtxt(path, delimiter=None, dtype=dtype, comments='#')
        # print(f"Successfully loaded {path} with delimiter=None") # Debug print
        return ground_truth_rect
    except Exception as e:
        # print(f"Failed with delimiter=None: {e}") # Debug print
        # If None also fails, store this as the potentially more relevant exception
        last_exception = e

    # 3. If all attempts failed, raise an informative exception
    if last_exception is not None:
        raise Exception(f'Could not read file {path} with specified delimiters {repr(delimiter)} or any whitespace. Last error: {last_exception}')
    else:
        # Should not happen if delimiter was provided, but as a fallback
        raise Exception(f'Could not read file {path}. No valid delimiter found or other reading error.')


def load_text_pandas(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = pd.read_csv(path, delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                low_memory=False).values
                return ground_truth_rect
            except Exception as e:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = pd.read_csv(path, delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                        low_memory=False).values
        return ground_truth_rect


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, delimiter, dtype)
    elif backend == 'pandas':
        return load_text_pandas(path, delimiter, dtype)


def load_str(path):
    with open(path, "r") as f:
        text_str = f.readline().strip().lower()
    return text_str
