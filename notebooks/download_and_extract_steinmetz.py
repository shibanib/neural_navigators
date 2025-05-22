import os
import sys
import requests
import zipfile
import tarfile
import shutil
import time

# Configuration
FIGSHARE_MAIN_DATASET_ZIP_ID = "9598406" # ID for the page, the direct download is different
# The actual file ID from the network tab when downloading the 8.25GB zip
# This seems to be the direct link for the 'allData.zip' which contains 'allData.tar'
ACTUAL_FILE_DOWNLOAD_ID = "17959061" # This corresponds to the 8.25GB zip file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DOWNLOAD_DIR = os.path.join(BASE_DIR, "steinmetz_temp_download")
# This will be the dataroot for steinmetz_NMA.ipynb
FINAL_DATA_PARENT_DIR = os.path.join(BASE_DIR, "steinmetz_data") # Parent for 'lfp_data'
FINAL_SESSIONS_DIR = os.path.join(FINAL_DATA_PARENT_DIR, "raw_sessions")


DOWNLOAD_URL = f"https://ndownloader.figshare.com/files/{ACTUAL_FILE_DOWNLOAD_ID}"
DOWNLOADED_ZIP_NAME = "steinmetz_main_dataset.zip" # The ~8.25 GB file
INNER_TAR_NAME = "allData.tar"
# Figshare's allData.tar extracts its contents into a directory named "allData" when untarred directly
SESSION_TAR_SOURCE_DIR_NAME = "allData"


def download_file_with_progress(url, filepath):
    """Downloads a file from a URL to a given filepath, showing progress."""
    print(f"Downloading {url} to {filepath}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        # Make a HEAD request first to check content-length if possible and to ensure the server is okay with the User-Agent
        head_response = requests.head(url, headers=headers, allow_redirects=True, timeout=20)
        head_response.raise_for_status()
        total_size = int(head_response.headers.get('content-length', 0))
        print(f"Expected file size: {total_size / (1024 * 1024):.2f} MB")

        if total_size < 1024 * 1024 * 1000: # Check if less than ~1GB, which is too small for the expected dataset
            print(f"Warning: Expected size ({total_size / (1024 * 1024):.2f} MB) seems too small for the Steinmetz dataset.")
            print("This might indicate an issue with the download link or that Figshare is not serving the correct file.")
            # Proceed with caution, or you could add a stricter check here to abort if too small.

        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        # Re-check total_size from the GET request's headers, in case HEAD was different or not fully informative
        total_size_get = int(response.headers.get('content-length', total_size)) # Use previous total_size as fallback

        with open(filepath, 'wb') as f:
            downloaded_size = 0
            start_time = time.time()
            for chunk in response.iter_content(chunk_size=8192*4):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size_get > 0:
                        done = int(50 * downloaded_size / total_size_get)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded_size / (1024 * 1024):.2f} MB / {total_size_get / (1024 * 1024):.2f} MB")
                    else:
                        sys.stdout.write(f"\rDownloaded {downloaded_size / (1024 * 1024):.2f} MB (total size unknown)")
                    sys.stdout.flush()
        elapsed_time = time.time() - start_time
        sys.stdout.write(f'\nDownload complete in {elapsed_time:.2f} seconds. Final size: {os.path.getsize(filepath) / (1024*1024):.2f} MB\n')
        
        # Verify downloaded size against expected size more strictly
        if total_size_get > 0 and downloaded_size < total_size_get * 0.9: # If less than 90% of expected
             print(f"Error: Downloaded size ({downloaded_size / (1024*1024):.2f} MB) is significantly less than expected ({total_size_get / (1024*1024):.2f} MB).")
             print("The downloaded file might be incomplete or incorrect.")
             # os.remove(filepath) # Optionally remove the faulty download
             # return False # And indicate failure
        elif os.path.getsize(filepath) < 1024 * 1024 * 1000: # If less than 1GB, it's very likely wrong.
            print(f"Error: The downloaded file is only {os.path.getsize(filepath) / (1024*1024):.2f}MB, which is too small for the dataset.")
            return False

        return True
    except requests.exceptions.Timeout:
        print(f"\nError: The download request timed out while trying to reach {url}")
        if os.path.exists(filepath): os.remove(filepath)
        return False
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading {url}: {e}")
        if os.path.exists(filepath): os.remove(filepath)
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred during download: {e}")
        if os.path.exists(filepath): os.remove(filepath)
        return False

def extract_zip(zip_filepath, dest_dir):
    """Extracts a zip file to a destination directory."""
    print(f"Extracting {zip_filepath} to {dest_dir}...")
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print("ZIP extraction complete.")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_filepath} is not a valid ZIP file or is corrupted.")
        return False
    except Exception as e:
        print(f"Error extracting ZIP {zip_filepath}: {e}")
        return False

def extract_tar(tar_filepath, dest_dir, mode='r:*'): # Changed mode to auto-detect compression
    """Extracts a tar file to a destination directory."""
    print(f"Extracting {tar_filepath} to {dest_dir}...")
    try:
        with tarfile.open(tar_filepath, mode) as tar_ref:
            tar_ref.extractall(path=dest_dir)
        print(f"TAR extraction complete for {os.path.basename(tar_filepath)}.")
        return True
    except tarfile.ReadError as e:
        print(f"Error: {tar_filepath} is not a valid TAR file or is corrupted: {e}")
        return False
    except Exception as e:
        print(f"Error extracting TAR {tar_filepath}: {e}")
        return False

def main():
    """Main function to download and extract Steinmetz data."""
    print("Starting Steinmetz dataset extraction process (assuming manual download)...")

    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(FINAL_SESSIONS_DIR, exist_ok=True)
    print(f"Temporary download directory (place downloaded .zip here): {TEMP_DOWNLOAD_DIR}")
    print(f"Final raw session data will be in: {FINAL_SESSIONS_DIR}")

    potential_zip_files = [f for f in os.listdir(TEMP_DOWNLOAD_DIR) if f.lower().endswith(".zip")]
    if not potential_zip_files:
        print(f"Error: No .zip file found in {TEMP_DOWNLOAD_DIR}.")
        print(f"Please manually download the Steinmetz dataset (approx 8.25GB, from the main Figshare page)")
        print(f"and place it in {TEMP_DOWNLOAD_DIR} before running this script again.")
        return

    initial_downloaded_zip_path = ""
    max_size = 0
    for zip_file_name in potential_zip_files:
        path = os.path.join(TEMP_DOWNLOAD_DIR, zip_file_name)
        size = os.path.getsize(path)
        if size > max_size:
            max_size = size
            initial_downloaded_zip_path = path
            
    if not initial_downloaded_zip_path:
        print("Error: Could not identify the initially downloaded ZIP file. Aborting.")
        return

    print(f"Found manually downloaded initial ZIP file: {initial_downloaded_zip_path} (Size: {max_size / (1024*1024):.2f} MB)")

    if max_size < 1024 * 1024 * 1000: # Less than 1GB
        print(f"Warning: The found ZIP file {initial_downloaded_zip_path} is smaller than 1GB.")
        print("This might not be the correct Steinmetz dataset archive.")
        proceed = input("Do you want to attempt to extract it anyway? (yes/no): ").lower()
        if proceed != 'yes':
            print("Aborting.")
            return
    
    # --- MODIFICATION FOR NESTED ZIP ---
    # Stage 1: Extract the initial downloaded ZIP (e.g., 9598406.zip)
    # We expect this to contain 'spikeAndBehavioralData.zip'
    nested_zip_name = "spikeAndBehavioralData.zip"
    path_to_nested_zip = os.path.join(TEMP_DOWNLOAD_DIR, nested_zip_name)

    if not os.path.exists(path_to_nested_zip):
        print(f"Extracting initial ZIP: {initial_downloaded_zip_path} to find {nested_zip_name}...")
        if not extract_zip(initial_downloaded_zip_path, TEMP_DOWNLOAD_DIR):
            print(f"Failed to extract {initial_downloaded_zip_path}. Aborting.")
            return
        if not os.path.exists(path_to_nested_zip):
            print(f"Error: Nested ZIP '{nested_zip_name}' not found in {TEMP_DOWNLOAD_DIR} after extracting initial ZIP.")
            print(f"Contents of {TEMP_DOWNLOAD_DIR}: {os.listdir(TEMP_DOWNLOAD_DIR)}")
            return
    else:
        print(f"Nested ZIP '{nested_zip_name}' already exists. Skipping extraction of initial ZIP.")

    # Stage 2: Extract the nested ZIP (spikeAndBehavioralData.zip) to get allData.tar
    inner_tar_path = os.path.join(TEMP_DOWNLOAD_DIR, INNER_TAR_NAME) # INNER_TAR_NAME is 'allData.tar'
    if not os.path.exists(inner_tar_path):
        print(f"Extracting nested ZIP: {path_to_nested_zip} to find {INNER_TAR_NAME}...")
        if not extract_zip(path_to_nested_zip, TEMP_DOWNLOAD_DIR):
            print(f"Failed to extract nested ZIP {path_to_nested_zip}. Aborting.")
            return
        if not os.path.exists(inner_tar_path):
            print(f"Error: {INNER_TAR_NAME} not found in {TEMP_DOWNLOAD_DIR} after extracting nested ZIP '{nested_zip_name}'.")
            print(f"Contents of {TEMP_DOWNLOAD_DIR}: {os.listdir(TEMP_DOWNLOAD_DIR)}")
            return
    else:
        print(f"{INNER_TAR_NAME} already exists. Skipping extraction of nested ZIP '{nested_zip_name}'.")
    # --- END OF MODIFICATION ---
        
    # 3. Extract allData.tar to get individual session .tar files
    session_tars_parent_dir = os.path.join(TEMP_DOWNLOAD_DIR, SESSION_TAR_SOURCE_DIR_NAME) # SESSION_TAR_SOURCE_DIR_NAME is "allData"
    needs_allData_tar_extraction = True
    if os.path.exists(session_tars_parent_dir) and os.listdir(session_tars_parent_dir):
         print(f"Found existing directory {session_tars_parent_dir} (expected to contain session tars). Checking content...")
         if any(f.endswith(".tar") for f in os.listdir(session_tars_parent_dir)):
            needs_allData_tar_extraction = False
            print(f"Session .tar files seem to exist in {session_tars_parent_dir}. Skipping extraction of {INNER_TAR_NAME}.")
         else:
            print(f"Directory {session_tars_parent_dir} exists but no .tar files found. Will attempt to extract {INNER_TAR_NAME}.")
    
    if needs_allData_tar_extraction:
        print(f"Extracting {inner_tar_path} (this is '{INNER_TAR_NAME}') into {TEMP_DOWNLOAD_DIR}...")
        # This extraction should create the 'allData' directory which is session_tars_parent_dir
        if not extract_tar(inner_tar_path, TEMP_DOWNLOAD_DIR): 
            print(f"Failed to extract {inner_tar_path}. Aborting.")
            return
        if not os.path.exists(session_tars_parent_dir) or not os.listdir(session_tars_parent_dir):
            print(f"Error: Directory {session_tars_parent_dir} (from '{SESSION_TAR_SOURCE_DIR_NAME}') not found or empty after extracting {INNER_TAR_NAME}.")
            print(f"Contents of {TEMP_DOWNLOAD_DIR}: {os.listdir(TEMP_DOWNLOAD_DIR)}")
            # Check if individual .tar files were extracted directly into TEMP_DOWNLOAD_DIR
            if any(f.endswith(".tar") and f != INNER_TAR_NAME for f in os.listdir(TEMP_DOWNLOAD_DIR)):
                print(f"It seems session .tar files were extracted directly into {TEMP_DOWNLOAD_DIR}, instead of into an '{SESSION_TAR_SOURCE_DIR_NAME}' subfolder. Adjusting path.")
                session_tars_parent_dir = TEMP_DOWNLOAD_DIR # Adjust path for next step
            else:
                print(f"Cannot locate session .tar files after extracting {INNER_TAR_NAME}. Aborting.")
                return
    
    # 4. Extract each individual session .tar file into FINAL_SESSIONS_DIR/session_name/
    print(f"Looking for individual session .tar files in: {session_tars_parent_dir}")
    if not os.path.isdir(session_tars_parent_dir):
        print(f"Error: Expected {session_tars_parent_dir} to be a directory containing session .tar files. Aborting.")
        return

    session_tar_files = [f for f in os.listdir(session_tars_parent_dir) if f.endswith(".tar") and f != INNER_TAR_NAME]
    if not session_tar_files:
        print(f"No individual session .tar files (e.g. Cori_2016-12-18.tar) found in {session_tars_parent_dir}.")
        print(f"Contents of {session_tars_parent_dir}: {os.listdir(session_tars_parent_dir)}")
        print("Please check the archive structure. Aborting.")
        return
        
    print(f"Found {len(session_tar_files)} individual session .tar files to process.")

    for tar_filename in session_tar_files:
        full_tar_path = os.path.join(session_tars_parent_dir, tar_filename)
        session_name = os.path.splitext(tar_filename)[0] 
        session_extract_path = os.path.join(FINAL_SESSIONS_DIR, session_name)

        if os.path.exists(session_extract_path) and os.listdir(session_extract_path):
            expected_files_exist = False
            # Check for a couple of common file types found in session folders
            if (os.path.exists(os.path.join(session_extract_path, '_ss_lfp.timestamps.npy')) or \
                os.path.exists(os.path.join(session_extract_path, 'spikes.times.npy')) or \
                os.path.exists(os.path.join(session_extract_path, 'trials.feedbackType.npy')) ): # Added another common file
                expected_files_exist = True
            
            if expected_files_exist:
                print(f"Session data for {session_name} looks already extracted to {session_extract_path}. Skipping.")
                continue
            else:
                print(f"Directory {session_extract_path} exists but seems incomplete or empty. Re-extracting {session_name}.")

        os.makedirs(session_extract_path, exist_ok=True)
        
        print(f"Processing session: {session_name} from {tar_filename}")
        if not extract_tar(full_tar_path, session_extract_path): # Extract each session tar into its own folder
            print(f"Failed to extract {tar_filename} for session {session_name}. Continuing...")
            try:
                shutil.rmtree(session_extract_path)
                print(f"Cleaned up partially extracted directory: {session_extract_path}")
            except Exception as e_clean:
                print(f"Error cleaning up {session_extract_path}: {e_clean}")

    print(f"Note: Temporary directory {TEMP_DOWNLOAD_DIR} (with the original ZIP and intermediate files) has not been deleted.")
    print(f"\nData extraction finished.")
    print(f"Raw session data should be in: {os.path.abspath(FINAL_SESSIONS_DIR)}")
    print(f"Please update 'dataroot' in 'steinmetz_NMA.ipynb' to point to this directory.")
    print(f"Example: dataroot = r'{os.path.abspath(FINAL_SESSIONS_DIR)}'")

if __name__ == "__main__":
    main()