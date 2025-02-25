import filelock, json, os, zipfile

# ----------------------------------------------------------------------------
# Basic IO helpers


def ensure_dir_exists(name: str):
    """Make sure that a directory exists (create it if it doesn't)
    """
    if not os.path.exists(name):
        os.makedirs(name)


def load_json_file(filename: str) -> dict:
    """Load data from a json file
    """
    with open(filename, 'rt') as file:
        return json.load(file)


def save_json_file(data: object, filename: str):
    """Save data to a json file with the style we like (sorted keys,
    multi-line with indentation 2, extra new-line at the end)
    """
    with open(filename, 'wt') as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write('\n')


# ----------------------------------------------------------------------------
# Multi-process-safe file IO

class MultiProcessJsonFile:
    """Helper class for letting several processes write to the same JSON file.
    
    Indended for distributed jobs, where each process writes its own results to
    the same file.

    Usage example:

    json_file = MultiProcessJsonfile('some/path/file.json')
    with json_file.lock() as file_data:
        if not 'mean' in file_data:
            file_data['mean'] = compute_mean(...)
    """
        
    def __init__(self, file_name):
        self.file_name = file_name
        self.lock_file_name = self.file_name + '.lock'
        self.data = None

        with filelock.SoftFileLock(self.lock_file_name):
            if not os.path.exists(self.file_name) or os.path.getsize(self.file_name) == 0:
                file_contents = {}
                save_json_file(file_contents, self.file_name)

    class LockObject:
        def __init__(self, base):
            self.base = base

            self.lock = filelock.SoftFileLock(base.lock_file_name)

        def __enter__(self):
            self.lock.__enter__()
            self.data = load_json_file(self.base.file_name)
            return self.data # Shallow copy

        def __exit__(self, exc_type, exc_val, exc_tb):
            save_json_file(self.data, self.base.file_name)
            self.lock.__exit__(exc_type, exc_val, exc_tb)
            self.lock = None
            
    def lock(self) -> LockObject:
        """Lock the file and allow the user to edit the data.

        Intended to be used in a 'with' statement. See class-level comment for
        details.
        """
        return self.LockObject(self)


class MultiProcessZipFile:
    """Simple class representing a multi-process-safe zip file.
    
    Several processes can add files to the zip file independently, as long
    as they don't use the same file name / path within the archive. Also
    supports splitting the output into several output files on the fly, to
    avoid quadratic write complexity when the file grows large.
    """

    def __init__(self, file_name_base: str, files_per_zip: int):
        """Create a multi-process zip file
        
        Args:
          file_name_base: File name (and path) without the 'zip' suffix. Actual
            file names will be {file_name_base}_part{N}.zip
          files_per_zip: Maximum number of files per zip file, before
            swithcing to the next file.
        """
        
        self.file_name_base = file_name_base
        self.lock_file_name = file_name_base + '.lock'
        self.files_per_zip = files_per_zip
        self.counter = 0
    
    def write(self, input_path: str, zip_path: str):
        """Add a file to the archive.

        Args:
          input_path: Path to the file to read
          zip_path: Relative path to the file within the zip file

        Raises:
          Exception: If the file to add was already present in the zip file.
        """

        if self.files_per_zip > 0:
            part_no = self.counter // self.files_per_zip
            file_name = f"{self.file_name_base}_part{part_no}.zip"
        else:
            file_name = f"{self.file_name_base}.zip"

        with filelock.SoftFileLock(self.lock_file_name):
            zip = zipfile.ZipFile(file_name, 'a')
            if zip_path in zip.namelist():
                raise Exception(f'Name already present in zip file: {zip_path}')
            zip.write(input_path, zip_path)
            zip.close()
        
        self.counter += 1
