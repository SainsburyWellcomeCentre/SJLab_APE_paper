import pandas as pd
import os
import glob
from set_global_params import spreadsheet_path

def log_empty_file(file_path):
    """Logs the file path to empty_files.log if the dataframe is empty."""
    with open("empty_files.log", "a") as log_file:
        log_file.write(f"Empty file: {file_path}\n")

def create_excel_files_from_folders(root_folder):
    # Get all subfolders in the root directory
    figure_folders = [f for f in os.listdir(root_folder)
                      if os.path.isdir(os.path.join(root_folder, f))]

    for figure_folder in figure_folders:
        # Full path to this figure folder
        figure_path = os.path.join(root_folder, figure_folder)

        # Get all CSV and XLSX files, and sort them together
        all_files = glob.glob(os.path.join(figure_path, "*.csv")) + \
                    glob.glob(os.path.join(figure_path, "*.xlsx"))
        all_files.sort(key=lambda x: os.path.basename(x).lower())  # Case-insensitive sort

        # Skip if no files found
        if not all_files:
            print(f"No CSV or XLSX files found in {figure_folder}, skipping...")
            continue

        # Create output Excel filename based on folder name
        output_excel_file = os.path.join(root_folder, f'data_sheet_{figure_folder}.xlsx')

        # Create Excel writer object
        with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
            for file_path in all_files:
                file_ext = os.path.splitext(file_path)[1].lower()

                try:
                    if file_ext == '.csv':
                        df = pd.read_csv(file_path)
                        if df.empty:
                            log_empty_file(file_path)

                        sheet_name = os.path.basename(file_path).split('.')[0]
                        if len(sheet_name) > 31:
                            sheet_name = sheet_name[:31]

                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"Added {sheet_name} from CSV to {output_excel_file}")

                    elif file_ext == '.xlsx':
                        xls = pd.ExcelFile(file_path)
                        for sheet in xls.sheet_names:
                            df = xls.parse(sheet)
                            if df.empty:
                                log_empty_file(f"{file_path} - sheet: {sheet}")

                            sheet_name = f"{os.path.basename(file_path).split('.')[0]}_{sheet}"
                            if len(sheet_name) > 31:
                                sheet_name = sheet_name[:31]

                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                            print(f"Added {sheet_name} from XLSX to {output_excel_file}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        print(f"Successfully created Excel file: {output_excel_file}")

# Main execution
root_folder = spreadsheet_path  # Your root folder
create_excel_files_from_folders(root_folder)
