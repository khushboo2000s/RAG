import pandas as pd
import logging

# Function to read Excel files
def read_excel(file_path):
    try:
        # Read the Excel file
        excel_data = pd.read_excel(file_path)  # Directly read the first sheet
        logging.info(f"Excel file {file_path} read successfully. Number of rows: {len(excel_data)}")
        
        # Convert rows into a single string
        text = "\n".join(excel_data.astype(str).apply(" ".join, axis=1))  # Convert each row to a string
        return text
    except Exception as e:
        logging.error(f"Error reading Excel file {file_path}: {e}")
        return ""

# Main function for testing
if __name__ == "__main__":
    # Hardcoded file path for testing
    file_path = "D:\\Genai_project\\Retrieval Augmented Generation\\rag_final\\RAG_task\\data_files\\students.xlsx"
    
    # Output file to save the results
    output_file = "D:\\Genai_project\\Retrieval Augmented Generation\\rag_final\\RAG_task\\data_files\\output.txt"
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Read the Excel file and get the output
    output = read_excel(file_path)
    
    # Write the output to a file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        logging.info(f"Output successfully written to {output_file}")
    except Exception as e:
        logging.error(f"Error writing to output file {output_file}: {e}")
    
    # Confirmation message
    if output:
        print(f"The output has been saved to {output_file}")
    else:
        print("No data found or an error occurred. Check logs for details.")
