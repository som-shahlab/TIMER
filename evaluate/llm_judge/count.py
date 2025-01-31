import json

def count_instructions(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Get the detailed_results list and count its length
        num_instructions = len(data.get('detailed_results', []))
        
        print(f"Number of instructions in the file: {num_instructions}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    file_path = "path/to/your/file.json"
    count_instructions(file_path)
