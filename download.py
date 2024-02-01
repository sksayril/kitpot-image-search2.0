import gdown

# Replace the Google Drive file ID in the URL below with your file's ID
file_url = 'https://drive.google.com/file/d/1239CFg6hwhevnA10qJKugYC76Fkn3cz-/view?usp=sharing'

# Define the output file path
output_path = 'res_vector_embeddings.pkl'  # Replace 'downloaded_file.ext' with your desired file name

# Download the file
gdown.download(file_url, output_path, quiet=False)
