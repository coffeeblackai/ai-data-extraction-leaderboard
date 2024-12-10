import os
from bs4 import BeautifulSoup

def clean_html_file(file_path):
    # Read the original HTML content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse the HTML content
    soup = BeautifulSoup(content, 'html.parser')

    # Remove <script>, <style>, and <link> tags
    for tag_name in ['script', 'style', 'link']:
        for t in soup.find_all(tag_name):
            t.decompose()

    # Remove 'id' and 'class' attributes from all tags
    for tag in soup.find_all(True):
        if 'id' in tag.attrs:
            del tag.attrs['id']
        if 'class' in tag.attrs:
            del tag.attrs['class']

    # Convert the cleaned soup back to a string
    cleaned_html = str(soup)

    # Write the cleaned HTML back to the same file (overwrites original)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_html)

def clean_directory_of_html(directory):
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith('.html'):
            file_path = os.path.join(directory, filename)
            clean_html_file(file_path)
            print(f"Cleaned: {filename}")

if __name__ == '__main__':
    # Example usage:
    input_directory = 'dataset/input'
    clean_directory_of_html(input_directory)
