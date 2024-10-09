import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings

import sys
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import chardet
import requests
import os
import tempfile
import hashlib
from tqdm import tqdm  # For progress bar
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_tfidf

# Define color codes
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"
RESET = "\033[0m"

# Function to get color based on similarity score
def get_color(score):
    if score >= 80:
        return RED
    elif score >= 50:
        return YELLOW
    else:
        return GREEN

# Check if the correct number of arguments are provided
if len(sys.argv) != 3:
    print("Usage: python3 chkpay.py <file1.txt or URL> <file2.txt or URL>")
    sys.exit(1)  # Exit the script with an error code

# Get the file paths from command-line arguments
input1 = sys.argv[1]
input2 = sys.argv[2]

# Function to detect if input is a URL
def is_url(path):
    return path.startswith('http://') or path.startswith('https://')

# Function to download content from a URL to a temporary file
def download_url_content(url):
    try:
        # Announce that a URL is detected
        print(f"{CYAN}URL detected: {url}{RESET}")

        # Parse the URL to get the filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = 'downloaded_file'

        # Create a unique temporary file without using the original filename
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='wb')
        temp_file_path = temp_file.name

        # Send a GET request with stream=True to download in chunks
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get total file size from headers
        total_size = response.headers.get('content-length')
        if total_size is None:
            total_size = 0
        else:
            total_size = int(total_size)

        # Download the file with a progress bar
        block_size = 1024  # 1 KiB

        if total_size == 0:
            progress_bar = tqdm(unit='iB', unit_scale=True, desc="Downloading content")
        else:
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading content")

        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            temp_file.write(data)
        temp_file.close()
        progress_bar.close()

        # Remove the check for incomplete download
        # Proceed without checking if progress_bar.n == total_size

        return temp_file_path, filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading content from URL {url}: {e}")
        sys.exit(1)

# Initialize dictionaries to keep track of temporary files
temp_files = {}

# Process input1
if is_url(input1):
    file1_path, file1_filename = download_url_content(input1)
    temp_files[file1_path] = True  # Mark as temporary
    file1_display = file1_filename  # For display purposes
else:
    file1_path = input1
    temp_files[file1_path] = False  # Not temporary
    file1_display = os.path.basename(file1_path)

# Process input2
if is_url(input2):
    file2_path, file2_filename = download_url_content(input2)
    temp_files[file2_path] = True  # Mark as temporary
    file2_display = file2_filename  # For display purposes
else:
    file2_path = input2
    temp_files[file2_path] = False  # Not temporary
    file2_display = os.path.basename(file2_path)

# Function to compute SHA256 checksum of a file
def compute_sha256(file_path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        # Clean up temporary files before exiting
        for path, is_temp in temp_files.items():
            if is_temp and os.path.exists(path):
                os.remove(path)
        sys.exit(1)
    return sha256_hash.hexdigest()

# Declare that checksums are being computed (in CYAN)
print(f"{CYAN}Checking SHA256 checksums...{RESET}")

# Compute SHA256 checksums
checksum1 = compute_sha256(file1_path)
checksum2 = compute_sha256(file2_path)

# Display the checksums
print(f"\nChecksum for '{file1_display}':")
print(f"{GREEN}{checksum1}{RESET}")
print(f"\nChecksum for '{file2_display}':")
print(f"{GREEN}{checksum2}{RESET}\n")

# Check if checksums are the same
if checksum1 == checksum2:
    print(f"{RED}The files are the exact same. (SHA256 checksums match){RESET}")
    # Clean up temporary files before exiting
    for path, is_temp in temp_files.items():
        if is_temp and os.path.exists(path):
            os.remove(path)
    sys.exit(0)  # Exit the script successfully

# Declare that files are being compared (in CYAN)
print(f"{CYAN}Checksums differ. Proceeding to compare files...{RESET}")

# Display the filenames of the files being compared
print(f"{CYAN}Comparing files: '{file1_display}' and '{file2_display}'{RESET}\n")

# Function to detect the encoding of a file
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        rawdata = file.read(10000)
        result = chardet.detect(rawdata)
        return result['encoding']

# Load and read the text files line by line
def load_text(file_path):
    encoding = detect_encoding(file_path)
    ignore_symbols = ('#', '//', ';', '/*', '*/')

    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
        lines = []
        for line in file.readlines():
            stripped_line = line.strip()
            if (
                stripped_line and
                len(stripped_line) > 8 and
                not any(stripped_line.startswith(symbol) for symbol in ignore_symbols)
            ):
                lines.append(stripped_line)
    # Delete the temporary file after reading if it's a temporary file
    if temp_files.get(file_path, False):
        os.remove(file_path)
    return lines

# Load text from both files
lines1 = load_text(file1_path)
lines2 = load_text(file2_path)

# Remove temporary files after loading
for path, is_temp in list(temp_files.items()):
    if is_temp and os.path.exists(path):
        os.remove(path)
    temp_files.pop(path)

# Ensure the file with the most lines is lines1
if len(lines2) > len(lines1):
    lines1, lines2 = lines2, lines1
    file1_display, file2_display = file2_display, file1_display

# Join lines into single strings for TF-IDF
doc1 = ' '.join(lines1)
doc2 = ' '.join(lines2)

# Compute TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

# Compute Cosine Similarity for TF-IDF vectors
tfidf_similarity = cosine_similarity_tfidf(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
tfidf_similarity_percentage = round(tfidf_similarity * 100)

# Create ASCII scroll borders
scroll_width = 80  # Increased total width by 2
inner_width = scroll_width - 2  # Width inside the borders
scroll_top = "+" + "-" * (scroll_width - 2) + "+"
scroll_bottom = "+" + "-" * (scroll_width - 2) + "+"
scroll_line = "|" + " " * (scroll_width - 2) + "|"

# Helper function to strip ANSI color codes for alignment
import re
ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

def visible_length(s):
    return len(ansi_escape.sub('', s))

# Load pre-trained model from Hugging Face
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode lines into embeddings
embeddings1 = model.encode(lines1, convert_to_tensor=True)
embeddings2 = model.encode(lines2, convert_to_tensor=True)

# Calculate similarity scores
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

# Store results
detailed_results = []
best_matches = []
significant_matches = []

# Iterate through each line in the longer file (lines1)
for i in range(len(lines1)):
    # Calculate similarity with all lines in the shorter file (lines2)
    similarities = cosine_scores[i]
    best_match_idx = torch.argmax(similarities)
    similarity_score = round(similarities[best_match_idx].item() * 100)
    best_matches.append(similarity_score)

    if similarity_score >= 60:
        significant_matches.append(similarity_score)

    if similarity_score >= 50:
        color = get_color(similarity_score)
        line_1_text = f'Line from {file1_display}: "{lines1[i]}"'
        match_text = f'Best match in {file2_display}: "{lines2[best_match_idx]}"'
        score_text = f'Similarity Score: {color}{similarity_score}%{RESET}'

        # Format detailed result ensuring alignment
        # Calculate padding needed for each line
        def format_detail_line(text):
            visible_len = visible_length(text)
            padding = inner_width - visible_len - 1  # Corrected from -2 to -1
            return f"| {text}{' ' * padding}|"

        detailed_result = (
            format_detail_line(line_1_text) + "\n" +
            format_detail_line(match_text) + "\n" +
            format_detail_line(score_text) + "\n" +
            f"|{' ' * inner_width}|"
        )
        detailed_results.append(detailed_result)

# Calculate average similarities
average_similarity = round(np.mean(best_matches)) if best_matches else 0
significant_match_average = round(np.mean(significant_matches)) if significant_matches else 0

# Calculate overall similarity using weighted average including TF-IDF similarity
# Weights: average_similarity (1), significant_match_average (2), tfidf_similarity_percentage (2)
overall_similarity = (average_similarity + 2 * significant_match_average + 2 * tfidf_similarity_percentage) / 5

# Calculate overall score from overall_similarity
def calculate_overall_score(overall_similarity):
    score = (overall_similarity / 100) * 10
    score = round(score * 2) / 2
    return int(score) if score.is_integer() else score

overall_score = calculate_overall_score(overall_similarity)

# Determine the small word summary
def get_word_summary(score):
    if score >= 8:
        return "Definitely"
    elif 6 <= score < 8:
        return "Most likely"
    elif 4 <= score < 6:
        return "It's a possibility"
    elif 2 <= score < 4:
        return "Most likely not"
    else:
        return "Fully Unique"

word_summary = get_word_summary(overall_score)

# Function to get color for overall score
def get_overall_score_color(score):
    if score >= 8:
        return RED
    elif 5 <= score < 8:
        return YELLOW
    else:
        return GREEN

overall_score_color = get_overall_score_color(overall_score)
average_color = get_color(average_similarity)
significant_color = get_color(significant_match_average)
tfidf_color = get_color(tfidf_similarity_percentage)

# Function to get color for the conclusion
def get_conclusion_color(word_summary):
    if word_summary == "Definitely":
        return RED
    elif word_summary == "Most likely":
        return YELLOW
    elif word_summary == "It's a possibility":
        return YELLOW
    elif word_summary == "Most likely not":
        return GREEN
    else:  # "Fully Unique"
        return GREEN

conclusion_color = get_conclusion_color(word_summary)

# Helper functions for alignment
def format_scroll_line_right(message, score=None, color='', reset=''):
    if score is not None:
        score_text = f"{color}{score}%{reset}"
        msg_len = visible_length(message)
        score_len = visible_length(score_text)
        total_len = msg_len + score_len + 1  # +1 for space padding
        # Move result numbers 2 characters to the left
        spaces_needed = inner_width - total_len - 2
        if spaces_needed < 0:
            spaces_needed = 0
        line_content = f" {message}{' ' * spaces_needed}{score_text}"
    else:
        line_content = f" {message}"
    return f"|{line_content}{' ' * 2}|"

def format_score_line_right(message, score, color='', reset=''):
    score_str = f"{color}{score}{reset}"
    msg_len = visible_length(message)
    score_len = visible_length(score_str)
    total_len = msg_len + score_len + 1  # +1 for space padding
    # Move result numbers 2 characters to the left
    spaces_needed = inner_width - total_len - 2
    if spaces_needed < 0:
        spaces_needed = 0
    line_content = f" {message}{' ' * spaces_needed}{score_str}"
    return f"|{line_content}{' ' * 2}|"

# Function to center the conclusion line with colored conclusion
def format_conclusion_line(message, conclusion, color='', reset=''):
    conclusion_text = f"{color}{conclusion}{reset}"
    full_message = f"{message} {conclusion_text}"
    total_len = visible_length(full_message)
    spaces_needed = inner_width - total_len
    left_padding = spaces_needed // 2
    right_padding = spaces_needed - left_padding
    line_content = f"{' ' * left_padding}{full_message}{' ' * right_padding}"
    return f"|{line_content}|"

# Print summary section
print(scroll_top)
print(f"|{'Summary':^{inner_width}}|")
print(scroll_line)
summary_line1 = format_scroll_line_right("Average similarity between files:", average_similarity, average_color, RESET)
summary_line2 = format_scroll_line_right("Average similarity of matching lines:", significant_match_average, significant_color, RESET)
summary_line3 = format_scroll_line_right("Text & Keyword Similarity:", tfidf_similarity_percentage, tfidf_color, RESET)
blank_line = f"|{' ' * inner_width}|"
overall_score_line = format_score_line_right("Overall Similarity Score:", overall_score, overall_score_color, RESET)
conclusion_line = format_conclusion_line("Conclusion:", word_summary, conclusion_color, RESET)
print(summary_line1)
print(summary_line2)
print(summary_line3)
print(blank_line)
print(overall_score_line)
print(scroll_line)
print(conclusion_line)
print(scroll_bottom)

# Print the legend below the scroll
legend_text = f"    Legend: [{RED}Red ≥80%{RESET}], [{YELLOW}Yellow 50-79%{RESET}], [{GREEN}Green <50%{RESET}]"
print()
print(legend_text)

# Ask user for suspicious lines
show_details = input("\nWould you like to see suspicious lines? (yes/no): ").strip().lower()

if show_details in ['yes', 'y']:
    if detailed_results:
        print("\n" + scroll_top)
        print(f"|{'Suspicious Lines':^{inner_width}}|")
        print(scroll_line)
        for result in detailed_results:
            print(result)
        print(scroll_bottom)
    else:
        print("\nNo suspicious lines found.")

