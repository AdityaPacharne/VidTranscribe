import subprocess
import platform
import os
import sys
from datetime import datetime
import re

os_name = platform.system()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_DIR = os.path.join(BASE_DIR, "whisper.cpp")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
os.makedirs(TESTS_DIR, exist_ok=True)

def install_libraries():
    base_libraries = ["Pillow", "numpy==1.23.5", "torch==2.2.0", "transformers", "openai", "tk", "requests"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *base_libraries])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/whisper.git"])
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to install Whisper library.")
        sys.exit(1)
install_libraries()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from transformers import BartTokenizer, BartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')

def install_packages():
    if os_name == "Linux":
        subprocess.check_call(["sudo", "apt-get", "install", "-y", "ffmpeg", "ccache", "vlc"])
    elif os_name == "Darwin":
        subprocess.check_call(["brew", "install", "ffmpeg", "ccache"])
    elif os_name == "Windows":
        messagebox.showwarning("Manual Installation Required", "Please download and install VLC and ffmpeg manually from VLC Official site and https://www.gyan.dev/ffmpeg/builds/")
        sys.exit(1)
install_packages()

def clone_whisper():
    if not os.path.exists(WHISPER_DIR):
        subprocess.run(["git", "clone", "https://github.com/ggerganov/whisper.cpp.git", WHISPER_DIR], check=True)
clone_whisper()

def build_whisper():
    cflags = "-Iggml/include -Iggml/src -Iinclude -Isrc -Iexamples -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -std=c11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -pthread -fopenmp -Wdouble-promotion -mcpu=native"
    cxxflags = "-std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -fopenmp -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Iexamples -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -mcpu=native"

    os.chdir(WHISPER_DIR)
    command = (
        f'export CFLAGS="{cflags}"; '
        f'export CXXFLAGS="{cxxflags}"; '
    )
    subprocess.run(command, shell=True, check=True)

def browse_file():
    filepath = filedialog.askopenfilename()

    if filepath:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_name = model_var.get()
        model_name = f"{short_name}.en"
        model_path = os.path.join(WHISPER_DIR, "models", f"ggml-{model_name}.bin")

        if not os.path.exists(model_path):
            download_model(model_name)

        env = os.environ.copy()

        build_whisper()
        subprocess.run(['make'], env=env, check=True)

        temp_srt_file = os.path.join(TESTS_DIR, f"{timestamp}/{timestamp}.wav.srt")
        temp_timestamp_dir = os.path.join(TESTS_DIR, f"{timestamp}")

        convert_to_wav(filepath, timestamp)
        extract_text(timestamp, model_name)
        play_video_with_subtitles(filepath, temp_srt_file)

        combined_text = extract_combined_text_from_srt(temp_srt_file)

        inputs = tokenizer(combined_text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        os.chdir(temp_timestamp_dir)
        with open("summary.txt", 'w', encoding='utf-8') as file:
            content = file.write(summary)
        subprocess.check_call(["open", "."])


def download_model(model_name):
    model_download_script = os.path.join(WHISPER_DIR, "models", "download-ggml-model.sh")
    subprocess.run(["bash", model_download_script, model_name], check=True)

def convert_to_wav(filepath, timestamp):
    try:
        os.chdir(TESTS_DIR)
        os.makedirs(f'{timestamp}')
        output_path = os.path.join(TESTS_DIR, f'{timestamp}/{timestamp}.wav')
        subprocess.run(["ffmpeg", "-i", filepath, "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error converting to WAV: {e}")

def extract_text(timestamp, model_name):
    model_path = os.path.join(WHISPER_DIR, "models", f"ggml-{model_name}.bin")
    if not os.path.exists(model_path):
        messagebox.showerror("Error", f"Model file not found: {model_path}")
        return
    try:
        result = subprocess.run([os.path.join(WHISPER_DIR, "main"), "-m", model_path, "-f", os.path.join(TESTS_DIR, f"{timestamp}/{timestamp}.wav"), "-osrt"], capture_output=True, text=True, check=True)
        with open(os.path.join(TESTS_DIR, f"{timestamp}/{timestamp}.srt"), 'w') as output_file:
            output_file.write(result.stdout)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error extracting text: {e}")

def play_video_with_subtitles(video_path, subtitles_path):
    try:
        if os_name == "Darwin":
            subprocess.run(["open", "-a", "VLC", video_path, "--args", "--sub-file", subtitles_path], check=True)
        elif os_name == "Linux":
            subprocess.run(["vlc", video_path, "--sub-file", subtitles_path], check=True)
        else:
            raise EnvironmentError("Unsupported operating system")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error playing video with subtitles: {e}")
    except EnvironmentError as e:
        messagebox.showerror("Error", str(e))

def extract_combined_text_from_srt(file_path):
    combined_text = ''

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
      
    content = re.sub(r'\d+\s+\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}', '', content)
    lines = content.strip().split('\n')
  
    for line in lines:
        line = line.strip()
        if line:
            combined_text += ' ' + line

    combined_text = combined_text.strip()
    return combined_text

if __name__ == "__main__":

    root = tk.Tk()
    root.title("VidTranscribe")
    window_width = 1000
    window_height = 600
    root.geometry(f"{window_width}x{window_height}")

    background_image = Image.open("images/background.jpg")
    background_image = background_image.resize((window_width, window_height))
    background_photo = ImageTk.PhotoImage(background_image)

    background_label = tk.Label(root, image=background_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    dropdown_image = Image.open("images/dropdown.jpg")
    dropdown_image = dropdown_image.resize((260, 60))
    dropdown_photo = ImageTk.PhotoImage(dropdown_image)

    browse_image = Image.open("images/browse.jpg")
    browse_image = browse_image.resize((150, 40))
    browse_photo = ImageTk.PhotoImage(browse_image)

    model_label = tk.Label(root, image=dropdown_photo, bd=0)
    model_label.place(relx=0.252, rely=0.4, anchor='center', y=-40)
    
    radio_frame = tk.Frame(root, bg="white", bd=0)
    radio_frame.place(relx=0.2, rely=0.57, anchor='center', y=-20)

    font_style = ('Courier', 23, 'bold')
    text_color = 'black'

    models = ['tiny', 'base', 'small', 'medium', 'large']
    model_var = tk.StringVar(value=models[0])

    for model_name in models:  # Renamed 'model' to 'model_name'
        tk.Radiobutton(
            radio_frame,
            text=model_name,
            variable=model_var,
            value=model_name,
            bg="white",
            fg=text_color,
            font=font_style,
        ).pack(anchor='w')

    browse_button = tk.Button(root, image=browse_photo, command=browse_file, bd=0)
    browse_button.place(relx=0.21, rely=0.73, anchor='center')

    root.mainloop()
