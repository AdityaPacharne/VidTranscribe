import subprocess
import platform
import os
import shutil
import sys
from datetime import datetime

os_name = platform.system()
if os_name == "Linux":
    subprocess.check_call(["sudo", "apt-get", "install", "-y", "python3-tk"])

def install_libraries():
    required_libraries = ["Pillow", "transformers", "openai", "coremltools", "ane_transformers", "tk", "requests"]
    for i in required_libraries:
        subprocess.check_call([sys.executable, "-m", "pip", "install", i])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/whisper.git"])
install_libraries()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import requests

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        os_name = platform.system()
        if os_name == "Linux":
            subprocess.check_call(["sudo", "apt-get", "install", "-y", "ffmpeg"])
        elif os_name == "Darwin":
            subprocess.check_call(["/bin/bash", "-c", "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"])
            subprocess.check_call(["brew", "install", "ffmpeg"])
        elif os_name == "Windows":
            messagebox.showwarning("Manual Installation Required", "ffmpeg is not installed. Please download and install ffmpeg manually from https://www.gyan.dev/ffmpeg/builds/")
            sys.exit(1)
        else:
            messagebox.showerror("Error", "Unsupported OS for automatic ffmpeg installation. Please install ffmpeg manually.")
            sys.exit(1)
check_ffmpeg()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_DIR = os.path.join(BASE_DIR, "whisper.cpp")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
os.makedirs(TESTS_DIR, exist_ok = True)

def clone_whisper():
    if not os.path.exists(WHISPER_DIR):
        subprocess.run(["git", "clone", "https://github.com/ggerganov/whisper.cpp.git", WHISPER_DIR], check=True)

def browse_file():
    filepath = filedialog.askopenfilename()
    if filepath:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        model_name = model_var.get()
        if model_name:
            download_model(model_name)
            if sys.platform == "darwin" and "arm" in os.uname().machine:
                generate_coreml_model(model_name)
                build_whisper()
        convert_to_wav(filepath, timestamp)
        extract_text(timestamp, model_name)
        play_video_with_subtitles(filepath, os.path.join(TESTS_DIR, f"{timestamp}.wav.srt"))
    else:
        messagebox.showwarning("No File Selected", "No file selected.")

def download_model(model_name):
    model_download_script = os.path.join(WHISPER_DIR, "models", "download-ggml-model.sh")
    result = subprocess.run(["bash", model_download_script, model_name], check=True)
    if result.returncode != 0:
        messagebox.showerror("Error", f"Error downloading model: {model_name}")

def generate_coreml_model(model_name):
    model_coreml_script = os.path.join(WHISPER_DIR, "models", "generate-coreml-model.sh")
    result = subprocess.run(["bash", model_coreml_script, model_name], check=True)
    if result.returncode != 0:
        messagebox.showerror("Error", f"Error generating Core ML model: {model_name}")

def build_whisper():
    os.chdir(WHISPER_DIR)
    result = subprocess.run(["make", "clean"], check=True)
    if result.returncode != 0:
        messagebox.showerror("Error", "Error building whisper.cpp")

    env = os.environ.copy()
    env['WHISPER_COREML'] = '1'
    another_result = subprocess.run(['make', '-j'], env=env, check=True)
    if another_result.returncode != 0:
        messagebox.showerror("Error", "Error compiling whisper.cpp with parallel jobs")

def convert_to_wav(filepath, timestamp):
    try:
        output_path = os.path.join(TESTS_DIR, f'{timestamp}.wav')
        subprocess.run(["ffmpeg", "-i", filepath, "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error converting to WAV: {e}")

def extract_text(timestamp, model_name):
    output_file = os.path.join(TESTS_DIR, f"{timestamp}.srt")
    model_path = os.path.join(WHISPER_DIR, "models", f"ggml-{model_name}.bin")
    
    if not os.path.exists(model_path):
        messagebox.showerror("Error", f"Model file not found: {model_path}")
        return
    
    with open(output_file, 'w') as output:
        try:
            subprocess.run([os.path.join(WHISPER_DIR, "main"), "-m", model_path, "-f", os.path.join(TESTS_DIR, f"{timestamp}.wav"), "-osrt"], stdout=output, check=True)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Error extracting text: {e}")

def play_video_with_subtitles(video_path, subtitles_path):
    try:
        subprocess.run(["open", "-a", "VLC", video_path, "--args", "--sub-file", subtitles_path], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error playing video with subtitles: {e}")

if __name__ == "__main__":
    clone_whisper()

    models = ['tiny.en', 'base.en', 'small.en', 'medium.en', 'large.en']

    root = tk.Tk()
    window_width = 800  
    window_height = 600  
    root.geometry(f"{window_width}x{window_height}")  

    image = Image.open(os.path.join(BASE_DIR, "image.jpg"))
    photo = ImageTk.PhotoImage(image)
    background_label = tk.Label(root, image=photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    model_var = tk.StringVar(value=models[0])
    model_label = tk.Label(root, text="Select Model:")
    model_label.place(relx=0.5, rely=0.5, anchor='center', y=-40)
    model_dropdown = ttk.Combobox(root, textvariable=model_var, values=models, state='readonly')
    model_dropdown.place(relx=0.5, rely=0.5, anchor='center', y=-20)

    browse_button = tk.Button(root, text="Browse", command=browse_file)
    browse_button.place(relx=0.5, rely=0.7, anchor='center')  

    root.mainloop()
