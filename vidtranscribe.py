# Importing packages that are already installed
import subprocess
import platform
import os
import sys
from datetime import datetime

# Checking type of OS
os_name = platform.system()
architecture_name = os.uname().machine

# Created address variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_DIR = os.path.join(BASE_DIR, "whisper.cpp")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
os.makedirs(TESTS_DIR, exist_ok=True)

# Function to install libraries
def install_libraries():
    base_libraries = ["Pillow", "numpy==1.23.5", "torch==2.2.0", "transformers", "openai", "tk", "requests"]
    macos_arm_libraries = ["coremltools", "ane_transformers"]
    if os_name == "Darwin" and "arm" in architecture_name:
        required_libraries = base_libraries + macos_arm_libraries
    else:
        required_libraries = base_libraries
    subprocess.check_call([sys.executable, "-m", "pip", "install", *required_libraries])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/whisper.git"])
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to install Whisper library.")
        sys.exit(1)
install_libraries()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Function to install packages
def install_packages():
    if os_name == "Linux":
        subprocess.check_call(["sudo", "apt-get", "install", "-y", "ffmpeg", "ccache", "vlc"])
    elif os_name == "Darwin":
        # Checks if Homebrew is installed, if not install it
        try:
            subprocess.check_call(["brew", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Homebrew is already installed.")
        except subprocess.CalledProcessError:
            print("Homebrew not found. Installing...")
            subprocess.check_call(
                ["/bin/bash", "-c", "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"]
            )
        except FileNotFoundError:
            print("Homebrew not found. Installing...")
            subprocess.check_call(
                ["/bin/bash", "-c", "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"]
            )
        subprocess.check_call(["brew", "install", "ffmpeg", "ccache"])
        # subprocess.check_call(["brew", "install", "--cask", "vlc"])
    elif os_name == "Windows":
        messagebox.showwarning("Manual Installation Required", "Please download and install VLC and ffmpeg manually from VLC Official site and https://www.gyan.dev/ffmpeg/builds/")
        sys.exit(1)
install_packages()

# Clones the whisper.cpp github repo
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

        if sys.platform == "darwin" and "arm" in architecture_name:
            if not os.path.exists(os.path.join(WHISPER_DIR, "models", f"coreml-{model_name}.mlmodel")):
                generate_coreml_model(model_name)
                os.chdir(WHISPER_DIR)
                build_whisper()
                subprocess.run(['make', 'clean'], env=env, check=True)
                env['WHISPER_COREML'] = '1'
                subprocess.run(['make', '-j'], env=env, check=True)

        else:
            build_whisper()
            subprocess.run(['make'], env=env, check=True)

        convert_to_wav(filepath, timestamp)
        extract_text(timestamp, model_name)
        play_video_with_subtitles(filepath, os.path.join(TESTS_DIR, f"{timestamp}.wav.srt"))

def download_model(model_name):
    model_download_script = os.path.join(WHISPER_DIR, "models", "download-ggml-model.sh")
    subprocess.run(["bash", model_download_script, model_name], check=True)

def generate_coreml_model(model_name):
    model_coreml_script = os.path.join(WHISPER_DIR, "models", "generate-coreml-model.sh")
    subprocess.run(["bash", model_coreml_script, model_name], check=True)

def convert_to_wav(filepath, timestamp):
    try:
        output_path = os.path.join(TESTS_DIR, f'{timestamp}.wav')
        subprocess.run(["ffmpeg", "-i", filepath, "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error converting to WAV: {e}")

def extract_text(timestamp, model_name):
    model_path = os.path.join(WHISPER_DIR, "models", f"ggml-{model_name}.bin")
    if not os.path.exists(model_path):
        messagebox.showerror("Error", f"Model file not found: {model_path}")
        return
    try:
        result = subprocess.run([os.path.join(WHISPER_DIR, "main"), "-m", model_path, "-f", os.path.join(TESTS_DIR, f"{timestamp}.wav"), "-osrt"], capture_output=True, text=True, check=True)
        with open(os.path.join(TESTS_DIR, f"{timestamp}.srt"), 'w') as output_file:
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

    for model in models:
        tk.Radiobutton(
            radio_frame,
            text=model,
            variable=model_var,
            value=model,
            bg="white",
            fg=text_color,
            font=font_style,
        ).pack(anchor='w')

    browse_button = tk.Button(root, image=browse_photo, command=browse_file, bd=0)
    browse_button.place(relx=0.21, rely=0.73, anchor='center')

    root.mainloop()
