import subprocess
import platform
import os
import sys
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
from PIL import Image, ImageTk
from multiprocessing import Process

os_name = platform.system()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_DIR = os.path.join(BASE_DIR, "whisper.cpp")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
os.makedirs(TESTS_DIR, exist_ok=True)

def install_libraries():
    base_libraries = ["Pillow", "transformers", "openai", "tk", "requests"]
    macos_arm_libraries = ["coremltools", "ane_transformers"]
    machine_type = os.uname().machine
    required_libraries = base_libraries + macos_arm_libraries if os_name == "Darwin" and "arm" in machine_type else base_libraries
    subprocess.check_call([sys.executable, "-m", "pip", "install", *required_libraries])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/whisper.git"])
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to install Whisper library.")
        sys.exit(1)

def install_ffmpeg():
    if os_name == "Linux":
        subprocess.check_call(["sudo", "apt-get", "install", "-y", "ffmpeg", "ccache", "vlc"])
    elif os_name == "Darwin":
        subprocess.check_call(["brew", "install", "ffmpeg", "ccache"])
        subprocess.check_call(["brew", "install", "--cask", "vlc"])
    elif os_name == "Windows":
        messagebox.showwarning("Manual Installation Required", "Please download and install ffmpeg manually from https://www.gyan.dev/ffmpeg/builds/")
        sys.exit(1)
    else:
        messagebox.showerror("Error", "Unsupported OS for automatic ffmpeg installation.")
        sys.exit(1)

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        install_ffmpeg()

def clone_whisper():
    if not os.path.exists(WHISPER_DIR):
        subprocess.run(["git", "clone", "https://github.com/ggerganov/whisper.cpp.git", WHISPER_DIR], check=True)

def ffmpeg_and_cloning():
    check_ffmpeg()
    clone_whisper()

def parallel():
    p1 = Process(target=install_libraries)
    p2 = Process(target=ffmpeg_and_cloning)
    p1.start()
    p2.start()
    p1.join()
    p2.join()

def set_compilation_flags():
    architecture = platform.machine()
    cflags = "-Iggml/include -Iggml/src -Iinclude -Isrc -Iexamples -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -std=c11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -pthread -fopenmp -Wdouble-promotion"
    cxxflags = "-std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -fopenmp -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Iexamples -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP"
    cflags += " -mcpu=native" if architecture == "aarch64" else " -mcpu=cortex-a57"
    cxxflags += " -mcpu=native" if architecture == "aarch64" else " -mcpu=cortex-a57"
    os.environ['CFLAGS'] = cflags
    os.environ['CXXFLAGS'] = cxxflags

def build_whisper():
    set_compilation_flags()
    os.chdir(WHISPER_DIR)
    env = os.environ.copy()
    env['WHISPER_COREML'] = '1'
    subprocess.run(["make", "clean"], env=env, check=True)
    subprocess.run(['make', '-j'], env=env, check=True)

def browse_file():
    filepath = filedialog.askopenfilename()
    if filepath:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_var.get()
        model_path = os.path.join(WHISPER_DIR, "models", f"ggml-{model_name}.bin")
        if not os.path.exists(model_path):
            download_model(model_name)
        if sys.platform == "darwin" and "arm" in os.uname().machine:
            if not os.path.exists(os.path.join(WHISPER_DIR, "models", f"coreml-{model_name}.mlmodel")):
                generate_coreml_model(model_name)
            build_whisper()
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
    parallel()
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

