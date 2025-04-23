import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from dia.model import Dia
import torch
from themes import THEMES

global_model = None
model_lock = threading.Lock()

def load_model(device):
    global global_model
    with model_lock:
        global_model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device)
    return global_model

def unload_model():
    global global_model
    with model_lock:
        global_model = None

def get_model():
    with model_lock:
        return global_model

import queue
import time
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import pygame # Import pygame
import math # For ceiling function
import random # For seed generation
import os # For set_seed compatibility (though not strictly needed here)
import shutil
try:
    from send2trash import send2trash
except ImportError:
    send2trash = None  # Will show an error if used without install

import sys
from pathlib import Path  # Ensure Path available for OUTPUT_DIR

# --- Theme Management ---
class ThemeManager:
    def __init__(self, root):
        self.root = root
        self.theme_names = list(THEMES.keys())
        self.current_theme_index = 0
        self.current_theme = self.theme_names[self.current_theme_index]
        self.style = ttk.Style(self.root)
        self.apply_theme()

    def switch_theme(self):
        self.current_theme_index = (self.current_theme_index + 1) % len(self.theme_names)
        self.current_theme = self.theme_names[self.current_theme_index]
        self.apply_theme()

    def apply_theme(self):
        theme = THEMES[self.current_theme]
        self.style.theme_use('clam')
        self.root.configure(bg=theme["bg"])
        # ttk styles
        self.style.configure('TButton', background=theme["button_bg"], foreground=theme["button_fg"])
        self.style.map('TButton', background=[('active', theme.get("button_active_bg", theme["button_bg"]))], foreground=[('active', theme["button_fg"])])
        self.style.configure('TEntry', fieldbackground=theme["entry_bg"], foreground=theme["entry_fg"], insertcolor=theme["entry_fg"])
        self.style.configure('TFrame', background=theme["widget_bg"])
        self.style.configure('TLabelFrame', background=theme["widget_bg"], foreground=theme["widget_fg"])
        self.style.configure('TLabelFrame.Label', background=theme["widget_bg"], foreground=theme["widget_fg"])
        self.style.configure('TLabel', background=theme["widget_bg"], foreground=theme["widget_fg"])
        self.style.configure('TScale', background=theme["widget_bg"])
        self.style.configure('Horizontal.TScale', background=theme["widget_bg"])
        # Recursively update all children
        self._update_widget_colors(self.root, theme)

    def _update_widget_colors(self, widget, theme):
        # ttk widgets: set style
        if type(widget) is ttk.Button:
            widget.configure(style='TButton')
        elif type(widget) is ttk.Entry:
            widget.configure(style='TEntry')
        elif type(widget) is ttk.Frame:
            widget.configure(style='TFrame')
        elif type(widget) is ttk.LabelFrame:
            pass  # TLabelFrame picks up theme from style
        elif type(widget) is ttk.Label:
            widget.configure(style='TLabel')
        elif type(widget) is ttk.Scale:
            widget.configure(style='TScale')
        # tk widgets: set bg/fg
        elif type(widget) is tk.Frame:
            widget.configure(bg=theme.get("widget_bg", theme["bg"]))
        elif type(widget) is tk.LabelFrame:
            widget.configure(bg=theme.get("widget_bg", theme["bg"]))
        elif type(widget) is tk.Label:
            widget.configure(bg=theme.get("widget_bg", theme["bg"]), fg=theme.get("widget_fg", theme["fg"]))
        elif type(widget) is tk.Button:
            widget.configure(bg=theme.get("button_bg", theme["widget_bg"]), fg=theme.get("widget_fg", theme["fg"]))
        elif type(widget) is tk.Entry:
            widget.configure(bg=theme.get("entry_bg", theme["widget_bg"]), fg=theme.get("widget_fg", theme["fg"]), insertbackground=theme.get("widget_fg", theme["fg"]))
        elif type(widget) is tk.Text:
            widget.configure(bg=theme.get("widget_bg", theme["entry_bg"]), fg=theme.get("widget_fg", theme["fg"]))
        # Recursively update children
        for child in widget.winfo_children():
            self._update_widget_colors(child, theme)
        # Fallback: try to set bg for any widget that supports it
        try:
            widget.configure(bg=theme.get("widget_bg", theme["bg"]))
        except Exception:
            pass

# Determine script directory and output directory
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = Path(sys.executable).parent
else:
    SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Global Variables ---
device = None
model_load_queue = queue.Queue()
generation_queue = queue.Queue()
generated_audio_data = None
generated_sample_rate = 44100 # Nari default seems to be 44100 Hz

# --- Seed Setting Function (from cli.py) ---
def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN (if used)
    # Note: These might impact performance slightly.
    # You might want to make them optional if performance is critical.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        print("Warning: Could not set cuDNN deterministic behavior (likely not available/needed).")
    print(f"Set random seed to: {seed}")


# --- Model Loading ---
def load_model_thread(target_device):
    if not MODEL_AVAILABLE:
        model_load_queue.put(("error", "Dia/model_manager not found."))
        return
    try:
        print("Loading Dia model...")
        model_manager.load_model(target_device)
        print("Model loaded successfully.")
        model_load_queue.put(("success", "Model loaded successfully."))
    except Exception as e:
        print(f"Error loading Dia model: {e}")
        model_load_queue.put(("error", f"Failed to load model: {e}"))

# --- Text Preprocessing ---
def preprocess_text(raw_text: str) -> str:
    lines = raw_text.strip().splitlines()
    processed_lines = []
    speaker_index = 1
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            # Check if line already starts with a speaker tag like [S1], [S2], etc.
            if stripped_line.startswith("[S") and len(stripped_line) > 3 and stripped_line[2].isdigit() and stripped_line[3] == "]":
                processed_lines.append(stripped_line)
                try:
                    # Update speaker_index based on the *explicit* tag found
                    current_speaker = int(stripped_line[2])
                    # Assume next speaker alternates unless specified otherwise
                    speaker_index = 2 if current_speaker == 1 else 1
                except (ValueError, IndexError):
                     # If tag is malformed, fall back to alternating
                     speaker_index = 2 if speaker_index == 1 else 1
            else:
                # Add the speaker tag if missing
                processed_lines.append(f"[S{speaker_index}] {stripped_line}")
                # Alternate speaker for the *next* line
                speaker_index = 2 if speaker_index == 1 else 1
    return "\n".join(processed_lines)

# --- Inference Logic ---
def run_inference_thread(text_input: str, audio_prompt_path: str | None):
    global device, generated_sample_rate
    model = get_model()
    if not model:
         generation_queue.put(("error", "Model is not loaded."))
         return
    if not text_input or text_input.isspace():
        generation_queue.put(("error", "Formatted text input is empty."))
        return
    print(f"Generating audio for text:\n---\n{text_input}\n---")
    
    temp_audio_prompt_file_path = None # Keep track of the prompt file path if created
    prompt_path_for_generate = None

    try:
        if audio_prompt_path and Path(audio_prompt_path).exists():
            try:
                info = sf.info(audio_prompt_path)
                sr, audio_data = sf.read(audio_prompt_path, dtype='float32')
                print(f"Read audio prompt: {audio_prompt_path}, SR: {sr}, Shape: {audio_data.shape}")
                if audio_data is None or audio_data.size == 0 or np.max(np.abs(audio_data)) < 1e-6:
                     print("Warning: Audio prompt seems empty or silent, ignoring.")
                     prompt_path_for_generate = None
                else:
                    if audio_data.ndim > 1:
                        audio_data = np.mean(audio_data, axis=1) # Convert to mono

                    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                        temp_audio_prompt_file_path = f_audio.name # Store path for cleanup
                        sf.write(temp_audio_prompt_file_path, audio_data, sr, subtype='FLOAT')
                        prompt_path_for_generate = temp_audio_prompt_file_path
                        print(f"Created temporary audio prompt file: {temp_audio_prompt_file_path} (orig sr: {sr})")
            except Exception as e:
                print(f"Warning: Could not process audio prompt '{audio_prompt_path}': {e}. Ignoring prompt.")
                prompt_path_for_generate = None
        else:
            if audio_prompt_path: print(f"Warning: Audio prompt file not found: {audio_prompt_path}")
            prompt_path_for_generate = None

        # --- Generation ---
        start_time = time.time()
        print("Starting generation...")
        output_audio_np = None # Initialize
        with torch.inference_mode():
             output_audio_np = model.generate(
                 text_input,
                 max_tokens=3072,
                 cfg_scale=3.0,
                 temperature=1.3,
                 top_p=0.95,
                 use_cfg_filter=True,
                 cfg_filter_top_k=30,
                 use_torch_compile=False,
                 audio_prompt_path=prompt_path_for_generate,
             )
        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")

        # --- Process Output ---
        if output_audio_np is not None and output_audio_np.size > 0:
             # output_audio_np is already a NumPy array
             # Ensure it's mono (if model sometimes outputs stereo)
             if output_audio_np.ndim > 1:
                  # If output has multiple dimensions, check if it looks like [channels, samples] or [samples, channels]
                  # Assuming typical output is [samples] or [channels, samples]
                  if output_audio_np.shape[0] > 1 and output_audio_np.shape[1] > 1: # Check if it's likely multi-channel
                      print(f"Warning: Model output had shape {output_audio_np.shape}, averaging channels to mono.")
                      # Average across the channel dimension (usually axis 0 for TTS models)
                      # Or check shape[0] vs shape[1] to be more robust if needed
                      if output_audio_np.shape[0] < output_audio_np.shape[1]: # Likely [channels, samples]
                          output_audio_np = np.mean(output_audio_np, axis=0)
                      else: # Likely [samples, channels] - less common for raw audio gen
                          output_audio_np = np.mean(output_audio_np, axis=1)
                  else:
                      # Might be a [1, samples] or [samples, 1] array, squeeze it
                      output_audio_np = output_audio_np.squeeze()

             # Ensure float32 type, check for NaNs/Infs
             output_audio_np = output_audio_np.astype(np.float32)
             if np.any(np.isnan(output_audio_np)) or np.any(np.isinf(output_audio_np)):
                 print("Warning: Generated audio contains NaN or Inf values. Clipping.")
                 output_audio_np = np.nan_to_num(output_audio_np) # Replaces NaN with 0, Inf with large floats

             # Ensure it's 1D after processing
             if output_audio_np.ndim != 1:
                  print(f"Warning: Audio data ended up with {output_audio_np.ndim} dimensions after processing. Attempting to squeeze.")
                  output_audio_np = output_audio_np.squeeze()
                  if output_audio_np.ndim != 1:
                       raise ValueError(f"Could not reduce audio data to 1 dimension. Shape: {output_audio_np.shape}")

             generation_queue.put(("success", (generated_sample_rate, output_audio_np))) # Assume fixed SR for now
             print(f"Generated audio shape (after processing): {output_audio_np.shape}")
        else:
             generation_queue.put(("warning", "Generation produced no output or empty array."))
             print("Generation finished, but no valid audio was produced.")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        generation_queue.put(("error", f"Inference failed: {e}"))
    finally:
        # --- Cleanup Temporary Prompt File ---
        # Use the path stored earlier
        if temp_audio_prompt_file_path and Path(temp_audio_prompt_file_path).exists():
            try:
                Path(temp_audio_prompt_file_path).unlink()
                print(f"Deleted temporary audio prompt file: {temp_audio_prompt_file_path}")
            except OSError as e:
                print(f"Warning: Error deleting temporary prompt file {temp_audio_prompt_file_path}: {e}")

        # --- Explicitly Clear PyTorch CUDA Cache ---
        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                print("Cleared PyTorch CUDA cache.")
            except Exception as cache_e:
                print(f"Warning: Failed to clear CUDA cache: {cache_e}")
        # --- End Cache Clearing ---


# --- Tkinter GUI Application ---
class TTSApp:
    def __init__(self, root):
        self.root = root
        # --- Theme Manager ---
        self.theme_manager = ThemeManager(self.root)
        # --- Switch Theme Button ---
        self.switch_theme_button = ttk.Button(self.root, text="Switch Theme", command=self.theme_manager.switch_theme)
        self.switch_theme_button.pack(pady=(8, 2))
        # ... (existing code)
        # Keyboard shortcuts
        self.root.bind('<space>', self._on_space_key)
        self.root.bind('<Left>', self._on_left_key)
        self.root.bind('<Right>', self._on_right_key)

        # --- Time Label for Audio ---
        self.time_label = ttk.Label(self.root, text="00:00 / 00:00", font=("Consolas", 10))
        self.time_label.pack(pady=(0, 5))

        self.root.title("Rio")
        self.root.geometry("600x750") # Slightly taller for seed input
        # --- File & Playback State Initialization ---
        self.generated_files = []
        self.selected_audio_path = None
        self.loaded_audio_path = None
        self.loaded_sample_rate = generated_sample_rate
        
        # --- Pygame Mixer Initialization ---
        try:
            pygame.init() # Initialize all pygame modules
            pygame.mixer.init(frequency=generated_sample_rate)
            print("Pygame mixer initialized.")
        except Exception as e:
            messagebox.showerror("Audio Init Error", f"Failed to initialize audio playback system (Pygame): {e}")
            self._playback_enabled = False
        else:
            self._playback_enabled = True

        self.audio_prompt_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Initializing...")
        self.seed_var = tk.IntVar(value=-1) # Variable to store seed, default -1 (random)
        
        # --- Playback State ---
        self.is_playing = False
        self.is_paused = False
        self.temp_audio_file_path = None
        self.audio_duration = 0.0 # Duration in seconds
        self._seek_update_job = None # For seek bar update job
        self._is_user_seeking = False # Prevent seek updates during drag
        
        # --- Style ---
        style = ttk.Style(self.root)
        style.theme_use('clam') # Or 'alt', 'default', 'classic'

        # Configure dark mode colors (adjust as needed)
        dark_bg = '#2b2b2b'
        light_fg = '#ffffff'
        entry_bg = '#3c3f41'
        select_bg = '#4b6eaf' # A blue selection color
        button_bg = '#3c3f41'
        button_active_bg = '#4b6eaf'

        self.root.configure(bg=dark_bg)

        # Configure colors for various widgets
        style.configure('.',
                      background=dark_bg,
                      foreground=light_fg,
                      fieldbackground=entry_bg,
                      selectbackground=select_bg,
                      selectforeground=light_fg) # Ensure selected text is visible

        # Configure specific widget styles
        style.configure('TFrame', background=dark_bg)
        style.configure('TLabel', background=dark_bg, foreground=light_fg)
        style.configure('TButton',
                      background=button_bg,
                      foreground=light_fg,
                      borderwidth=1,
                      focuscolor=light_fg) # Indicate focus
        style.map('TButton',
                 background=[('active', button_active_bg)],
                 foreground=[('active', light_fg)])
        style.configure('TLabelFrame',
                      background=dark_bg,
                      foreground=light_fg)
        style.configure('TLabelFrame.Label', background=dark_bg, foreground=light_fg) # LabelFrame title
        style.configure('TScale',
                      background=dark_bg,
                      troughcolor=entry_bg, # Trough color
                      sliderthickness=15)
        style.configure('Horizontal.TScale', background=dark_bg)
        style.configure('TEntry', foreground=light_fg, fieldbackground=entry_bg, insertcolor=light_fg) # Entry text/cursor
        style.configure('TScrolledText', fieldbackground=entry_bg) # ScrolledText background isn't directly styleable?

        # Listbox background color
        list_bg = '#3c3f41'

        # --- Main Frame ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input Section ---
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(input_frame, text="Dialogue Script:").pack(anchor=tk.W)
        self.text_input = scrolledtext.ScrolledText(input_frame, height=10, width=60, wrap=tk.WORD)
        self.text_input.pack(fill=tk.X, expand=True)
        default_text = "Dia is an open weights text to dialogue model.\nYou get full control over scripts and voices.\nWow. Amazing. (laughs)\nTry it now!"
        self.text_input.insert(tk.END, default_text)

        # Configure the text widget colors directly (ttk styling is limited)
        self.text_input.configure(
            bg=entry_bg,
            fg=light_fg,
            insertbackground=light_fg, # Cursor color
            selectbackground=select_bg,
            selectforeground=light_fg
        )
        # Bind Ctrl+Backspace AFTER text_input is created and configured
        self.text_input.bind('<Control-BackSpace>', self._on_ctrl_backspace)

        prompt_frame = ttk.Frame(input_frame)
        prompt_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(prompt_frame, text="Select Audio Prompt (Optional)", command=self.select_audio_prompt).pack(side=tk.LEFT)
        self.prompt_label = ttk.Label(prompt_frame, text="No prompt selected.", wraplength=350)
        self.prompt_label.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)

        # --- Control Section ---
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.pack(fill=tk.X, pady=(0, 5)) # Add padding below

        # Generate Button
        self.generate_button = ttk.Button(control_frame, text="Generate Audio", command=self.start_generation)
        self.generate_button.pack(side=tk.LEFT, padx=(0, 10))
        self.generate_button.config(state=tk.DISABLED)

        # Seed Input
        seed_label = ttk.Label(control_frame, text="Seed (-1 random):")
        seed_label.pack(side=tk.LEFT, padx=(10, 2))
        seed_entry = ttk.Entry(control_frame, textvariable=self.seed_var, width=10)
        seed_entry.pack(side=tk.LEFT)

        # Reload/Unload Model Button
        self.reload_button = ttk.Button(control_frame, text="Reload Model", command=self.reload_model)
        self.reload_button.pack(side=tk.LEFT, padx=(10, 0))

        # Generated Files List
        list_frame = ttk.LabelFrame(main_frame, text="Generated Files", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.file_listbox = tk.Listbox(
            list_frame, yscrollcommand=list_scrollbar.set, selectmode=tk.SINGLE,
            bg=list_bg, fg=light_fg, selectbackground=select_bg, selectforeground=light_fg,
            borderwidth=0, highlightthickness=1, highlightcolor=light_fg, exportselection=False
        )
        list_scrollbar.config(command=self.file_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_listbox.bind('<<ListboxSelect>>', self._on_listbox_select)
        self.file_listbox.bind('<Double-Button-1>', self._on_listbox_double_click)
        self._populate_file_list()

        # Output/Playback Section
        output_frame = ttk.LabelFrame(main_frame, text="Playback Control", padding="10")
        output_frame.pack(fill=tk.X)
        button_frame = ttk.Frame(output_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))

        # Playback Buttons
        self.play_pause_button = ttk.Button(button_frame, text="Play", command=self.toggle_play_pause, width=8)
        self.play_pause_button.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_audio, width=8)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))

        # File Action Buttons
        self.save_button = ttk.Button(button_frame, text="Save As...", command=self.save_selected_audio, width=10)
        self.save_button.pack(side=tk.LEFT, padx=(0, 5))

        self.delete_button = ttk.Button(button_frame, text="Delete", command=self.delete_selected_audio, width=8)
        self.delete_button.pack(side=tk.LEFT, padx=(0, 10))

        self.rename_button = ttk.Button(button_frame, text="Rename", command=self.rename_selected_audio, width=8)
        self.rename_button.pack(side=tk.LEFT, padx=(0, 5))

        self.refresh_button = ttk.Button(button_frame, text="Refresh List", command=self.refresh_file_list, width=12)
        self.refresh_button.pack(side=tk.LEFT)

        # Seek Bar
        self.seek_bar = ttk.Scale(output_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self._on_seek_user)
        self.seek_bar.pack(fill="x", expand=True, pady=(5, 0))  # Always stretch horizontally and be visible

        self.seek_bar.pack(fill=tk.X, pady=(5, 0))
        self.seek_bar.bind("<ButtonRelease-1>", self._on_seek_release)
        self.seek_bar.bind("<ButtonPress-1>", self._on_seek_press)

        # Initial button states
        self._disable_playback_controls()
        self.save_button.config(state=tk.DISABLED)
        self.delete_button.config(state=tk.DISABLED)
        self.rename_button.config(state=tk.DISABLED)

        # --- Status Bar ---
        status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, padding="2")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        # Use 'Status.' prefix for potentially different styling
        style.configure('Status.TFrame', background=dark_bg)
        style.configure('Status.TLabel', background=dark_bg, foreground=light_fg)
        status_bar.configure(style='Status.TFrame')

        self.status_label = ttk.Label(status_bar, textvariable=self.status_text, anchor=tk.W)
        self.status_label.pack(fill=tk.X)
        self.status_label.configure(style='Status.TLabel')


        # Start model loading check
        self.check_model_load_queue()
        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _on_space_key(self, event):
        if event.widget == self.text_input:
            return
        self.toggle_play_pause()
        return "break"

    def _on_left_key(self, event):
        if event.widget == self.text_input:
            return
        self._move_seek_bar_by(-1)
        return "break"

    def _on_right_key(self, event):
        if event.widget == self.text_input:
            return
        self._move_seek_bar_by(1)
        return "break"

    def _on_ctrl_backspace(self, event):
        """Delete the word before the cursor in the text_input widget."""
        widget = event.widget
        # Get current cursor position
        index = widget.index(tk.INSERT)
        # Get the text up to the cursor
        text_before = widget.get("1.0", index)
        if not text_before:
            return "break"  # Nothing to delete
        # Find the position of the previous word boundary
        import re
        # Remove trailing whitespace (cursor may be after a space)
        trimmed = text_before.rstrip()
        if not trimmed:
            # Only whitespace before cursor
            widget.delete("1.0", index)
            return "break"
        # Use regex to find last word
        match = re.search(r"(\s*\S+)\s*$", text_before)
        if match:
            start = f"{index} - {len(match.group(0))}c"
        else:
            start = "1.0"
        widget.delete(start, index)
        return "break"  # Prevent default behavior

    def _move_seek_bar_by(self, step):
        cur = self.seek_bar.get()
        new = max(0, min(self.audio_duration, cur + step))
        self.seek_bar.set(new)
        # Always update time display
        self._update_time_label(new)
        self.status_text.set(f"Seek to: {new:.1f}s")


    # --- Methods for Select Prompt, Check Model Queue --- (No changes needed)
    def select_audio_prompt(self):
        filepath = filedialog.askopenfilename(
            title="Select Audio Prompt File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac"), ("All Files", "*.*")]
        )
        if filepath:
            self.audio_prompt_path.set(filepath)
            filename = Path(filepath).name
            self.prompt_label.config(text=f"Prompt: {filename}")
            print(f"Selected audio prompt: {filepath}")
        else:
             self.audio_prompt_path.set("")
             self.prompt_label.config(text="No prompt selected.")

    def check_model_load_queue(self):
        try:
            message_type, message = model_load_queue.get_nowait()
            if message_type == "success":
                self.status_text.set("Model ready.")
                self.generate_button.config(state=tk.NORMAL)
            elif message_type == "error":
                self.status_text.set(f"Error: {message}")
                messagebox.showerror("Model Load Error", message)
                self.generate_button.config(state=tk.DISABLED)
        except queue.Empty:
            self.root.after(200, self.check_model_load_queue)

    # --- Start Generation (MODIFIED FOR SEED) ---
    def start_generation(self):
        global generated_audio_data
        raw_text = self.text_input.get("1.0", tk.END).strip()
        prompt_path = self.audio_prompt_path.get() or None
        if not raw_text:
            messagebox.showwarning("Input Missing", "Please enter some text.")
            return

        # --- Get and Set Seed ---
        try:
            seed_value = self.seed_var.get()
            actual_seed = seed_value

            if seed_value == -1:
                actual_seed = random.randint(0, 2**32 - 1) # Generate a random seed
                self.status_text.set(f"Using random seed: {actual_seed}...")
                print(f"Using random seed: {actual_seed}")
            else:
                self.status_text.set(f"Using fixed seed: {seed_value}...")
                print(f"Using fixed seed: {seed_value}")

            set_seed(actual_seed) # Set the seed before generation

        except tk.TclError:
            messagebox.showerror("Invalid Seed", "Seed value must be an integer.")
            self.status_text.set("Invalid seed value.")
            return
        # --- End Seed Handling ---

        formatted_text = preprocess_text(raw_text)
        if not formatted_text:
             messagebox.showwarning("Input Empty", "The input text was empty after formatting.")
             self.status_text.set("Input text became empty.") # Update status
             return

        # Proceed with generation
        self.stop_audio() # Stop previous playback & cleanup
        self.status_text.set(f"{self.status_text.get()} Generating audio...") # Append to seed status
        self.generate_button.config(state=tk.DISABLED)
        self._disable_playback_controls() # Disable all playback UI
        generated_audio_data = None

        gen_thread = threading.Thread(
            target=run_inference_thread, args=(formatted_text, prompt_path), daemon=True
        )
        gen_thread.start()
        self.check_generation_queue()

    # --- Check Generation Queue (Modified for Duration and UI Update) ---
    def check_generation_queue(self):
        global generated_audio_data, generated_sample_rate
        try:
            message_type, data = generation_queue.get_nowait()
            self.generate_button.config(state=tk.NORMAL) # Re-enable generate button

            if message_type == "success":
                sr, audio_np = data
                generated_sample_rate = sr # Use the actual sample rate if model provides it
                generated_audio_data = audio_np
                try:
                    # --- Prepare audio for playback ---
                    audio_to_save = generated_audio_data.astype(np.float32)
                    # Normalize to [-1, 1] for safety, though sf.write with FLOAT handles range
                    max_val = np.max(np.abs(audio_to_save))
                    if max_val == 0:
                        print("Warning: Generated audio is silent.")
                    elif max_val > 1.0:
                        print(f"Normalizing audio (max abs value was {max_val:.3f})")
                        audio_to_save /= max_val

                    # Save directly to OUTPUT_DIR with formatted filename
                    from datetime import datetime
                    raw_text = self.text_input.get("1.0", tk.END).strip()
                    text_snippet = (raw_text.strip().replace("\n", " ")[:20] or "untitled").replace(' ', '_').replace('/', '_')
                    now = datetime.now()
                    date_str = now.strftime("%Y%m%d_%H%M%S")
                    persistent_name = f"{date_str}_{text_snippet}.wav"
                    persistent_path = OUTPUT_DIR / persistent_name
                    sf.write(persistent_path, audio_to_save, generated_sample_rate, subtype='FLOAT')
                    print(f"Saved generated audio to: {persistent_path}")
                    self._add_to_file_list(str(persistent_path))
                    self.selected_audio_path = persistent_path
                    self.temp_audio_file_path = str(persistent_path)  # For playback compatibility

                    # Get duration using soundfile
                    info = sf.info(self.temp_audio_file_path)
                    self.audio_duration = info.duration
                    print(f"Audio duration: {self.audio_duration:.2f} seconds")

                    # --- Update UI for Playback ---
                    self.status_text.set("Audio generation complete.")
                    if self._playback_enabled:
                        if pygame.mixer.get_init() and pygame.mixer.get_init()[0] != generated_sample_rate:
                            print(f"Re-initializing mixer for sample rate: {generated_sample_rate} Hz")
                            pygame.mixer.quit()
                            pygame.mixer.init(frequency=generated_sample_rate)
                        self.play_pause_button.config(state=tk.NORMAL, text="Play")
                        self.stop_button.config(state=tk.NORMAL)
                        self.seek_bar.config(state=tk.NORMAL, to=math.ceil(self.audio_duration))
                        self.seek_bar.set(0)
                    self.save_button.config(state=tk.NORMAL)
                    self._reset_playback_state_vars()  # Reset flags like is_playing
                    # Do NOT delete the file after playback; this is now a permanent output.

                except sf.LibsndfileError as e:
                    messagebox.showerror("Audio Write Error", f"Failed to write temporary audio file: {e}\nAudio data might be invalid.")
                    self.status_text.set("Error saving temporary audio.")
                    self._cleanup_temp_audio_file()
                    self._disable_playback_controls()
                    # Keep save button enabled if raw data exists, user might save elsewhere
                    self.save_button.config(state=tk.NORMAL if generated_audio_data is not None else tk.DISABLED)
                except Exception as e:
                    messagebox.showerror("Audio Prep Error", f"Failed to prepare audio for playback: {e}")
                    self.status_text.set("Error preparing audio.")
                    self._cleanup_temp_audio_file()
                    self._disable_playback_controls()
                    self.save_button.config(state=tk.NORMAL if generated_audio_data is not None else tk.DISABLED)

            elif message_type == "warning":
                 warning_msg = f"Warning: {data}"
                 self.status_text.set(warning_msg)
                 messagebox.showwarning("Generation Warning", data)
                 self._disable_playback_controls()
            elif message_type == "error":
                error_msg = f"Error: {data}"
                self.status_text.set(error_msg)
                messagebox.showerror("Generation Error", data)
                self._disable_playback_controls()

        except queue.Empty:
            # Continue checking
            self.root.after(200, self.check_generation_queue)
        except Exception as e:
             # Catch unexpected errors during queue processing
             self.status_text.set(f"Error checking results: {e}")
             self.generate_button.config(state=tk.NORMAL) # Ensure button is usable
             messagebox.showerror("Application Error", f"Error processing generation results: {e}")
             self._disable_playback_controls()


    # --- Playback Controls (Mostly same, added robustness) ---
    def toggle_play_pause(self):
        if not self._playback_enabled or not self.temp_audio_file_path: return

        try:
            # Ensure mixer is initialized
            if not pygame.mixer.get_init():
                print("Re-initializing Pygame mixer...")
                pygame.mixer.init(frequency=generated_sample_rate)
                if not pygame.mixer.get_init():
                     messagebox.showerror("Playback Error", "Failed to initialize audio mixer.")
                     return

            # Always use the current seek bar value as the play/resume position
            start_pos = self.seek_bar.get()
            if not self.is_playing:
                pygame.mixer.music.load(self.temp_audio_file_path)
                pygame.mixer.music.play()
                if start_pos > 0.1:
                    pygame.mixer.music.set_pos(start_pos)
                self.status_text.set("Playing...")
                print(f"Playing audio (requested start: {start_pos:.2f}s).")
                self.is_playing = True
                self.is_paused = False
                self.play_pause_button.config(text="Pause")
                self._start_seek_updater() # Start updating the seek bar
            else: # Pause
                pygame.mixer.music.pause()
                self.is_playing = False
                self.is_paused = True
                self.play_pause_button.config(text="Resume")
                self.status_text.set("Paused.")
                print("Paused audio.")
                self._stop_seek_updater() # Stop updating seek bar

        except pygame.error as e:
             messagebox.showerror("Playback Error", f"Pygame error during playback: {e}\nTry regenerating or restarting.")
             self.status_text.set("Playback error.")
             self._reset_playback_ui() # Reset controls
             self._cleanup_temp_audio_file() # Clean up potentially problematic file
             self._disable_playback_controls()
        except Exception as e:
            messagebox.showerror("Playback Error", f"Unexpected error during playback control: {e}")
            self.status_text.set("Playback error.")
            self._reset_playback_ui()

    def stop_audio(self):
        if not self._playback_enabled: return
        try:
            self._stop_seek_updater() # Stop updates first
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload() # Explicitly unload the file
                print("Stopped and unloaded audio.")
            # Even if not busy, reset UI and clean up potential temp file
            self._reset_playback_ui() # Reset buttons and seek bar
            self.status_text.set("Stopped.")
            self._cleanup_temp_audio_file() # Try cleaning up now after unload

        except pygame.error as e:
             messagebox.showerror("Playback Error", f"Pygame error stopping audio: {e}")
             self._reset_playback_ui()
        except Exception as e:
             messagebox.showerror("Playback Error", f"Error stopping audio: {e}")
             self._reset_playback_ui()
        # Ensure seek updater is stopped if not already
        self._stop_seek_updater()


    # --- Seeking Methods (Mostly same, added robustness) ---
    def _on_seek_press(self, event=None):
        # Called when user presses the seek bar (slider)
        self._is_user_seeking = True

    def _start_seek_updater(self):
        self._stop_seek_updater()
        self._update_seek_bar()

    def _update_seek_bar(self):
        # Only update if playing, not seeking by user, and mixer is valid
        if self.is_playing and not self._is_user_seeking and pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            try:
                current_time_ms = pygame.mixer.music.get_pos()
                if current_time_ms != -1:
                    current_time_sec = current_time_ms / 1000.0
                    # Prevent seek bar going beyond max value due to timing delays
                    display_time_sec = min(current_time_sec, self.audio_duration)
                    self.seek_bar.set(display_time_sec)
                    self._update_time_label(display_time_sec)

                    # Check if playback naturally finished using a small buffer
                    if self.audio_duration > 0 and current_time_sec >= self.audio_duration - 0.15:
                        # Instead of resetting UI, rewind and pause at start
                        self.seek_bar.set(0)
                        self._update_time_label(0)
                        self.is_playing = False
                        self.is_paused = True
                        self.play_pause_button.config(text="Play")
                        self.status_text.set("Playback finished. Ready to replay.")
                        self._stop_seek_updater()
                        return
                else:
                    # get_pos() returned -1, means stopped unexpectedly?
                    if self.is_playing:
                        self._reset_playback_ui()
                        self.status_text.set("Playback stopped unexpectedly.")
                    self._stop_seek_updater()
                    return

                # Schedule the next update if still playing and not finished
                self._seek_update_job = self.root.after(200, self._update_seek_bar)

            except Exception as e:
                print(f"Error in _update_seek_bar: {e}")
                self._stop_seek_updater()
        else:
            # Ensure the updater stops if state changes (e.g., stopped, paused)
            self._stop_seek_updater()

    def _stop_seek_updater(self):
        if self._seek_update_job:
            try:
                self.root.after_cancel(self._seek_update_job)
            except Exception: # Catch potential errors if job is already invalid
                pass

    def _update_time_label(self, current_time):
        # Format seconds as mm:ss
        def fmt(secs):
            mins = int(secs) // 60
            secs = int(secs) % 60
            return f"{mins:02}:{secs:02}"
        total = self.audio_duration if hasattr(self, 'audio_duration') else 0
        self.time_label.config(text=f"{fmt(current_time)} / {fmt(total)}")

    def _on_seek_user(self, value_str):
        # Called when user drags the slider (not released yet)
        try:
            value = float(value_str)
            self.status_text.set(f"Seek: {value:.1f}s")
            self._update_time_label(value)
        except Exception:
            pass

    def _on_seek_release(self, event=None):
        # Called when user RELEASES the mouse button on the slider
        was_seeking = self._is_user_seeking # Store flag before potentially changing state
        self._is_user_seeking = False # Reset flag immediately

        if was_seeking and self._playback_enabled and self.temp_audio_file_path:
            seek_time = self.seek_bar.get() # Get final value from the slider itself
            try:
                if pygame.mixer.get_init():
                    # Clamp seek time for safety
                    seek_time = max(0, min(seek_time, self.audio_duration - 0.05)) # Seek slightly before end

                    # Always set position, even if not playing
                    if self.is_playing:
                        pygame.mixer.music.set_pos(seek_time)
                        self.status_text.set(f"Playing from {seek_time:.1f}s")
                        self._start_seek_updater()
                    elif self.is_paused:
                        pygame.mixer.music.set_pos(seek_time)
                        self.status_text.set(f"Seeked to {seek_time:.1f}s (Paused)")
                    else:
                        # Not playing or paused: just update seek bar, playback will start from here
                        self.status_text.set(f"Ready to play from {seek_time:.1f}s")

                    print(f"User seeked to {seek_time:.2f} seconds.")

            except pygame.error as e:
                messagebox.showerror("Seek Error", f"Pygame error during seek: {e}")
                self._reset_playback_ui()
            except Exception as e:
                messagebox.showerror("Seek Error", f"Could not seek audio: {e}")
                self._reset_playback_ui()

        # Called when user RELEASES the mouse button on the slider
        was_seeking = self._is_user_seeking # Store flag before potentially changing state
        self._is_user_seeking = False # Reset flag immediately

        if was_seeking and self._playback_enabled and self.temp_audio_file_path:
            seek_time = self.seek_bar.get() # Get final value from the slider itself
            try:
                if pygame.mixer.get_init():
                    # Clamp seek time for safety
                    seek_time = max(0, min(seek_time, self.audio_duration - 0.05)) # Seek slightly before end

                    # Pygame's seeking behavior can be tricky.
                    # Playing from a specific point is often more reliable than seek on a playing stream.
                    if self.is_playing:
                        # Stop, reload (optional but safer?), play from seek time
                        pygame.mixer.music.stop()
                        pygame.mixer.music.load(self.temp_audio_file_path) # Reload might be needed
                        pygame.mixer.music.play()
                        pygame.mixer.music.set_pos(seek_time)
                        self.status_text.set(f"Playing from {seek_time:.1f}s")
                        self._start_seek_updater() # Restart updater if it was stopped
                    elif self.is_paused:
                        # If paused, just set position. User needs to press Resume.
                        # Need to load if stopped completely before pausing
                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.load(self.temp_audio_file_path)
                            # Don't auto-play if paused
                        pygame.mixer.music.set_pos(seek_time)
                        self.status_text.set(f"Seeked to {seek_time:.1f}s (Paused)")
                    else: # Was stopped, start playing from seeked position
                         pygame.mixer.music.load(self.temp_audio_file_path)
                         pygame.mixer.music.play()
                         pygame.mixer.music.set_pos(seek_time)
                         self.is_playing = True # Update state
                         self.is_paused = False
                         self.play_pause_button.config(text="Pause")
                         self.status_text.set(f"Playing from {seek_time:.1f}s")
                         self._start_seek_updater()

                    print(f"User seeked to {seek_time:.2f} seconds.")

            except pygame.error as e:
                messagebox.showerror("Seek Error", f"Pygame error during seek: {e}")
                self._reset_playback_ui()
            except Exception as e:
                messagebox.showerror("Seek Error", f"Could not seek audio: {e}")
                self._reset_playback_ui()


    # --- UI Reset and Cleanup (Mostly same) ---
    def _reset_playback_state_vars(self):
         self.is_playing = False
         self.is_paused = False

    def _reset_playback_ui(self):
         self._reset_playback_state_vars()
         # Only enable playback buttons if audio data/file exists and playback is possible
         can_play = self._playback_enabled and self.temp_audio_file_path and Path(self.temp_audio_file_path).exists()

         if can_play:
              self.play_pause_button.config(text="Play", state=tk.NORMAL)
              self.stop_button.config(state=tk.NORMAL)
              self.seek_bar.set(0) # Reset position
              self.seek_bar.config(state=tk.NORMAL, to=math.ceil(self.audio_duration)) # Ensure range is set
         else:
              self._disable_playback_controls() # Call the full disable function


    def _disable_playback_controls(self):
        self._stop_seek_updater() # Stop updates if any
        self.play_pause_button.config(state=tk.DISABLED, text="Play")
        self.stop_button.config(state=tk.DISABLED)
        self.seek_bar.set(0)
        # Also disable and reset the range of the seek bar
        self.seek_bar.config(state=tk.DISABLED, to=100) # Reset 'to' value to default/placeholder
        self.audio_duration = 0.0 # Reset duration tracking
        # Save button depends only on whether data exists, not playback state
        self.save_button.config(state=tk.NORMAL if generated_audio_data is not None else tk.DISABLED)
        self._reset_playback_state_vars()


    def save_audio(self):
        global generated_audio_data, generated_sample_rate
        if generated_audio_data is not None and generated_sample_rate is not None:
            filepath = filedialog.asksaveasfilename(
                title="Save Generated Audio", defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All Files", "*.*")]
            )
            if filepath:
                try:
                    # Save the raw generated data, apply normalization if needed
                    audio_to_save = generated_audio_data.astype(np.float32)
                    max_val = np.max(np.abs(audio_to_save))
                    if max_val == 0:
                         print("Saving silent audio.")
                    elif max_val > 1.0:
                         print("Normalizing audio before saving.")
                         audio_to_save /= max_val

                    self.status_text.set(f"Saving to {Path(filepath).name}...")
                    self.root.update_idletasks() # Ensure status message shows
                    sf.write(filepath, audio_to_save, generated_sample_rate, subtype='FLOAT')
                    self.status_text.set(f"Audio saved to {Path(filepath).name}")
                    print(f"Audio saved successfully to {filepath}")
                    # Briefly show success then revert status
                    final_status = "Audio generation complete." if self.temp_audio_file_path else "Model ready."
                    self.root.after(2000, lambda s=final_status: self.status_text.set(s))
                except Exception as e:
                    messagebox.showerror("Save Error", f"Could not save audio file: {e}")
                    self.status_text.set("Save error.")
                # No finally needed here, status is handled above
        else:
            messagebox.showinfo("No Audio", "No audio has been generated to save.")


    def _cleanup_temp_audio_file(self):
        path_to_delete = self.temp_audio_file_path
        # Only delete if NOT in OUTPUT_DIR (i.e., only delete true temp files)
        if path_to_delete:
            try:
                temp_path = Path(path_to_delete)
                if temp_path.exists() and not str(temp_path).startswith(str(OUTPUT_DIR)):
                    temp_path.unlink()
                    print(f"Deleted temporary playback file: {path_to_delete}")
                elif temp_path.exists():
                    print(f"Refused to delete file in OUTPUT_DIR: {path_to_delete}")
            except PermissionError as e:
                print(f"Warning: PermissionError deleting temporary file {path_to_delete}: {e}. File might still be in use.")
            except OSError as e:
                print(f"Warning: OSError deleting temporary playback file {path_to_delete}: {e}")
            except Exception as e:
                print(f"Warning: Unexpected error deleting temporary file {path_to_delete}: {e}")
        self.temp_audio_file_path = None  # Clear path reference last, after logic


    def on_closing(self):
        """Handles window close event."""
        print("Closing application...")
        # Stop and unload music first to release file handle
        if self._playback_enabled and pygame.mixer.get_init():
             try:
                  if pygame.mixer.music.get_busy():
                       pygame.mixer.music.stop()
                  pygame.mixer.music.unload() # Unload file explicitly
                  print("Unloaded Pygame music.")
             except pygame.error as e:
                  print(f"Pygame error during unload: {e}")
             except Exception as e:
                  print(f"Error unloading music: {e}")
             # Wait briefly ONLY IF unload might need time (usually not needed, but can help on some OS)
             # time.sleep(0.1)

        # Attempt cleanup AFTER unloading
        self._cleanup_temp_audio_file()

        # Quit Pygame systems
        if self._playback_enabled and pygame.get_init():
             try:
                 pygame.mixer.quit()
                 pygame.quit()
                 print("Pygame quit.")
             except Exception as e:
                  print(f"Error quitting Pygame: {e}")

        # Destroy Tkinter window
        self.root.destroy()

    def _populate_file_list(self):
        """Scans the output directory and updates the listbox."""
        self.generated_files = []
        self.file_listbox.delete(0, tk.END)
        if not OUTPUT_DIR.exists():
            print(f"Warning: Output directory {OUTPUT_DIR} not found during population.")
            return
        try:
            files = sorted(
                [p for p in OUTPUT_DIR.glob("*.wav") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            self.generated_files = files
            for f_path in self.generated_files:
                self.file_listbox.insert(tk.END, f_path.name)
        except Exception as e:
            print(f"Error populating file list from {OUTPUT_DIR}: {e}")
            messagebox.showwarning("List Error", f"Could not read files from {OUTPUT_DIR}:\n{e}")

    def _clear_selection_state(self):
        """Clears file selection and disables related controls."""
        self.selected_audio_path = None
        for btn in (self.save_button, self.delete_button, self.rename_button):
            btn.config(state=tk.DISABLED)
        self.play_pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.seek_bar.config(state=tk.DISABLED)
        self.status_text.set("No file selected.")

    def _on_listbox_select(self, event=None):
        """Handles selection change in the listbox."""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            self._clear_selection_state()
            return
        idx = selected_indices[0]
        try:
            filename = self.file_listbox.get(idx)
            for p in self.generated_files:
                if p.name == filename:
                    self.selected_audio_path = p
                    self.save_button.config(state=tk.NORMAL)
                    self.delete_button.config(state=tk.NORMAL)
                    self.rename_button.config(state=tk.NORMAL)
                    self.status_text.set(f"Selected: {p.name}")
                    return
            print(f"Error: Filename '{filename}' not found.")
            self._clear_selection_state()
        except Exception as e:
            print(f"Error during listbox selection: {e}")
            self._clear_selection_state()

    def _on_listbox_double_click(self, event=None):
        """Loads the selected audio for playback on double-click."""
        self._on_listbox_select()
        if self.selected_audio_path and self._playback_enabled:
            self._load_selected_audio()
        elif not self.selected_audio_path:
            print("Double-click ignored, no valid file selected.")
        else:
            messagebox.showerror("Playback Disabled", "Audio playback system not initialized.")

    def _load_selected_audio(self):
        """Loads the selected audio file into Pygame for playback."""
        self._update_time_label(0)
        if not self.selected_audio_path or not self.selected_audio_path.exists():
            messagebox.showerror("Load Error", f"Selected audio file not found:\n{self.selected_audio_path}")
            self._reset_playback_ui()
            self.loaded_audio_path = None
            self.refresh_file_list()
            return

        try:
            if self._playback_enabled and pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()

            self._reset_playback_state_vars()
            info = sf.info(str(self.selected_audio_path))
            sr, dur = info.samplerate, info.duration
            mixer_needs_init = True
            if pygame.mixer.get_init():
                cur_freq, _, _ = pygame.mixer.get_init()
                if cur_freq == sr:
                    mixer_needs_init = False
                else:
                    pygame.mixer.quit()

            if mixer_needs_init:
                pygame.mixer.init(frequency=sr)
                pygame.mixer.music.set_volume(0.8)

            pygame.mixer.music.load(str(self.selected_audio_path))
            self.loaded_audio_path = self.selected_audio_path
            self.loaded_sample_rate = sr
            self.audio_duration = dur
            self.play_pause_button.config(state=tk.NORMAL, text="Play")
            self.stop_button.config(state=tk.NORMAL)
            self.seek_bar.config(state=tk.NORMAL, to=math.ceil(dur))
            self.seek_bar.set(0)
            self.status_text.set(f"Loaded: {self.selected_audio_path.name}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self._reset_playback_ui()
            self.loaded_audio_path = None

    def _add_to_file_list(self, filepath_str):
        """Adds a new file to the internal list and the top of the listbox."""
        filepath = Path(filepath_str)
        if filepath.exists() and filepath not in self.generated_files:
            self.generated_files.insert(0, filepath)
            self.file_listbox.insert(0, filepath.name)
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(0)
        self.file_listbox.activate(0)
        self._on_listbox_select()

    def refresh_file_list(self):
        """Stops playback, re-scans the output directory, and updates the listbox."""
        if self.loaded_audio_path:
            self.stop_audio()
            if self._playback_enabled and pygame.mixer.get_init():
                pygame.mixer.music.unload()
            self.loaded_audio_path = None
            self._reset_playback_ui()
        self._clear_selection_state()
        self._populate_file_list()
        self.status_text.set("File list refreshed.")

    def save_selected_audio(self):
        """Saves a copy of the selected audio file."""
        if self.selected_audio_path and self.selected_audio_path.exists():
            default = self.selected_audio_path.name
            filepath = filedialog.asksaveasfilename(
                title="Save Selected Audio As...", initialdir=str(OUTPUT_DIR),
                initialfile=default, defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All Files", "*.*")]
            )
            if filepath:
                try:
                    target = Path(filepath)
                    if target.resolve() == self.selected_audio_path.resolve():
                        messagebox.showwarning("Save As", "Cannot overwrite the original file.")
                        return
                    shutil.copy2(str(self.selected_audio_path), filepath)
                    self.status_text.set(f"Saved copy to {target.name}")
                except Exception as e:
                    messagebox.showerror("Save Error", str(e))
                    self.status_text.set("Save error.")
        else:
            messagebox.showinfo("No File Selected", "Select a file from the list to save.")

    def delete_selected_audio(self):
        """Deletes the selected audio file after confirmation."""
        if not self.selected_audio_path:
            messagebox.showwarning("Delete Error", "No file selected in the list.")
            return
        if not self.selected_audio_path.exists():
            messagebox.showerror("Delete Error", f"Selected file not found:\n{self.selected_audio_path.name}")
            self._clear_selection_state()
            self._populate_file_list()
            return
        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {self.selected_audio_path.name}?")
        if confirm:
            p = self.selected_audio_path
            try:
                if self.loaded_audio_path == p and self._playback_enabled and pygame.mixer.get_init():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                # Move to recycle bin if permanent file
                if send2trash and str(p).startswith(str(OUTPUT_DIR)):
                    send2trash(str(p))
                elif str(p).startswith(str(OUTPUT_DIR)):
                    p.unlink()
                    messagebox.showwarning("Delete Warning", "send2trash not installed. File permanently deleted.")
                else:
                    # For temp files, just unlink
                    p.unlink()
                if p in self.generated_files:
                    self.generated_files.remove(p)
                idx = self.file_listbox.curselection()
                if idx:
                    self.file_listbox.delete(idx[0])
            except Exception as e:
                messagebox.showerror("Delete Error", str(e))
            self.play_pause_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)
            self.seek_bar.config(state=tk.DISABLED)
            self.status_text.set("No file selected.")

    def rename_selected_audio(self):
        """Renames the selected audio file, stopping playback if necessary."""
        if not self.selected_audio_path:
            messagebox.showwarning("Rename Error", "No file selected in the list.")
            return
        new_path = filedialog.asksaveasfilename(
            title="Rename Selected Audio", initialdir=str(OUTPUT_DIR),
            initialfile=self.selected_audio_path.name, defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All Files", "*.*")]
        )
        if new_path:
            try:
                target = Path(new_path)
                # If the file is currently loaded for playback, stop and unload it first
                if self.loaded_audio_path == self.selected_audio_path and self._playback_enabled and pygame.mixer.get_init():
                    try:
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                        print("Unloaded Pygame music before renaming.")
                    except Exception as e:
                        print(f"Warning: Error unloading music before rename: {e}")
                    self.loaded_audio_path = None
                    self._reset_playback_ui()
                self.selected_audio_path.rename(target)
                self._populate_file_list()
                self.status_text.set(f"Renamed to {target.name}")
            except Exception as e:
                messagebox.showerror("Rename Error", str(e))

    def toggle_play_pause(self, *_):
        """Toggles playback of the selected audio file. Accepts optional event arg for key binding."""
        if not self.selected_audio_path:
            messagebox.showinfo("No File", "Select a file to play.")
            return
        # Load selected file if not already loaded or changed
        if self.loaded_audio_path != self.selected_audio_path:
            self._load_selected_audio()
        # Play (from seek bar position)
        if not self.is_playing:
            seek_time = self.seek_bar.get()
            pygame.mixer.music.play()
            pygame.mixer.music.set_pos(seek_time)
            self.is_playing = True
            self.is_paused = False
            self.play_pause_button.config(text="Pause")
            self._start_seek_updater()  # Start updating seek bar as playback starts
            self._update_time_label(seek_time)
        # Pause
        elif self.is_playing and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True
            self.play_pause_button.config(text="Resume")
            self._stop_seek_updater()  # Stop seek updates when paused
        # Resume
        elif self.is_paused:
            seek_time = self.seek_bar.get()
            pygame.mixer.music.play()
            pygame.mixer.music.set_pos(seek_time)
            self.is_playing = True
            self.is_paused = False
            self.play_pause_button.config(text="Pause")
            self._start_seek_updater()  # Resume seek bar updates when resuming playback
            self._update_time_label(seek_time)


    def reload_model(self):
        """Unload and reload the Dia model."""
        if not MODEL_AVAILABLE:
            messagebox.showerror("Model Not Available", "Dia/model_manager not found.")
            return
        try:
            model_manager.unload_model()
            self.status_text.set("Model unloaded. Reloading...")
            self.root.update_idletasks()
            model_manager.load_model(device)
            self.status_text.set("Model reloaded successfully.")
        except Exception as e:
            self.status_text.set(f"Reload failed: {e}")
            messagebox.showerror("Reload Error", str(e))


# --- Main Execution ---
if __name__ == "__main__":
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps") # Check for MPS support
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    root = tk.Tk()
    status_text_initial = "Loading Dia model..."
    app = TTSApp(root) # Initialize the GUI
    app.status_text.set(status_text_initial) # Set initial status message

    # Always load the model in a background thread
    def auto_load_model():
        try:
            load_model(device)
            app.status_text.set("Model loaded successfully.")
            app.generate_button.config(state=tk.NORMAL)
        except Exception as e:
            app.status_text.set(f"Error loading model: {e}")
            app.generate_button.config(state=tk.DISABLED)
            app._disable_playback_controls()
    threading.Thread(target=auto_load_model, daemon=True).start()

    # Disable playback if pygame failed
    if not app._playback_enabled:
        app._disable_playback_controls()
        app.status_text.set("Error: Pygame audio init failed. Playback disabled.")

    # Start the Tkinter event loop
    root.mainloop()