# TranscriptionWindow.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import faster_whisper
import torch
import threading


class TranscriptionWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Audio Transcription")
        self.root.attributes("-alpha", 0.8)
        self.root.configure(bg="black")

        self.root.lift()
        self.root.attributes('-topmost', True)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width // 3.5)
        window_height = screen_height // 8
        self.root.geometry(f"{window_width}x{window_height}")

        # Initialize Whisper model with a smaller size and memory efficient settings
        try:
            self.model = faster_whisper.WhisperModel(
                "medium",  # Using medium model instead of large-v2
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="float16"  # Use half precision to reduce memory usage
            )
        except Exception as e:
            # Fallback to CPU if GPU initialization fails
            self.model = faster_whisper.WhisperModel(
                "medium",
                device="cpu",
                compute_type="int8"  # Use int8 quantization for CPU
            )

        # Create buttons frame
        self.button_frame = tk.Frame(self.root, bg="black")
        self.button_frame.grid(row=0, column=0, pady=5, sticky=tk.EW)

        # Add file selection button
        self.file_button = tk.Button(self.button_frame, text="Select Audio File", command=self.select_file)
        self.file_button.pack(side=tk.LEFT, padx=5)

        # Text display area
        self.text_widget = tk.Text(self.root, wrap=tk.WORD, font=("Gothic", 16),
                                bg="black", fg="white", bd=0, highlightthickness=0)
        self.text_widget.grid(padx=5, pady=5, row=1, column=0, sticky=tk.NSEW)

        self.scrollbar = ttk.Scrollbar(self.root, command=self.text_widget.yview)
        self.scrollbar.grid(row=1, column=1, sticky=tk.NS)
        self.text_widget["yscrollcommand"] = self.scrollbar.set

        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.transcription_thread = None

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav")]
        )
        if file_path:
            # Start transcription in a separate thread
            self.transcription_thread = threading.Thread(
                target=self.transcribe_audio,
                args=(file_path,),
                daemon=True
            )
            self.transcription_thread.start()

    def transcribe_audio(self, audio_path):
        try:
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(tk.END, "Transcribing audio...\n")
            self.root.update()

            # Clear CUDA cache before transcription
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Transcribe with memory efficient settings
            segments, _ = self.model.transcribe(
                audio_path,
                beam_size=1,  # Reduced beam size
                language="en",
                condition_on_previous_text=False,  # Disable to save memory
                vad_filter=True  # Enable voice activity detection to process only speech
            )
            
            text_to_display = ""
            for segment in segments:
                text_to_display += f"{segment.text}\n\n"

            # Update GUI in the main thread
            self.root.after(0, self.update_text_widget, text_to_display)
        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def update_text_widget(self, text):
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)

    def show_error(self, error_message):
        messagebox.showerror("Error", f"An error occurred: {error_message}")

    def mainloop(self):
        self.root.mainloop()
if __name__=="__main__":
    app=TranscriptionWindow()
    app.mainloop()