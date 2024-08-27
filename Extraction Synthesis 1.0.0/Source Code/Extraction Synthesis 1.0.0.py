import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from Audio.MusicToInstrument import MusicToInstrument
import base64
import sys
import io

# Blank icon (1x1 transparent GIF)
BLANK_ICON = base64.b64decode(b'R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7')

class MusicToInstrumentGUI:
    def __init__(self, master):
        master.configure(background='#2b2b2b')
        master.option_add('*Font', 'Helvetica 10')

        self.master = master
        master.title("Digital Down's Extraction Synthesis 1.0.0")
        master.geometry("500x250")
        master.resizable(False, False)
        # Set blank icon
        self.icon = tk.PhotoImage(data=BLANK_ICON)
        master.iconphoto(True, self.icon)

        # Load the background image
        if getattr(sys, 'frozen', False):
            self.background_image = tk.PhotoImage(file=os.path.join(sys._MEIPASS, 'Background E3.png'))
        else:
            self.background_image = tk.PhotoImage(file='Background E3.png')

        # Create a label with the background image
        self.background_label = tk.Label(master, image=self.background_image)
        self.background_label.place(x=0, y=0, width=500, height=250)

        # Keep a reference to the image to prevent it from being garbage collected
        self.background_label.image = self.background_image

        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.normalize = tk.BooleanVar(value=True)
        self.threshold = tk.IntVar(value=100)

        # Input folder selection
        tk.Label(master, text="Input Folder:", foreground='#ffffff', background='#000000', highlightthickness=0).grid(row=0, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(master, textvariable=self.input_folder, width=40, foreground='#ffffff', background='#000000', highlightthickness=0).grid(row=0, column=1, columnspan=2, padx=10, pady=5)
        tk.Button(master, text="Browse", command=self.browse_input_folder, foreground='#ffffff', background='#000000', highlightthickness=0).grid(row=0, column=3, padx=10, pady=5)

        # Output folder selection
        tk.Label(master, text="Output Folder:", foreground='#ffffff', background='#000000', highlightthickness=0).grid(row=1, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(master, textvariable=self.output_folder, width=40, foreground='#ffffff', background='#000000', highlightthickness=0).grid(row=1, column=1, columnspan=2, padx=10, pady=5)
        tk.Button(master, text="Browse", command=self.browse_output_folder, foreground='#ffffff', background='#000000', highlightthickness=0).grid(row=1, column=3, padx=10, pady=5)

        # Normalize checkbox
        self.normalize_checkbox = tk.Checkbutton(master, text="Normalize", variable=self.normalize, onvalue=True, offvalue=False, foreground='#ffffff', background='#000000', highlightthickness=0, selectcolor='#2b2b2b')
        self.normalize_checkbox.grid(row=2, column=0, sticky="w", padx=10, pady=10)

        # Threshold entry
        tk.Label(master, text="Threshold:", foreground='#ffffff', background='#000000', highlightthickness=0).grid(row=2, column=1, sticky="e", padx=10, pady=10)
        self.threshold_entry = tk.Entry(master, textvariable=self.threshold, width=5, foreground='#ffffff', background='#000000', highlightthickness=0)
        self.threshold_entry.grid(row=2, column=2, padx=10, pady=10)

        # Run button
        self.run_button = tk.Button(master, text="Process", command=self.run_script, background='#000000', foreground='#ffffff')
        self.run_button.grid(row=2, column=3, padx=10, pady=10)
        self.run_button.config(state=tk.DISABLED)

        # Output box
        self.output_box = tk.Text(master, width=60, height=4, foreground='#ffffff', background='#000000', highlightthickness=0)
        self.output_box.grid(row=3, column=0, columnspan=4, padx=10, pady=10)
        self.output_box.config(state=tk.DISABLED)

        # Supported file formats label
        supported_formats_label = tk.Label(master, text="Currently only supports WAV files", background='#000000', foreground='#ffffff')
        supported_formats_label.grid(row=4, column=0, columnspan=4, padx=10, pady=5)

        # Bind events to update run button state
        self.input_folder.trace_add("write", self.update_run_button_state)
        self.output_folder.trace_add("write", self.update_run_button_state)

    def browse_input_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.input_folder.set(folder_path)

    def browse_output_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_folder.set(folder_path)

    def update_run_button_state(self, *args):
        if self.input_folder.get() and self.output_folder.get():
            self.run_button.config(state=tk.NORMAL)
        else:
            self.run_button.config(state=tk.DISABLED)
    def redirect_output(self):
        class StdoutRedirector(io.StringIO):
            def __init__(self, text_widget):
                self.text_widget = text_widget
                io.StringIO.__init__(self)

            def write(self, string):
                self.text_widget.config(state=tk.NORMAL)
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
                self.text_widget.config(state=tk.DISABLED)
                self.text_widget.update()

        sys.stdout = StdoutRedirector(self.output_box)
    def run_script(self):
        input_folder = self.input_folder.get()
        output_folder = self.output_folder.get()
        threshold = self.threshold.get()

        try:
            self.output_box.config(state=tk.NORMAL)
            self.output_box.delete("1.0", tk.END)
            self.output_box.config(state=tk.DISABLED)

            # Redirect stdout to the output box
            old_stdout = sys.stdout
            self.redirect_output()

            if self.normalize.get():
                MusicToInstrument.process_audio_folder_normalize(input_folder, output_folder, incomplete_folder=output_folder, threshold=threshold)
            else:
                MusicToInstrument.process_audio_folder(input_folder, output_folder, incomplete_folder=output_folder, threshold=threshold)

            # Restore stdout
            sys.stdout = old_stdout

            messagebox.showinfo("Success", "Audio processing completed successfully!")
        except Exception as e:
            # Restore stdout in case of exception
            sys.stdout = old_stdout
            
            self.output_box.config(state=tk.NORMAL)
            self.output_box.insert(tk.END, f"Error: {str(e)}\n")
            self.output_box.config(state=tk.DISABLED)
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = MusicToInstrumentGUI(root)
    root.mainloop()