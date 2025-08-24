# src/gui.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import os
import threading
from skimage import color

# Import our modules
from src_model import SegColorNet
from src_utils import load_checkpoint

class ColorApp:
    """
    GUI Application for Semantic Colorization
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Semantic Colorizer - Deep Learning Image Colorization")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Initialize variables
        self.current_image = None
        self.original_image = None
        self.segmentation_mask = None
        self.colorized_image = None
        self.selected_region = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup UI
        self.setup_ui()

        # Load model in background
        self.load_model_async()

    def setup_ui(self):
        """Setup the user interface"""
        # Create main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(
            main_frame, 
            text="Semantic Colorizer", 
            font=('Arial', 20, 'bold'),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=(0, 20))

        # Button frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=(0, 20))

        # Load Image Button
        self.load_btn = tk.Button(
            button_frame,
            text="📁 Load Image",
            command=self.load_image,
            font=('Arial', 12),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            relief='raised',
            bd=2
        )
        self.load_btn.pack(side='left', padx=(0, 10))

        # Process Button
        self.process_btn = tk.Button(
            button_frame,
            text="🎨 Colorize Image",
            command=self.colorize_image,
            font=('Arial', 12),
            bg='#2196F3',
            fg='white',
            padx=20,
            pady=10,
            relief='raised',
            bd=2,
            state='disabled'
        )
        self.process_btn.pack(side='left', padx=(0, 10))

        # Save Button
        self.save_btn = tk.Button(
            button_frame,
            text="💾 Save Result",
            command=self.save_image,
            font=('Arial', 12),
            bg='#FF9800',
            fg='white',
            padx=20,
            pady=10,
            relief='raised',
            bd=2,
            state='disabled'
        )
        self.save_btn.pack(side='left', padx=(0, 10))

        # Reset Button
        self.reset_btn = tk.Button(
            button_frame,
            text="🔄 Reset",
            command=self.reset_interface,
            font=('Arial', 12),
            bg='#f44336',
            fg='white',
            padx=20,
            pady=10,
            relief='raised',
            bd=2
        )
        self.reset_btn.pack(side='left')

        # Progress bar
        self.progress_frame = tk.Frame(main_frame, bg='#f0f0f0')
        self.progress_frame.pack(fill='x', pady=(0, 10))

        self.progress_label = tk.Label(
            self.progress_frame,
            text="Loading model...",
            font=('Arial', 10),
            bg='#f0f0f0',
            fg='#666666'
        )
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate',
            length=400
        )
        self.progress_bar.pack(pady=(5, 0))

        # Image display frame
        self.image_frame = tk.Frame(main_frame, bg='#ffffff', relief='sunken', bd=2)
        self.image_frame.pack(fill='both', expand=True)

        # Create canvas for images
        self.setup_image_canvases()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please load an image to begin")
        self.status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=('Arial', 10),
            bg='#e0e0e0',
            fg='#333333',
            anchor='w',
            padx=10
        )
        self.status_bar.pack(side='bottom', fill='x')

    def setup_image_canvases(self):
        """Setup canvases for displaying images"""
        # Original image canvas
        original_frame = tk.LabelFrame(
            self.image_frame,
            text="Original Image",
            font=('Arial', 12, 'bold'),
            bg='#ffffff',
            fg='#333333'
        )
        original_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)

        self.original_canvas = tk.Canvas(
            original_frame,
            bg='#f8f8f8',
            relief='sunken',
            bd=1
        )
        self.original_canvas.pack(fill='both', expand=True, padx=5, pady=5)

        # Segmentation canvas
        seg_frame = tk.LabelFrame(
            self.image_frame,
            text="Segmentation",
            font=('Arial', 12, 'bold'),
            bg='#ffffff',
            fg='#333333'
        )
        seg_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)

        self.seg_canvas = tk.Canvas(
            seg_frame,
            bg='#f8f8f8',
            relief='sunken',
            bd=1
        )
        self.seg_canvas.pack(fill='both', expand=True, padx=5, pady=5)

        # Colorized image canvas
        result_frame = tk.LabelFrame(
            self.image_frame,
            text="Colorized Result",
            font=('Arial', 12, 'bold'),
            bg='#ffffff',
            fg='#333333'
        )
        result_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        self.result_canvas = tk.Canvas(
            result_frame,
            bg='#f8f8f8',
            relief='sunken',
            bd=1
        )
        self.result_canvas.pack(fill='both', expand=True, padx=5, pady=5)

    def load_model_async(self):
        """Load the model in a separate thread"""
        def load_model():
            try:
                self.progress_bar.start(10)
                self.status_var.set("Loading AI model...")

                # Initialize model
                self.model = SegColorNet(n_classes=2, pretrained=True).to(self.device)

                # Try to load trained weights
                checkpoint_paths = [
                    'outputs/checkpoints/best_model.pth',
                    'outputs/checkpoints/recent_model.pth',
                    'outputs/checkpoints/final_model.pth'
                ]

                model_loaded = False
                for checkpoint_path in checkpoint_paths:
                    if os.path.exists(checkpoint_path):
                        try:
                            epoch, loss = load_checkpoint(self.model, None, checkpoint_path, self.device)
                            self.status_var.set(f"Model loaded from {checkpoint_path} (epoch {epoch})")
                            model_loaded = True
                            break
                        except Exception as e:
                            continue

                if not model_loaded:
                    self.status_var.set("Using untrained model (no checkpoint found)")

                self.model.eval()

                # Hide progress bar
                self.progress_bar.stop()
                self.progress_bar.pack_forget()
                self.progress_label.pack_forget()
                self.progress_frame.pack_forget()

                self.status_var.set("Model ready - Load an image to begin colorization")

            except Exception as e:
                self.progress_bar.stop()
                self.status_var.set(f"Error loading model: {str(e)}")
                messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")

        # Start loading in background thread
        threading.Thread(target=load_model, daemon=True).start()

    def load_image(self):
        """Load an image file"""
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.gif'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]

        filename = filedialog.askopenfilename(
            title="Select an image",
            filetypes=file_types
        )

        if filename:
            try:
                self.status_var.set("Loading image...")

                # Load image using OpenCV
                img = cv2.imread(filename)
                if img is None:
                    raise ValueError("Could not load image file")

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize for display and processing
                self.original_image = cv2.resize(img, (256, 256))
                self.current_image = self.original_image.copy()

                # Display original image
                self.display_image(self.original_image, self.original_canvas)

                # Clear other canvases
                self.seg_canvas.delete('all')
                self.result_canvas.delete('all')

                # Enable process button
                self.process_btn.config(state='normal')

                self.status_var.set(f"Image loaded: {os.path.basename(filename)}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_var.set("Error loading image")

    def display_image(self, image_array, canvas):
        """Display numpy array as image on canvas"""
        try:
            # Convert numpy array to PIL Image
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)

            pil_image = Image.fromarray(image_array)

            # Resize to fit canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                # Calculate aspect ratio preserving size
                img_width, img_height = pil_image.size
                aspect_ratio = img_width / img_height

                if canvas_width / canvas_height > aspect_ratio:
                    # Canvas is wider, fit to height
                    new_height = min(canvas_height - 10, img_height)
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Canvas is taller, fit to width
                    new_width = min(canvas_width - 10, img_width)
                    new_height = int(new_width / aspect_ratio)

                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)

            # Clear canvas and display image
            canvas.delete('all')
            canvas.create_image(
                canvas.winfo_width() // 2,
                canvas.winfo_height() // 2,
                anchor='center',
                image=photo
            )

            # Keep a reference to prevent garbage collection
            canvas.image = photo

        except Exception as e:
            print(f"Error displaying image: {e}")

    def colorize_image(self):
        """Perform image colorization"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if self.model is None:
            messagebox.showwarning("Warning", "Model is still loading. Please wait.")
            return

        try:
            self.status_var.set("Processing image...")
            self.process_btn.config(state='disabled')

            # Convert image to LAB color space
            img_normalized = self.current_image / 255.0
            img_lab = color.rgb2lab(img_normalized)

            # Extract L channel
            L_channel = img_lab[:, :, 0:1] / 100.0  # Normalize to [0, 1]

            # Convert to tensor
            L_tensor = torch.from_numpy(L_channel.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

            # Perform inference
            with torch.no_grad():
                seg_logits, color_ab = self.model(L_tensor)

                # Get segmentation
                seg_pred = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()

                # Get colorized image
                rgb_colorized = self.model.predict_colors(L_tensor)
                rgb_colorized_np = rgb_colorized[0].cpu().numpy().transpose(1, 2, 0)

                # Clip values to [0, 1] range
                rgb_colorized_np = np.clip(rgb_colorized_np, 0, 1)

            # Store results
            self.segmentation_mask = seg_pred
            self.colorized_image = rgb_colorized_np

            # Display segmentation
            self.display_segmentation(seg_pred)

            # Display colorized result
            self.display_image(rgb_colorized_np, self.result_canvas)

            # Enable save button
            self.save_btn.config(state='normal')

            self.status_var.set("Colorization completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to colorize image: {str(e)}")
            self.status_var.set("Error during colorization")

        finally:
            self.process_btn.config(state='normal')

    def display_segmentation(self, seg_mask):
        """Display segmentation mask"""
        try:
            # Create colorized segmentation for display
            seg_colored = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)

            # Color different classes
            seg_colored[seg_mask == 0] = [0, 0, 255]    # Background - Blue
            seg_colored[seg_mask == 1] = [255, 255, 0]  # Foreground - Yellow

            self.display_image(seg_colored, self.seg_canvas)

        except Exception as e:
            print(f"Error displaying segmentation: {e}")

    def save_image(self):
        """Save the colorized result"""
        if self.colorized_image is None:
            messagebox.showwarning("Warning", "No colorized image to save")
            return

        file_types = [
            ('PNG files', '*.png'),
            ('JPEG files', '*.jpg'),
            ('All files', '*.*')
        ]

        filename = filedialog.asksaveasfilename(
            title="Save colorized image",
            defaultextension='.png',
            filetypes=file_types
        )

        if filename:
            try:
                # Convert to PIL Image and save
                img_to_save = (self.colorized_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_to_save)
                pil_image.save(filename)

                self.status_var.set(f"Image saved: {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Image saved successfully as {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

    def reset_interface(self):
        """Reset the interface to initial state"""
        # Clear images
        self.current_image = None
        self.original_image = None
        self.segmentation_mask = None
        self.colorized_image = None

        # Clear canvases
        self.original_canvas.delete('all')
        self.seg_canvas.delete('all')
        self.result_canvas.delete('all')

        # Reset buttons
        self.process_btn.config(state='disabled')
        self.save_btn.config(state='disabled')

        # Reset status
        self.status_var.set("Interface reset - Load an image to begin")

    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = ColorApp(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()