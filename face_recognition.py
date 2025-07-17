import os
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.manifold import TSNE
import ttkbootstrap as tb  
import cv2


class EnhancedFaceRecognitionApp:
    def __init__(self, root):
        # Use ttkbootstrap's themed root
        self.root = tb.Window(themename="superhero")  # Try "superhero", "cosmo", "minty", etc.
        self.root.title("Advanced Face Recognition System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize variables
        self.data_path = r"C:\Users\Noure\Downloads\Machine Ass3\faces id"
        self.num_subjects = 40
        self.images_per_subject = 10
        self.image_size = (112, 92)
        self.total_pixels = 112 * 92
        self.data_matrix = None
        self.labels = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.current_image_idx = 0
        self.pca_model = None
        self.lda_model = None
        
        # Style configuration (use ttkbootstrap styles)
        self.style = tb.Style()
        # Use colorful styles for buttons and labels
        self.button_style = "success.Outline.TButton"
        self.title_style = "primary.Inverse.TLabel"
        self.frame_style = "info.TFrame"

        # Create GUI elements
        self.create_widgets()
        
        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
    def create_widgets(self):
        # Main container
        main_frame = tb.Frame(self.root, padding=10, style=self.frame_style)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        control_frame = tb.LabelFrame(main_frame, text="Controls", padding=15, bootstyle="danger")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Title
        tb.Label(control_frame, text="Face Recognition System", style=self.title_style, font=("Arial", 16, "bold")).pack(pady=(0, 15))
        
        # Path selection
        path_frame = tb.Frame(control_frame)
        path_frame.pack(fill=tk.X, pady=(0, 15))
        tb.Label(path_frame, text="Dataset Path:", bootstyle="info").pack(anchor=tk.W)
        self.path_entry = tb.Entry(path_frame, width=40)
        self.path_entry.insert(0, self.data_path)
        self.path_entry.pack(fill=tk.X, pady=(0, 5))
        tb.Button(path_frame, text="Browse", command=self.browse_path, bootstyle="warning-outline").pack(pady=(0, 5))
        
        # Data processing buttons
        tb.Label(control_frame, text="Data Processing:", style=self.title_style).pack(anchor=tk.W, pady=(5, 0))
        tb.Button(control_frame, text="1. Generate Data Matrix & Labels", 
                  command=self.generate_data_matrix, bootstyle=self.button_style).pack(fill=tk.X, pady=5)
        tb.Button(control_frame, text="2. Split Dataset", 
                  command=self.split_dataset, bootstyle=self.button_style).pack(fill=tk.X, pady=5)
        
        # Analysis buttons
        tb.Label(control_frame, text="Analysis Methods:", style=self.title_style).pack(anchor=tk.W, pady=(10, 0))
        tb.Button(control_frame, text="3. Run PCA Classification", 
                  command=self.run_pca_classification, bootstyle=self.button_style).pack(fill=tk.X, pady=5)
        tb.Button(control_frame, text="4. Run LDA Classification", 
                  command=self.run_lda_classification, bootstyle=self.button_style).pack(fill=tk.X, pady=5)
        tb.Button(control_frame, text="5. Tune KNN Classifier", 
                  command=self.tune_knn_classifier, bootstyle=self.button_style).pack(fill=tk.X, pady=5)
        tb.Button(control_frame, text="6. PCA Projection + 1-NN", 
                  command=self.run_pca_projection_and_1nn, bootstyle=self.button_style).pack(fill=tk.X, pady=5)
        tb.Button(control_frame, text="7. Visualize t-SNE", 
                  command=self.visualize_tsne, bootstyle=self.button_style).pack(fill=tk.X, pady=5)
        
        # Image navigation
        tb.Label(control_frame, text="Image Navigation:", style=self.title_style).pack(anchor=tk.W, pady=(10, 0))
        nav_frame = tb.Frame(control_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        tb.Button(nav_frame, text="Previous", command=self.show_previous_image, bootstyle="secondary").pack(side=tk.LEFT, expand=True)
        tb.Button(nav_frame, text="Next", command=self.show_next_image, bootstyle="secondary").pack(side=tk.LEFT, expand=True)
        
        # Status area
        tb.Label(control_frame, text="Status:", style=self.title_style).pack(anchor=tk.W, pady=(10, 0))
        self.status_text = tk.Text(control_frame, height=10, width=40, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Results
        result_frame = tb.LabelFrame(main_frame, text="Results & Visualizations", padding=15, bootstyle="info")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook for multiple tabs
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Data Visualization
        self.data_viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_viz_tab, text="Data Visualization")
        self.create_data_viz_tab()
        
        # Tab 2: Classification Results
        self.classification_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.classification_tab, text="Classification")
        self.create_classification_tab()
        
        # Tab 3: Image Display
        self.image_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.image_tab, text="Image Browser")
        self.create_image_tab()
        
    def create_data_viz_tab(self):
        # Main figure for data visualization
        self.data_figure = Figure(figsize=(8, 6), dpi=100)
        self.data_canvas = FigureCanvasTkAgg(self.data_figure, master=self.data_viz_tab)
        self.data_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Info text below the figure
        self.data_info = tk.Text(self.data_viz_tab, height=5, width=80, wrap=tk.WORD)
        self.data_info.pack(fill=tk.X, pady=(10, 0))
        
    def create_classification_tab(self):
        # Figure for classification results
        self.class_figure = Figure(figsize=(8, 6), dpi=100)
        self.class_canvas = FigureCanvasTkAgg(self.class_figure, master=self.classification_tab)
        self.class_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Confusion matrix figure
        self.confusion_figure = Figure(figsize=(8, 6), dpi=100)
        self.confusion_canvas = FigureCanvasTkAgg(self.confusion_figure, master=self.classification_tab)
        self.confusion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_image_tab(self):
        # Frame for image display and controls
        image_frame = ttk.Frame(self.image_tab)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image display
        self.image_figure = Figure(figsize=(6, 6), dpi=100)
        self.image_ax = self.image_figure.add_subplot(111)
        self.image_canvas = FigureCanvasTkAgg(self.image_figure, master=image_frame)
        self.image_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Image info
        self.image_info = ttk.Label(image_frame, text="No image loaded", style='Title.TLabel')
        self.image_info.pack(pady=(10, 0))
        
        # Navigation buttons
        nav_frame = ttk.Frame(image_frame)
        nav_frame.pack(pady=10)
        tb.Button(nav_frame, text="First", command=self.show_first_image, bootstyle="info").pack(side=tk.LEFT, padx=5)
        tb.Button(nav_frame, text="Previous", command=self.show_previous_image, bootstyle="info").pack(side=tk.LEFT, padx=5)
        tb.Button(nav_frame, text="Next", command=self.show_next_image, bootstyle="info").pack(side=tk.LEFT, padx=5)
        tb.Button(nav_frame, text="Last", command=self.show_last_image, bootstyle="info").pack(side=tk.LEFT, padx=5)
        
        # Random image button
        tb.Button(image_frame, text="Random Image", command=self.show_random_image, bootstyle="success").pack(pady=5)
        
        # Subject selection
        subject_frame = ttk.Frame(image_frame)
        subject_frame.pack(pady=5)
        ttk.Label(subject_frame, text="Subject:").pack(side=tk.LEFT)
        self.subject_var = tk.IntVar()
        self.subject_spin = ttk.Spinbox(subject_frame, from_=1, to=self.num_subjects, 
                                       textvariable=self.subject_var, width=5,
                                       command=self.load_subject_images)
        self.subject_spin.pack(side=tk.LEFT, padx=5)
        
        # Image number selection
        img_frame = ttk.Frame(image_frame)
        img_frame.pack(pady=5)
        ttk.Label(img_frame, text="Image:").pack(side=tk.LEFT)
        self.img_var = tk.IntVar()
        self.img_spin = ttk.Spinbox(img_frame, from_=1, to=self.images_per_subject, 
                                   textvariable=self.img_var, width=5,
                                   command=self.load_specific_image)
        self.img_spin.pack(side=tk.LEFT, padx=5)
        
       
    
    def browse_path(self):
        path = filedialog.askdirectory()
        if path:
            self.data_path = path
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)
    
    def log_status(self, message):
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update()
    
    def clear_status(self):
        self.status_text.delete(1.0, tk.END)
    
    def generate_data_matrix(self):
        """Generate Data Matrix D and label vector y"""
        self.clear_status()
        try:
            total_images = self.num_subjects * self.images_per_subject
            self.data_matrix = np.zeros((total_images, self.total_pixels))
            self.labels = np.zeros(total_images, dtype=int)
            
            img_index = 0
            for subject_id in range(1, self.num_subjects + 1):
                subject_dir = os.path.join(self.data_path, f's{subject_id}')
                for img_num in range(1, self.images_per_subject + 1):
                    img_path = os.path.join(subject_dir, f'{img_num}.pgm')
                    try:
                        img = Image.open(img_path).convert('L')
                        img_array = np.array(img).flatten()
                        self.data_matrix[img_index] = img_array
                        self.labels[img_index] = subject_id
                        img_index += 1
                    except FileNotFoundError:
                        self.log_status(f"Warning: Missing image {img_path}")
                        continue
            
            self.log_status("Successfully generated Data Matrix and Labels!")
            self.log_status(f"Data Matrix shape: {self.data_matrix.shape}")
            self.log_status(f"Labels shape: {self.labels.shape}")
            
            # Display data info and visualization
            self.display_data_matrix_info()
            self.show_image(0)  # Show first image
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data matrix:\n{str(e)}")
    
    def display_data_matrix_info(self):
        """Display information about the data matrix with visualizations"""
        if self.data_matrix is None:
            return
            
        self.data_figure.clear()
        
        # Calculate statistics
        mean_face = np.mean(self.data_matrix, axis=0)
        std_face = np.std(self.data_matrix, axis=0)
        
        # Create subplots
        ax1 = self.data_figure.add_subplot(221)
        ax2 = self.data_figure.add_subplot(222)
        ax3 = self.data_figure.add_subplot(223)
        ax4 = self.data_figure.add_subplot(224)
        
        # Plot mean face
        ax1.imshow(mean_face.reshape(self.image_size), cmap='gray')
        ax1.set_title("Mean Face")
        ax1.axis('off')
        
        # Plot standard deviation face
        ax2.imshow(std_face.reshape(self.image_size), cmap='gray')
        ax2.set_title("Standard Deviation Face")
        ax2.axis('off')
        
        # Plot histogram of pixel values
        ax3.hist(self.data_matrix.flatten(), bins=50, color='blue', alpha=0.7)
        ax3.set_title("Pixel Value Distribution")
        ax3.set_xlabel("Pixel Intensity")
        ax3.set_ylabel("Frequency")
        
        # Plot first few eigenfaces if PCA has been computed
        if hasattr(self, 'pca_model') and self.pca_model is not None:
            for i in range(min(3, self.pca_model.components_.shape[0])):
                ax4.imshow(self.pca_model.components_[i].reshape(self.image_size), cmap='gray')
            ax4.set_title("Top Eigenfaces")
            ax4.axis('off')
        else:
            # Plot random sample faces instead
            sample_indices = np.random.choice(len(self.data_matrix), 3, replace=False)
            for i, idx in enumerate(sample_indices):
                ax4.imshow(self.data_matrix[idx].reshape(self.image_size), cmap='gray')
            ax4.set_title("Sample Faces")
            ax4.axis('off')
        
        self.data_figure.tight_layout()
        self.data_canvas.draw()
        
        # Update info text
        self.data_info.delete(1.0, tk.END)
        self.data_info.insert(tk.END, f"Data Matrix Dimensions: {self.data_matrix.shape}\n")
        self.data_info.insert(tk.END, f"Number of Subjects: {self.num_subjects}\n")
        self.data_info.insert(tk.END, f"Images per Subject: {self.images_per_subject}\n")
        self.data_info.insert(tk.END, f"Total Images: {len(self.data_matrix)}\n")
        self.data_info.insert(tk.END, f"Mean Pixel Value: {np.mean(self.data_matrix):.2f}\n")
        self.data_info.insert(tk.END, f"Standard Deviation: {np.std(self.data_matrix):.2f}\n")
    
    def split_dataset(self):
        """Split dataset into training (odd rows) and test (even rows) sets, 5 per person."""
        if self.data_matrix is None:
            messagebox.showerror("Error", "Please generate data matrix first!")
            return

        try:
            # Odd indices (0,2,4,...) for training, even (1,3,5,...) for testing
            train_indices = np.arange(0, self.data_matrix.shape[0], 2)
            test_indices = np.arange(1, self.data_matrix.shape[0], 2)

            self.X_train = self.data_matrix[train_indices]
            self.X_test = self.data_matrix[test_indices]
            self.y_train = self.labels[train_indices]
            self.y_test = self.labels[test_indices]

            self.log_status("Dataset split successfully!")
            self.log_status(f"Training set shape: {self.X_train.shape}")
            self.log_status(f"Test set shape: {self.X_test.shape}")
            self.log_status(f"Training subjects: {len(np.unique(self.y_train))}")
            self.log_status(f"Test subjects: {len(np.unique(self.y_test))}")

            # Update data visualization
            self.display_data_matrix_info()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to split dataset:\n{str(e)}")
    
    def run_pca_classification(self):
        """Perform PCA-based classification"""
        if self.X_train is None:
            messagebox.showerror("Error", "Please split dataset first!")
            return
        
        try:
            self.class_figure.clear()
            ax = self.class_figure.add_subplot(111)
            
            alphas = [0.8, 0.85, 0.9, 0.95]
            accuracies = []
            components = []
            best_accuracy = 0
            best_alpha = 0
            
            for alpha in alphas:
                self.pca_model = PCA(n_components=alpha, svd_solver='full')
                self.pca_model.fit(self.X_train)
                
                X_train_pca = self.pca_model.transform(self.X_train)
                X_test_pca = self.pca_model.transform(self.X_test)
                
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(X_train_pca, self.y_train)
                
                y_pred = knn.predict(X_test_pca)
                accuracy = accuracy_score(self.y_test, y_pred)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_alpha = alpha
                
                accuracies.append(accuracy)
                components.append(self.pca_model.n_components_)
                
                self.log_status(f"PCA (α={alpha}): Accuracy={accuracy:.4f}, Components={self.pca_model.n_components_}")
            
            # Plot results
            ax.plot(alphas, accuracies, marker='o', color='blue', label='Accuracy')
            ax.set_xlabel('Alpha (Variance Retention)')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'PCA Classification (Best α={best_alpha}, Acc={best_accuracy:.2%})')
            ax.grid(True)
            
            # Annotate points with component count
            for i, txt in enumerate(components):
                ax.annotate(f'{txt} comps', (alphas[i], accuracies[i]), 
                           textcoords="offset points", xytext=(0,10), ha='center')
            
            # Plot second y-axis for components
            ax2 = ax.twinx()
            ax2.plot(alphas, components, marker='s', color='red', linestyle='--', label='Components')
            ax2.set_ylabel('Number of Components')
            
            # Combine legends
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='lower right')
            
            self.class_figure.tight_layout()
            self.class_canvas.draw()
            
            # Plot confusion matrix for best model
            self.plot_confusion_matrix(best_alpha)
            
            # Update data visualization with eigenfaces
            self.display_data_matrix_info()
            
        except Exception as e:
            messagebox.showerror("Error", f"PCA Classification failed:\n{str(e)}")
    
    def plot_confusion_matrix(self, alpha=None, model_type='pca'):
        """Plot confusion matrix for the classification results"""
        self.confusion_figure.clear()
        ax = self.confusion_figure.add_subplot(111)
        
        if alpha is not None and model_type == 'pca':
            # Recompute PCA with best alpha
            self.pca_model = PCA(n_components=alpha, svd_solver='full')
            self.pca_model.fit(self.X_train)
            X_train_pca = self.pca_model.transform(self.X_train)
            X_test_pca = self.pca_model.transform(self.X_test)
            
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train_pca, self.y_train)
            y_pred = knn.predict(X_test_pca)
        elif model_type == 'lda':
            y_pred = self.lda_model.predict(self.X_test)
        else:
            return
            
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        self.confusion_figure.tight_layout()
        self.confusion_canvas.draw()
    
    def run_lda_classification(self):
        """Perform LDA-based classification"""
        if self.X_train is None:
            messagebox.showerror("Error", "Please split dataset first!")
            return

        try:
            n_classes = len(np.unique(self.y_train))
            n_features = self.X_train.shape[1]
            lda_n_components = min(n_classes - 1, n_features)
            self.lda_model = LinearDiscriminantAnalysis(n_components=1)
            self.lda_model.fit(self.X_train, self.y_train)

            X_train_lda = self.lda_model.transform(self.X_train)
            X_test_lda = self.lda_model.transform(self.X_test)

            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train_lda, self.y_train)

            y_pred = knn.predict(X_test_lda)
            accuracy = accuracy_score(self.y_test, y_pred)

            self.log_status(f"LDA Accuracy: {accuracy:.4f}")
            self.log_status(f"LDA Components: {lda_n_components}")

            # Visualize LDA components
            self.class_figure.clear()
            ax = self.class_figure.add_subplot(111)

            if X_test_lda.shape[1] >= 2:
                # 2D scatter plot if at least 2 components
                scatter = ax.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=self.y_test, alpha=0.6)
                ax.set_xlabel('LDA Component 1')
                ax.set_ylabel('LDA Component 2')
                ax.set_title(f'LDA Projection (Accuracy: {accuracy:.2%})')
                legend = ax.legend(*scatter.legend_elements(), title="Subjects")
                ax.add_artist(legend)
            else:
                # 1D plot if only 1 component
                for label in np.unique(self.y_test):
                    ax.scatter(X_test_lda[self.y_test == label, 0], 
                               np.zeros_like(X_test_lda[self.y_test == label, 0]) + label, 
                               label=f"Class {label}", alpha=0.6)
                ax.set_xlabel('LDA Component 1')
                ax.set_yticks(np.unique(self.y_test))
                ax.set_yticklabels([f"Class {label}" for label in np.unique(self.y_test)])
                ax.set_title(f'LDA Projection (1D, Accuracy: {accuracy:.2%})')
                ax.legend()

            self.class_figure.tight_layout()
            self.class_canvas.draw()
            
            # Plot confusion matrix
            self.plot_confusion_matrix(model_type='lda')

        except Exception as e:
            messagebox.showerror("Error", f"LDA Classification failed:\n{str(e)}")
    
    def tune_knn_classifier(self):
        """Tune KNN classifier with different k values"""
        if self.X_train is None:
            messagebox.showerror("Error", "Please split dataset first!")
            return

        try:
            k_values = [1, 3, 5, 7]

            # PCA results
            pca = PCA(n_components=0.9, svd_solver='full')
            pca.fit(self.X_train)
            X_train_pca = pca.transform(self.X_train)
            X_test_pca = pca.transform(self.X_test)

            pca_accuracies = []
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train_pca, self.y_train)
                y_pred = knn.predict(X_test_pca)
                accuracy = accuracy_score(self.y_test, y_pred)
                pca_accuracies.append(accuracy)
                self.log_status(f"PCA KNN (k={k}): Accuracy={accuracy:.4f}")

            # LDA results
            n_classes = len(np.unique(self.y_train))
            n_features = self.X_train.shape[1]
            lda_n_components = min(n_classes - 1, n_features)
            lda = LinearDiscriminantAnalysis(n_components=1)
            lda.fit(self.X_train, self.y_train)
            X_train_lda = lda.transform(self.X_train)
            X_test_lda = lda.transform(self.X_test)

            lda_accuracies = []
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train_lda, self.y_train)
                y_pred = knn.predict(X_test_lda)
                accuracy = accuracy_score(self.y_test, y_pred)
                lda_accuracies.append(accuracy)
                self.log_status(f"LDA KNN (k={k}): Accuracy={accuracy:.4f}")

            # Plot results
            self.class_figure.clear()
            ax = self.class_figure.add_subplot(111)

            ax.plot(k_values, pca_accuracies, label='PCA (α=0.9)', marker='o')
            ax.plot(k_values, lda_accuracies, label='LDA', marker='o')

            ax.set_xlabel('Number of Neighbors (k)')
            ax.set_ylabel('Accuracy')
            ax.set_title('KNN Classifier Tuning')
            ax.legend()
            ax.grid(True)

            # Tabulate results in the status log
            self.log_status("\nKNN Accuracy Table:")
            self.log_status("k\tPCA (α=0.9)\tLDA")
            for i, k in enumerate(k_values):
                self.log_status(f"{k}\t{pca_accuracies[i]:.4f}\t\t{lda_accuracies[i]:.4f}")

        except Exception as e:
            messagebox.showerror("Error", f"KNN Tuning failed:\n{str(e)}")
    
    def run_pca_projection_and_1nn(self):
        """Compute PCA projection matrix for various alphas and classify using 1-NN."""
        if self.X_train is None:
            messagebox.showerror("Error", "Please split dataset first!")
            return

        try:
            self.class_figure.clear()
            ax = self.class_figure.add_subplot(111)

            alphas = [0.8, 0.85, 0.9, 0.95]
            accuracies = []
            components = []
            best_accuracy = 0
            best_alpha = 0

            for alpha in alphas:
                # Compute PCA projection matrix U for given alpha
                pca = PCA(n_components=alpha, svd_solver='full')
                pca.fit(self.X_train)
                U = pca.components_.T  # Projection matrix

                # Project training and test sets
                X_train_proj = (self.X_train - pca.mean_) @ U
                X_test_proj = (self.X_test - pca.mean_) @ U

                # 1-NN classifier
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(X_train_proj, self.y_train)
                y_pred = knn.predict(X_test_proj)
                accuracy = accuracy_score(self.y_test, y_pred)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_alpha = alpha

                accuracies.append(accuracy)
                components.append(U.shape[1])
                self.log_status(f"PCA α={alpha}: Accuracy={accuracy:.4f}, Components={U.shape[1]}")

            # Plot results
            ax.plot(alphas, accuracies, marker='o', color='blue', label='Accuracy')
            ax.set_xlabel('Alpha (Variance Retention)')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'PCA Projection + 1-NN (Best α={best_alpha}, Acc={best_accuracy:.2%})')
            ax.grid(True)
            
            # Add components information
            ax2 = ax.twinx()
            ax2.plot(alphas, components, marker='s', color='red', linestyle='--', label='Components')
            ax2.set_ylabel('Number of Components')
            
            # Combine legends
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='lower right')
            
            for i, txt in enumerate(components):
                ax.annotate(f'{txt} comps', (alphas[i], accuracies[i]), 
                           textcoords="offset points", xytext=(0,10), ha='center')

            self.class_figure.tight_layout()
            self.class_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"PCA Projection + 1-NN failed:\n{str(e)}")
    
    def visualize_tsne(self):
        """Visualize the dataset using t-SNE dimensionality reduction"""
        if self.data_matrix is None:
            messagebox.showerror("Error", "Please generate data matrix first!")
            return

        try:
            # Use a subset of the data for faster computation
            sample_size = min(200, len(self.data_matrix))
            indices = np.random.choice(len(self.data_matrix), sample_size, replace=False)
            X_sample = self.data_matrix[indices]
            y_sample = self.labels[indices]

            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X_sample)

            self.class_figure.clear()
            ax = self.class_figure.add_subplot(111)

            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, alpha=0.7)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_title('t-SNE Visualization of Face Dataset')
            legend = ax.legend(*scatter.legend_elements(), title="Subjects")
            ax.add_artist(legend)
            
            self.class_figure.tight_layout()
            self.class_canvas.draw()

            self.log_status("t-SNE visualization completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"t-SNE visualization failed:\n{str(e)}")
    
    def show_image(self, idx):
        """Display the image at the given index"""
        if self.data_matrix is None:
            return
            
        self.current_image_idx = idx
        self.image_ax.clear()
        
        # Display the image
        img_array = self.data_matrix[idx].reshape(self.image_size)
        self.image_ax.imshow(img_array, cmap='gray')
        self.image_ax.set_title(f"Image {idx+1} (Subject {self.labels[idx]})")
        self.image_ax.axis('off')
        
        # Update image info
        self.image_info.config(text=f"Image {idx+1}/{len(self.data_matrix)} - Subject {self.labels[idx]}")
        
        # Update spinboxes
        subject_id = self.labels[idx]
        image_num = (idx % self.images_per_subject) + 1  # Assuming ordered by subject
        
        self.subject_var.set(subject_id)
        self.img_var.set(image_num)
        
        self.image_figure.tight_layout()
        self.image_canvas.draw()
    
    def show_next_image(self):
        if self.data_matrix is None:
            return
        new_idx = (self.current_image_idx + 1) % len(self.data_matrix)
        self.show_image(new_idx)
    
    def show_previous_image(self):
        if self.data_matrix is None:
            return
        new_idx = (self.current_image_idx - 1) % len(self.data_matrix)
        self.show_image(new_idx)
    
    def show_first_image(self):
        if self.data_matrix is None:
            return
        self.show_image(0)
    
    def show_last_image(self):
        if self.data_matrix is None:
            return
        self.show_image(len(self.data_matrix) - 1)
    
    def show_random_image(self):
        if self.data_matrix is None:
            return
        random_idx = np.random.randint(0, len(self.data_matrix))
        self.show_image(random_idx)
    
    def load_subject_images(self):
        """Load images for the selected subject"""
        if self.data_matrix is None:
            return
            
        subject_id = self.subject_var.get()
        if subject_id < 1 or subject_id > self.num_subjects:
            return
            
        # Find first image of this subject
        subject_idx = (subject_id - 1) * self.images_per_subject
        self.show_image(subject_idx)
    
    def load_specific_image(self):
        """Load specific image of the current subject"""
        if self.data_matrix is None:
            return
            
        subject_id = self.subject_var.get()
        image_num = self.img_var.get()
        
        if (subject_id < 1 or subject_id > self.num_subjects or 
            image_num < 1 or image_num > self.images_per_subject):
            return
            
        # Calculate image index
        subject_idx = (subject_id - 1) * self.images_per_subject
        image_idx = subject_idx + (image_num - 1)
        
        self.show_image(image_idx)

    def compute_projection_matrices(self):
        """Compute PCA projection matrix U for various alphas and project train/test sets."""
        if self.X_train is None:
            messagebox.showerror("Error", "Please split dataset first!")
            return

        alphas = [0.8, 0.85, 0.9, 0.95]
        projections = {}

        for alpha in alphas:
            # Fit PCA to retain 'alpha' variance
            pca = PCA(n_components=alpha, svd_solver='full')
            pca.fit(self.X_train)
            U = pca.components_.T  # Projection matrix U (features x components)

            # Project training and test sets using U
            X_train_proj = (self.X_train - pca.mean_) @ U
            X_test_proj = (self.X_test - pca.mean_) @ U

            projections[alpha] = {
                'U': U,
                'X_train_proj': X_train_proj,
                'X_test_proj': X_test_proj
            }
            self.log_status(f"Alpha={alpha}: U shape={U.shape}, X_train_proj shape={X_train_proj.shape}, X_test_proj shape={X_test_proj.shape}")

        return projections

    def manual_lda_projection(self):
        """
        Perform multiclass LDA as described in the pseudocode.
        Returns the LDA projection matrix U (10304 x 39) and projected train/test sets.
        """
        if self.X_train is None:
            messagebox.showerror("Error", "Please split dataset first!")
            return

        X = self.X_train
        y = self.y_train
        n_features = X.shape[1]
        class_labels = np.unique(y)
        m = len(class_labels)
        overall_mean = np.mean(X, axis=0)

        # Step i: Calculate mean vector for every class
        class_means = {}
        n_k = {}
        for label in class_labels:
            X_k = X[y == label]
            class_means[label] = np.mean(X_k, axis=0)
            n_k[label] = X_k.shape[0]

        # Step ii: Compute Sb (between-class scatter)
        Sb = np.zeros((n_features, n_features))
        for label in class_labels:
            mean_diff = (class_means[label] - overall_mean).reshape(-1, 1)
            Sb += n_k[label] * (mean_diff @ mean_diff.T)

        # Step iii: Compute Sw (within-class scatter)
        Sw = np.zeros((n_features, n_features))
        for label in class_labels:
            X_k = X[y == label]
            mean_k = class_means[label]
            S_k = (X_k - mean_k).T @ (X_k - mean_k)
            Sw += S_k

        # Step iv: Solve the generalized eigenvalue problem for inv(Sw) * Sb
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvecs = eigvecs[:, idx]
        # Take the first 39 dominant eigenvectors
        U = eigvecs[:, :m-1]  # shape: (features, 39)

        # Project train and test sets
        X_train_proj = (self.X_train - overall_mean) @ U
        X_test_proj = (self.X_test - overall_mean) @ U

        self.log_status(f"LDA: U shape={U.shape}, X_train_proj shape={X_train_proj.shape}, X_test_proj shape={X_test_proj.shape}")

        return U, X_train_proj, X_test_proj

    def run_manual_lda_and_1nn(self):
        """Run manual LDA and report 1-NN accuracy."""
        result = self.manual_lda_projection()
        if result is None:
            return
        U, X_train_proj, X_test_proj = result

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train_proj, self.y_train)
        y_pred = knn.predict(X_test_proj)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.log_status(f"Manual LDA + 1-NN: Accuracy={accuracy:.4f}")

    def compare_lda_pca(self):
        """Compare LDA and best PCA (by accuracy) using 1-NN."""
        # Run LDA
        result = self.manual_lda_projection()
        if result is None:
            return
        _, X_train_lda, X_test_lda = result
        knn_lda = KNeighborsClassifier(n_neighbors=1)
        knn_lda.fit(X_train_lda, self.y_train)
        y_pred_lda = knn_lda.predict(X_test_lda)
        lda_acc = accuracy_score(self.y_test, y_pred_lda)
        self.log_status(f"[Comparison] Manual LDA + 1-NN: Accuracy={lda_acc:.4f}")

        # Run PCA for all alphas and pick the best
        alphas = [0.8, 0.85, 0.9, 0.95]
        best_pca_acc = 0
        best_alpha = None
        for alpha in alphas:
            pca = PCA(n_components=alpha, svd_solver='full')
            pca.fit(self.X_train)
            U_pca = pca.components_.T
            X_train_pca = (self.X_train - pca.mean_) @ U_pca
            X_test_pca = (self.X_test - pca.mean_) @ U_pca
            knn_pca = KNeighborsClassifier(n_neighbors=1)
            knn_pca.fit(X_train_pca, self.y_train)
            y_pred_pca = knn_pca.predict(X_test_pca)
            pca_acc = accuracy_score(self.y_test, y_pred_pca)
            if pca_acc > best_pca_acc:
                best_pca_acc = pca_acc
                best_alpha = alpha
        self.log_status(f"[Comparison] Best PCA + 1-NN (alpha={best_alpha}): Accuracy={best_pca_acc:.4f}")

        # Print summary
        self.log_status(f"Summary:\nLDA Accuracy: {lda_acc:.4f}\nBest PCA Accuracy: {best_pca_acc:.4f} (alpha={best_alpha})")

    def show_non_faces_images(self, idx=0):
        """Display images from the non-faces folder in the image tab."""
        non_faces_path = r"C:\Users\Noure\Downloads\Machine Ass3\non faces images"
        if not os.path.exists(non_faces_path):
            messagebox.showerror("Error", f"Folder not found:\n{non_faces_path}")
            return

        # List all image files (support .jpg, .png, .pgm, etc.)
        image_files = [f for f in os.listdir(non_faces_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.pgm'))]
        if not image_files:
            messagebox.showinfo("Info", "No images found in the non-faces folder.")
            return

        self.non_faces_path = non_faces_path
        self.non_faces_files = image_files
        self.non_faces_idx = idx % len(image_files)

        img_path = os.path.join(non_faces_path, self.non_faces_files[self.non_faces_idx])
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)

        self.image_ax.clear()
        self.image_ax.imshow(img_array, cmap='gray')
        self.image_ax.set_title(f"Non-Face Image: {self.non_faces_files[self.non_faces_idx]}")
        self.image_ax.axis('off')
        self.image_info.config(text=f"Non-Face Image: {self.non_faces_files[self.non_faces_idx]} ({self.non_faces_idx+1}/{len(self.non_faces_files)})")
        self.image_figure.tight_layout()
        self.image_canvas.draw()

        # Update navigation buttons for non-face images
        self.update_non_face_navigation()

    def update_non_face_navigation(self):
        """Update the state of navigation buttons for non-face images."""
        if not self.non_faces_files:
            return

        # Enable/disable navigation buttons based on the index
        if len(self.non_faces_files) == 1:
            # Only one image, disable both buttons
            self.btn_prev_non_face.config(state=tk.DISABLED)
            self.btn_next_non_face.config(state=tk.DISABLED)
        elif self.non_faces_idx == 0:
            # First image, disable previous button
            self.btn_prev_non_face.config(state=tk.DISABLED)
            self.btn_next_non_face.config(state=tk.NORMAL)
        elif self.non_faces_idx == len(self.non_faces_files) - 1:
            # Last image, disable next button
            self.btn_next_non_face.config(state=tk.DISABLED)
            self.btn_prev_non_face.config(state=tk.NORMAL)
        else:
            # Middle images, enable both buttons
            self.btn_prev_non_face.config(state=tk.NORMAL)
            self.btn_next_non_face.config(state=tk.NORMAL)

    def show_next_non_face(self):
        if self.non_faces_files:
            self.non_faces_idx = (self.non_faces_idx + 1) % len(self.non_faces_files)
            self.show_non_faces_images(self.non_faces_idx)

    def show_prev_non_face(self):
        if self.non_faces_files:
            self.non_faces_idx = (self.non_faces_idx - 1) % len(self.non_faces_files)
            self.show_non_faces_images(self.non_faces_idx)


if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedFaceRecognitionApp(root)
    root.mainloop()