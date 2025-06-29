import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import logging
from tkinterdnd2 import TkinterDnD

from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DefectPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Defect Probability Prediction on Production Line")
        self.root.geometry("1200x850")
        self.root.resizable(True, True)

        self.data_path = tk.StringVar(value=os.path.join('data', 'raw_data.csv'))
        self.target_column = tk.StringVar(value='is_defect')
        self.selected_model = tk.StringVar()
        self.tuning_metric = tk.StringVar(value='f1')
        self.best_model_selection_metric = tk.StringVar(value='Cross-Val Accuracy')

        self.processor = None
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.current_trained_models = {}
        self.loaded_prediction_model = None
        self.best_model_name = None
        self.best_model_performance = 0

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = (None,) * 6
        self.preprocessor = None
        self.feature_names = None
        self.original_feature_names_for_gui = None

        self._create_widgets()
        self._setup_initial_state()

    def _create_widgets(self):
        data_frame = ttk.LabelFrame(self.root, text="Data Configuration", padding="10")
        data_frame.pack(side="top", fill="x", padx=10, pady=5)

        ttk.Label(data_frame, text="CSV File Path:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(data_frame, textvariable=self.data_path, width=70).grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        ttk.Button(data_frame, text="Browse File", command=self._browse_file).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(data_frame, text="Target Column Name:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(data_frame, textvariable=self.target_column, width=30).grid(row=1, column=1, padx=5, pady=2, sticky="w")
        
        ttk.Button(data_frame, text="Load and Preprocess Data", command=self._load_and_preprocess_data).grid(row=1, column=2, padx=5, pady=2)
        data_frame.grid_columnconfigure(1, weight=1)

        model_frame = ttk.LabelFrame(self.root, text="Model Training and Evaluation", padding="10")
        model_frame.pack(side="top", fill="x", padx=10, pady=5)

        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.model_combobox = ttk.Combobox(model_frame, textvariable=self.selected_model, state="readonly")
        self.model_combobox.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.model_combobox.set('RandomForestClassifier')
        
        ttk.Button(model_frame, text="Train Base Model", command=self._train_selected_model).grid(row=0, column=2, padx=5, pady=2)
        ttk.Button(model_frame, text="Train All Base Models", command=self._train_all_models).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(model_frame, text="Tuning Metric:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Combobox(model_frame, textvariable=self.tuning_metric, values=['f1', 'recall', 'accuracy', 'roc_auc', 'precision'], state="readonly").grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.tuning_metric.set('f1')

        ttk.Button(model_frame, text="Tune Hyperparameters (Selected)", command=self._tune_selected_model).grid(row=1, column=2, padx=5, pady=2)
        ttk.Button(model_frame, text="Evaluate All Models", command=self._evaluate_all_models).grid(row=1, column=3, padx=5, pady=2)

        ttk.Label(model_frame, text="Best Model By:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.best_model_metric_combobox = ttk.Combobox(model_frame, textvariable=self.best_model_selection_metric,
                                                        values=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Cross-Val Accuracy'],
                                                        state="readonly")
        self.best_model_metric_combobox.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        self.best_model_metric_combobox.set('Cross-Val Accuracy')
        self.best_model_metric_combobox.bind("<<ComboboxSelected>>", self._update_best_model_on_metric_change)


        ttk.Button(model_frame, text="Save Best Model", command=self._save_best_model).grid(row=3, column=0, padx=5, pady=2)
        
        ttk.Button(model_frame, text="Load Model from Dropdown", command=self._load_model_for_prediction).grid(row=3, column=1, padx=5, pady=2, sticky="ew")
        ttk.Button(model_frame, text="Load Model from File", command=self._load_model_file_dialog).grid(row=3, column=2, padx=5, pady=2, columnspan=2, sticky="ew")

        model_frame.grid_columnconfigure(1, weight=1)


        results_frame = ttk.LabelFrame(self.root, text="Results and Visualization", padding="10")
        results_frame.pack(side="top", fill="both", expand=True, padx=10, pady=5)

        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill="both", expand=True)

        self.results_tab = ttk.Frame(self.notebook)
        self.plots_tab = ttk.Frame(self.notebook)
        self.prediction_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.results_tab, text="Results Table")
        self.notebook.add(self.plots_tab, text="Plots")
        self.notebook.add(self.prediction_tab, text="Prediction")

        self.results_tree = ttk.Treeview(self.results_tab, columns=('Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Cross-Val Accuracy'), show='headings')
        for col in self.results_tree['columns']:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120, anchor=tk.CENTER)
        self.results_tree.pack(fill="both", expand=True)

        # Scrollable Plots Tab
        self.plot_canvas = tk.Canvas(self.plots_tab)
        self.plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.plot_scrollbar = ttk.Scrollbar(self.plots_tab, orient="vertical", command=self.plot_canvas.yview)
        self.plot_scrollbar.pack(side=tk.RIGHT, fill="y")

        self.plot_canvas.configure(yscrollcommand=self.plot_scrollbar.set)
        self.plot_canvas.bind('<Configure>', lambda e: self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all")))
        
        self.scrollable_frame = ttk.Frame(self.plot_canvas)
        self.plot_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.scrollable_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        plot_buttons_frame = ttk.Frame(self.scrollable_frame)
        plot_buttons_frame.pack(side="top", fill="x", pady=5) # Changed to side="top" to be below the canvas
        ttk.Button(plot_buttons_frame, text="Confusion Matrix (Best Model)", command=self._plot_best_model_confusion_matrix).pack(side="left", padx=5, pady=2)
        ttk.Button(plot_buttons_frame, text="Model Comparison (Accuracy)", command=lambda: self._plot_model_comparison('Accuracy')).pack(side="left", padx=5, pady=2)
        ttk.Button(plot_buttons_frame, text="Model Comparison (F1-Score)", command=lambda: self._plot_model_comparison('F1-Score')).pack(side="left", padx=5, pady=2)
        ttk.Button(plot_buttons_frame, text="Model Comparison (ROC-AUC)", command=lambda: self._plot_model_comparison('ROC-AUC')).pack(side="left", padx=5, pady=2)
        ttk.Button(plot_buttons_frame, text="ROC Curve (Best Model)", command=self._plot_best_model_roc_curve).pack(side="left", padx=5, pady=2)
        ttk.Button(plot_buttons_frame, text="PR Curve (Best Model)", command=self._plot_best_model_pr_curve).pack(side="left", padx=5, pady=2)
        ttk.Button(plot_buttons_frame, text="Feature Importance (Best Model)", command=self._plot_best_model_feature_importance).pack(side="left", padx=5, pady=2)


        self._create_prediction_interface(self.prediction_tab)

    def _setup_initial_state(self):
        self.model_combobox['values'] = list(self.trainer.models.keys())
        if self.model_combobox['values']:
            self.selected_model.set(self.model_combobox['values'][0])

    def _browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename:
            self.data_path.set(filename)
            logging.info(f"File selected: {filename}")

    def _load_and_preprocess_data(self):
        data_path = self.data_path.get()
        target_col = self.target_column.get()
        if not data_path:
            messagebox.showerror("Error", "Please select a CSV file.")
            return

        self.processor = DataProcessor(data_path, target_column=target_col)
        if not self.processor.load_data():
            messagebox.showerror("Error", "Failed to load data. Check file path and format.")
            return

        if not self.processor.preprocess_data(use_smote=True):
            messagebox.showerror("Error", "Failed to preprocess data. Check your CSV file or target column.")
            return

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.processor.get_data()
        self.preprocessor = self.processor.get_preprocessor()
        self.original_feature_names_for_gui = self.processor.get_original_feature_names()
        self.feature_names = self.processor.get_feature_names()
        self.evaluator.set_feature_names(self.feature_names)

        if self.X_train is None:
            messagebox.showerror("Error", "Failed to preprocess data. Check your CSV file.")
            return
        
        messagebox.showinfo("Done", "Data successfully loaded and preprocessed!")
        logging.info("Data ready for model training.")
        if self.original_feature_names_for_gui is not None and len(self.original_feature_names_for_gui) > 0:
            self._update_prediction_input_fields(self.original_feature_names_for_gui)


    def _train_selected_model(self):
        if self.X_train is None:
            messagebox.showerror("Error", "Data not loaded and preprocessed. Please load and preprocess data first.")
            return

        model_name = self.selected_model.get()
        if not model_name:
            messagebox.showerror("Error", "Please select a model to train.")
            return

        try:
            model = self.trainer.train_model(model_name, self.X_train, self.y_train)
            if model:
                self.current_trained_models[model_name] = model
                messagebox.showinfo("Training Complete", f"Base model '{model_name}' successfully trained.")
                self._evaluate_model_and_update_results(model_name, model)
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred while training base model {model_name}: {e}")
            logging.error(f"Error training base model {model_name}: {e}", exc_info=True)

    def _train_all_models(self):
        if self.X_train is None:
            messagebox.showerror("Error", "Data not loaded and preprocessed. Please load and preprocess data first.")
            return
        
        try:
            self.trainer = ModelTrainer()
            self.current_trained_models = self.trainer.train_all_models(self.X_train, self.y_train)
            messagebox.showinfo("Training Complete", "All base models successfully trained.")
            self._evaluate_all_models()
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred while training all base models: {e}")
            logging.error(f"Error training all base models: {e}", exc_info=True)

    def _tune_selected_model(self):
        if self.X_train is None:
            messagebox.showerror("Error", "Data not loaded and preprocessed. Please load and preprocess data first.")
            return
        
        model_name = self.selected_model.get()
        if not model_name:
            messagebox.showerror("Error", "Please select a model for hyperparameter tuning.")
            return
        
        tuning_metric = self.tuning_metric.get()
        if not tuning_metric:
            messagebox.showerror("Error", "Please select a metric for hyperparameter tuning.")
            return

        try:
            tuned_model = self.trainer.tune_hyperparameters(model_name, self.X_train, self.y_train, scoring=tuning_metric)
            if tuned_model:
                tuned_model_key = f"{model_name}_tuned_by_{tuning_metric}"
                self.current_trained_models[tuned_model_key] = tuned_model
                messagebox.showinfo("Tuning Complete", f"Hyperparameter tuning for model '{model_name}' completed. Best parameters found.")
                self._evaluate_model_and_update_results(tuned_model_key, tuned_model)
        except Exception as e:
            messagebox.showerror("Tuning Error", f"An error occurred during hyperparameter tuning for model {model_name}: {e}")
            logging.error(f"Error during hyperparameter tuning for {model_name}: {e}", exc_info=True)


    def _evaluate_model_and_update_results(self, model_name, model):
        if self.X_train is None or self.X_val is None:
            messagebox.showerror("Error", "Evaluation data (X_train, X_val) is not available.")
            return

        X_train_val = np.vstack((self.X_train, self.X_val))
        y_train_val = np.concatenate((self.y_train, self.y_val))

        results = self.evaluator.evaluate_model(model_name, model, self.X_val, self.y_val, self.X_test, self.y_test, X_train_val, y_train_val)
        
        if results is not None:
            self._update_results_tree()
            self._update_best_model_info()
        else:
            logging.error(f"Failed to get evaluation results for model {model_name}.")


    def _evaluate_all_models(self):
        if not self.current_trained_models:
            messagebox.showerror("Error", "No trained models to evaluate. Please train models first.")
            return
        
        self.evaluator = ModelEvaluator()
        self.best_model_performance = 0
        self.best_model_name = None

        if self.feature_names is not None and len(self.feature_names) > 0:
            self.evaluator.set_feature_names(self.feature_names)
        else:
            logging.warning("Feature names not available for evaluator. Feature importance plots may be affected.")


        for model_name, model in self.current_trained_models.items():
            self._evaluate_model_and_update_results(model_name, model)
        
        messagebox.showinfo("Evaluation Complete", "All trained models evaluated. Results in the table.")
        self.notebook.select(self.results_tab)
        self._update_best_model_info()

    def _update_best_model_info(self):
        selected_metric = self.best_model_selection_metric.get()
        logging.info(f"Updating best model info based on metric: {selected_metric}")
        self.best_model_name, self.best_model_performance = self.evaluator.get_best_model_info(selected_metric)
        if self.best_model_name:
            logging.info(f"Current best model: {self.best_model_name} with {selected_metric}: {self.best_model_performance:.4f}")
        else:
            logging.info("No best model determined yet or results are empty.")

    def _update_best_model_on_metric_change(self, event=None):
        self._update_best_model_info()
        self._update_results_tree()

    def _update_results_tree(self):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        selected_metric = self.best_model_selection_metric.get()
        results_df = self.evaluator.get_results()

        if selected_metric in results_df.columns:
            results_df = results_df.sort_values(by=selected_metric, ascending=False)
        else:
            logging.warning(f"Sorting metric '{selected_metric}' not found in results. Displaying unsorted results.")

        for index, row in results_df.iterrows():
            formatted_row = [row['Model']] + [f"{val:.4f}" if isinstance(val, (float, np.floating)) else val for val in row[1:]]
            self.results_tree.insert("", "end", values=formatted_row)

    def _plot_best_model_confusion_matrix(self):
        logging.info(f"Attempting to plot confusion matrix. Best model: {self.best_model_name}")
        if self.best_model_name is None:
            messagebox.showerror("Error", "No best model to display confusion matrix. Train and evaluate models first.")
            return
        
        if self.best_model_name not in self.evaluator.last_predictions:
            messagebox.showerror("Error", "No saved test predictions for the current best model. Please re-evaluate models.")
            logging.error(f"Predictions not found for {self.best_model_name} in evaluator.last_predictions.")
            return

        logging.info(f"Predictions found for {self.best_model_name}. Proceeding with plotting confusion matrix.")
        self.ax.clear()
        self.evaluator.plot_confusion_matrix(self.best_model_name, self.ax)
        self.figure.tight_layout()
        self.canvas.draw()
        self.notebook.select(self.plots_tab)


    def _plot_model_comparison(self, metric):
        logging.info(f"Attempting to plot model comparison for metric: {metric}.")
        results_df = self.evaluator.get_results()
        if results_df.empty:
            messagebox.warning("Warning", "No data to plot. Evaluate models first.")
            logging.warning("Results DataFrame is empty for model comparison plot.")
            return

        self.ax.clear()
        self.evaluator.plot_metric_comparison(metric, self.ax)
        self.figure.tight_layout()
        self.canvas.draw()
        self.notebook.select(self.plots_tab)

    def _plot_best_model_roc_curve(self):
        logging.info(f"Attempting to plot ROC curve. Best model: {self.best_model_name}")
        if self.best_model_name is None:
            messagebox.showerror("Error", "No best model to display ROC curve. Train and evaluate models first.")
            return
        
        if self.best_model_name not in self.evaluator.last_predictions:
            messagebox.showerror("Error", "No saved test predictions (probabilities) for the current best model. Please re-evaluate models.")
            logging.error(f"Predictions (probabilities) not found for {self.best_model_name} in evaluator.last_predictions.")
            return

        logging.info(f"Predictions (probabilities) found for {self.best_model_name}. Proceeding with plotting ROC curve.")
        self.ax.clear()
        self.evaluator.plot_roc_curve(self.best_model_name, self.ax)
        self.figure.tight_layout()
        self.canvas.draw()
        self.notebook.select(self.plots_tab)

    def _plot_best_model_pr_curve(self):
        logging.info(f"Attempting to plot PR curve. Best model: {self.best_model_name}")
        if self.best_model_name is None:
            messagebox.showerror("Error", "No best model to display PR curve. Train and evaluate models first.")
            return
        
        if self.best_model_name not in self.evaluator.last_predictions:
            messagebox.showerror("Error", "No saved test predictions (probabilities) for the current best model. Please re-evaluate models.")
            logging.error(f"Predictions (probabilities) not found for {self.best_model_name} in evaluator.last_predictions.")
            return

        logging.info(f"Predictions (probabilities) found for {self.best_model_name}. Proceeding with plotting PR curve.")
        self.ax.clear()
        self.evaluator.plot_pr_curve(self.best_model_name, self.ax)
        self.figure.tight_layout()
        self.canvas.draw()
        self.notebook.select(self.plots_tab)

    def _plot_best_model_feature_importance(self):
        logging.info(f"Attempting to plot feature importance. Best model: {self.best_model_name}")
        if self.best_model_name is None:
            messagebox.showerror("Error", "No best model to display feature importance. Train and evaluate models first.")
            return
        
        model = self.current_trained_models.get(self.best_model_name)
        if model is None:
            messagebox.showerror("Error", f"Best model '{self.best_model_name}' not found among trained models. Please retrain it.")
            logging.error(f"Best model '{self.best_model_name}' not found in current_trained_models.")
            return
        
        if self.feature_names is None or len(self.feature_names) == 0:
            messagebox.showerror("Error", "Feature names are not loaded. Please load and preprocess data.")
            logging.error("Feature names are None or empty.")
            return

        logging.info(f"Feature names and model found. Proceeding with plotting feature importance for {self.best_model_name}.")
        self.ax.clear()
        self.evaluator.plot_feature_importance(self.best_model_name, model, self.ax)
        self.figure.tight_layout()
        self.canvas.draw()
        self.notebook.select(self.plots_tab)


    def _save_best_model(self):
        if self.best_model_name is None or self.best_model_name not in self.current_trained_models:
            messagebox.showerror("Error", "No best trained model to save. Train and evaluate models.")
            return
        
        try:
            model = self.current_trained_models[self.best_model_name]
            self.trainer.save_model(model, self.best_model_name)
            if self.preprocessor:
                self.trainer.save_model(self.preprocessor, 'preprocessor')
            messagebox.showinfo("Save Complete", f"Best model '{self.best_model_name}' and preprocessor successfully saved.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save the best model or preprocessor: {e}")
            logging.error(f"Error saving best model: {e}", exc_info=True)

    def _load_model_for_prediction(self):
        model_name_to_load = self.selected_model.get()
        if not model_name_to_load:
            messagebox.showerror("Error", "Please select a model to load.")
            return

        model = self.trainer.load_model(model_name_to_load)
        preprocessor_loaded = self.trainer.load_model('preprocessor')

        if model and preprocessor_loaded:
            self.loaded_prediction_model = model
            self.preprocessor = preprocessor_loaded
            
            if self.original_feature_names_for_gui is not None and len(self.original_feature_names_for_gui) > 0:
                 self._update_prediction_input_fields(self.original_feature_names_for_gui)
                 messagebox.showinfo("Model Loaded", f"Model '{model_name_to_load}' and preprocessor successfully loaded for prediction.")
            else:
                 messagebox.showwarning("Warning", "Original feature names not available. "
                                        "Please load and preprocess data first, then load the model for prediction.")
        else:
            messagebox.showerror("Error", f"Failed to load model '{model_name_to_load}' or preprocessor. Ensure they are saved and correct names are used.")

    def _load_model_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Joblib files", "*.joblib")])
        if not file_path:
            return

        model_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            model = self.trainer.load_model_from_path(file_path)
            preprocessor_loaded = self.trainer.load_model('preprocessor')

            if model and preprocessor_loaded:
                self.loaded_prediction_model = model
                self.preprocessor = preprocessor_loaded
                
                self.selected_model.set(model_name)

                if self.original_feature_names_for_gui is not None and len(self.original_feature_names_for_gui) > 0:
                    self._update_prediction_input_fields(self.original_feature_names_for_gui)
                    messagebox.showinfo("Model Loaded", f"Model '{model_name}' and preprocessor successfully loaded for prediction.")
                else:
                    messagebox.showwarning("Warning", "Original feature names not available. Please load and preprocess data first, then load the model for prediction.")
            else:
                messagebox.showerror("Loading Error", f"Failed to load model '{model_name}' or preprocessor. Ensure they are saved and correct.")
        except Exception as e:
            messagebox.showerror("Loading Error", f"An error occurred while loading the model: {e}")
            logging.error(f"Error loading model via file dialog: {e}", exc_info=True)


    def _create_prediction_interface(self, parent_tab):
        self.prediction_inputs = {}
        self.input_frames = []

        self.prediction_info_label = ttk.Label(parent_tab, text="Load data and a model to input parameters for prediction.")
        self.prediction_info_label.pack(pady=10)

        self.input_fields_container = ttk.Frame(parent_tab)
        self.input_fields_container.pack(fill="x", padx=10, pady=5)

        self.predict_button = ttk.Button(parent_tab, text="Predict Defect Probability", command=self._make_prediction, state=tk.DISABLED)
        self.predict_button.pack(pady=5)

        self.prediction_result_label = ttk.Label(parent_tab, text="Prediction Result: ", font=("Arial", 12, "bold"))
        self.prediction_result_label.pack(pady=10)

    def _update_prediction_input_fields(self, feature_names_to_display):
        for frame in self.input_frames:
            frame.destroy()
        self.input_frames.clear()
        self.prediction_inputs.clear()

        self.prediction_info_label.config(text="Enter feature values for prediction:")

        for i, feature in enumerate(feature_names_to_display):
            frame = ttk.Frame(self.input_fields_container)
            frame.pack(fill="x", padx=5, pady=2)
            self.input_frames.append(frame)

            ttk.Label(frame, text=f"{feature}:").pack(side="left", padx=5)
            entry = ttk.Entry(frame, width=30)
            entry.pack(side="right", expand=True, fill="x", padx=5)
            self.prediction_inputs[feature] = entry
        
        self.predict_button.config(state=tk.NORMAL)


    def _make_prediction(self):
        logging.info("Predict button clicked. Entering _make_prediction method.")
        try:
            if not hasattr(self, 'loaded_prediction_model') or self.loaded_prediction_model is None:
                messagebox.showerror("Error", "Prediction model not loaded. Please load a model.")
                logging.error("Prediction model not loaded.")
                return
            if self.preprocessor is None:
                messagebox.showerror("Error", "Preprocessor not trained or loaded. Load and preprocess data or load a preprocessor first.")
                logging.error("Preprocessor not loaded.")
                return
            if self.original_feature_names_for_gui is None or len(self.original_feature_names_for_gui) == 0:
                messagebox.showerror("Error", "Feature names not available. Load and preprocess data first.")
                logging.error("Original feature names for GUI are not available.")
                return


            input_values = {}
            try:
                for feature in self.original_feature_names_for_gui:
                    entry = self.prediction_inputs.get(feature)
                    if entry is None:
                        raise ValueError(f"Input field for feature '{feature}' not found. This indicates a mismatch between expected and available input fields.")
                    input_values[feature] = float(entry.get())
            except ValueError as ve:
                messagebox.showerror("Input Error", f"Please enter numerical values for all features. Error: {ve}")
                logging.error(f"Input conversion error: {ve}", exc_info=True)
                return
            except Exception as e:
                messagebox.showerror("Input Error", f"An unexpected error occurred while reading input data: {e}")
                logging.error(f"Unexpected error reading input data: {e}", exc_info=True)
                return


            input_df = pd.DataFrame([input_values], columns=self.original_feature_names_for_gui)

            try:
                transformed_input = self.preprocessor.transform(input_df)

                prediction_proba = self.loaded_prediction_model.predict_proba(transformed_input)[:, 1][0] if hasattr(self.loaded_prediction_model, "predict_proba") else "N/A"
                prediction_class = self.loaded_prediction_model.predict(transformed_input)[0]

                result_text = f"Predicted Defect Probability: {prediction_proba:.4f}\n" \
                              f"Predicted Class (0=No Defect, 1=Defect): {int(prediction_class)}"
                self.prediction_result_label.config(text=result_text)
                self.root.update_idletasks()
                logging.info("Prediction successful and result label updated.")
            except Exception as e:
                messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}. Check input data or model compatibility.")
                logging.error(f"Error during prediction: {e}", exc_info=True)
        except Exception as e:
            messagebox.showerror("Unhandled Prediction Error", f"An unexpected error occurred in prediction: {e}")
            logging.critical(f"Unhandled error in _make_prediction: {e}", exc_info=True)
        logging.info("_make_prediction method finished.")