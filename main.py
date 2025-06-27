import tkinter as tk
from src.gui import DefectPredictionApp
import logging
import os
from tkinterdnd2 import TkinterDnD

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('models'):
        os.makedirs('models')
    
    logging.info("Application started.")
    root = TkinterDnD.Tk()
    app = DefectPredictionApp(root)
    root.mainloop()
    logging.info("Application closed.")

if __name__ == "__main__":
    main()