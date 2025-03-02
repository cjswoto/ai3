import tkinter as tk

def update_status(update_status, message):
    update_status.config(text=message)
    update_status.update_idletasks()
