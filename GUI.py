import tkinter as tk
from tkinter import filedialog as fd
from PIL import ImageTk, Image
root = tk.Tk()
image = Image.open(r'2.png')
image = image.resize((450, 350), Image.Resampling.LANCZOS)
my_img = ImageTk.PhotoImage(image)
my_img = tk.Label(image = my_img)
my_img.pack()
root.mainloop()
