import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


class CarDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Detection Project with OpenCV")

        # Load and resize the image
        image = Image.open("image.jpg")
        image = image.resize((320, 230), Image.LANCZOS)  # Resize with Lanczos filter for antialiasing
        self.photo = ImageTk.PhotoImage(image)

        # Display the image on a Label
        self.label_image = tk.Label(self.root, image=self.photo)
        self.label_image.image = self.photo  # Keep a reference to the image
        self.label_image.pack(pady=20)

        self.label_intro = tk.Label(self.root, text="Welcome to the Car Detection Project with OpenCV!",
                                    font=("Bookman Old Style", 30,"bold"))
        self.label_intro.pack()

        self.start_button = tk.Button(self.root, text="Start Project", command=self.start_project,
                                      font=("Bookman Old Style", 25, "bold",), bg="white", activebackground="lightpink")
        self.start_button.pack(pady=20)

        self.label_team = tk.Label(self.root, text="Team Members:", font=("Bookman Old Style", 16,"bold"))
        self.label_team.pack(pady=6)

        # Create a ttk.Treeview widget
        self.tree = ttk.Treeview(self.root, columns=('Name', 'USN'), show='headings', height=6)
        self.tree.pack(padx=20, pady=10)

        # Define custom fonts for the entire Treeview
        custom_font = ("Bookman Old Style", 12, "bold")

        # Create a ttk.Style instance
        style = ttk.Style()

        # Configure style elements
        style.configure("Treeview.Heading", font=custom_font)  # Configure heading font
        style.configure("Treeview", font=custom_font)  # Configure treeview item font

        # Displaying multiple NAME and USN pairs in the Treeview
        names_usns = [
            ("SUMANTH A", "4SU22CS408"),
            ("MARISWAMI", "4SU22CS402"),
            ("KARIGOWDA", "4SU22CS401"),
            ("UDAYA ", "4SU22CS411")
        ]

        for name, usn in names_usns:
            self.tree.insert('', 'end', values=(name, usn))

        # Make 'Name' and 'USN' headings bold and set padx and pady
        self.tree.heading('Name', text='NAME', anchor=tk.CENTER)
        self.tree.heading('USN', text='USN', anchor=tk.CENTER)

        # Set column widths and stretch properties
        self.tree.column('Name', anchor=tk.CENTER, width=150, stretch=True)
        self.tree.column('USN', anchor=tk.CENTER, width=150, stretch=True)

    def start_project(self):
        self.cap = cv2.VideoCapture('Video.mp4')
        self.countline = 550
        self.minh = 40
        self.minw = 40

        self.algo = cv2.createBackgroundSubtractorKNN()

        self.detect = []
        self.offset = 6
        self.counter = 0

        self.process_video()

    def process_video(self):
        while True:
            ret, frame1 = self.cap.read()
            if not ret:
                break

            grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grey, (3, 3), 5)
            img_sub = self.algo.apply(blur)
            d = cv2.dilate(img_sub, np.ones((5, 5)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilatada = cv2.morphologyEx(d, cv2.MORPH_CLOSE, kernel)
            dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.line(frame1, (0, self.countline), (1300, self.countline), (0, 0, 250), 3)

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w >= self.minw and h >= self.minh:
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    center = self.center_handle(x, y, w, h)
                    self.detect.append(center)
                    cv2.circle(frame1, center, 4, (0, 0, 255), -1)
                    self.update_counter(center)

            cv2.putText(frame1, "Car Counter: " + str(self.counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (120, 255, 255), 2)
            cv2.imshow("Detector", frame1)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        self.cap.release()

    def center_handle(self, x, y, w, h):
        cx = x + int(w / 2)
        cy = y + int(h / 2)
        return cx, cy

    def update_counter(self, center):
        for (x, y) in self.detect:
            if y < (self.countline + self.offset) and y > (self.countline - self.offset):
                self.counter += 1
                self.detect.remove((x, y))
                print("Car counter: " + str(self.counter))


if __name__ == "__main__":
    root = tk.Tk()
    app = CarDetectionApp(root)
    root.mainloop()
