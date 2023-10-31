import os
import take_imgs
import norm_img
import inference
from tkinter import *
from tkinter import ttk
import tkinter as tk
import threading
# import cv2
# from PIL import ImageTk
# import PIL
import csv
import tkinter.messagebox
import time

def view_attendance():
    root.withdraw()  # Hide the main window
    view_attendance_window = tk.Toplevel(root)
    view_attendance_window.title("View Attendance")

    # Set the dimensions for the registration window
    main_window_width = 382
    main_window_height = 450
    view_attendance_window.geometry(f"{main_window_width}x{main_window_height}")

    def download_attendance():
        selected_program = program_combobox.get()
        selected_date = date_entry.get()

        if not (selected_program and selected_date):
            return  # Don't proceed if any field is empty

        # Format the date as 'dd_mm_yyyy' for the file name
        formatted_date = selected_date.replace('/', '_')

        # Generate the CSV file name
        csv_file_name = f'attendanceList-{formatted_date}.csv'

        base_folder = 'Attendance_records'

        # Check if the file exists
        program_folder = os.path.join(base_folder, selected_program)
        csv_file_path = os.path.join(program_folder, csv_file_name)

        if os.path.isfile(csv_file_path):
            # Offer the file for download
            with open(csv_file_path, 'rb') as file:
                file_data = file.read()

            # Create a download dialog
            download_dialog = tk.Toplevel(root)
            download_dialog.title("Download Attendance")

            def close_download_dialog():
                download_dialog.destroy()

            download_label = tk.Label(download_dialog, text="Click the button to download the attendance file:")
            download_label.pack(pady=(0, 20))

            download_button = tk.Button(
                download_dialog,
                text="Download",
                command=lambda: close_download_dialog(),
                width=20,
                height=2,
                bg='#3498db',
                fg='#ffffff',
                bd=2,
                relief=tk.FLAT,
                activebackground="Green",
                activeforeground="White",
            )
            download_button.pack()

            # Create a temporary file to store the data for download
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            saved_file_path = f'attendanceList-{selected_program}-{formatted_date}-{timestamp}.csv'
            with open(saved_file_path, 'wb') as saved_file:
                saved_file.write(file_data)

            download_dialog.protocol("WM_DELETE_WINDOW", close_download_dialog)
            download_dialog.mainloop()
        else:
            # Show a message if the file doesn't exist
            tk.messagebox.showinfo("File Not Found", "Attendance file not found for the selected program and date.")

    # Create labels and dropdowns for program and date selection
    program_label = tk.Label(view_attendance_window, text="Select Program:")
    program_label.pack()
    program_combobox = ttk.Combobox(view_attendance_window, values=["RDS", "REI", "RWS", "RSD"])
    program_combobox.pack(pady=(0, 20))

    date_label = tk.Label(view_attendance_window, text="Select Date (dd/mm/yyyy):")
    date_label.pack()
    date_entry = tk.Entry(view_attendance_window)
    date_entry.pack(pady=(0, 20))

    view_button = tk.Button(
        view_attendance_window,
        text="View Attendance",
        command=download_attendance,
        width=42,
        bg='#3498db',
        fg='#ffffff',
        bd=2,
        relief=tk.FLAT,
        activebackground="Green",
        activeforeground="White",
    )
    view_button.pack(pady=(0, 20))  # Increase the bottom padding to widen the gap

    def back_to_main_screen():
        view_attendance_window.destroy()  # Close the registration window
        root.deiconify()  # Show the main window again

    # Back button to go back to the main screen
    back_button = tk.Button(
        view_attendance_window,
        text="Back",
        command=back_to_main_screen,
        width=42,
        height=2,
        bg='red',
        fg='#ffffff',
        bd=2,
        relief=tk.FLAT,
        activebackground="Green",
        activeforeground="White",
    )
    back_button.pack(pady=8)


def cfaces_call(name, student_id, programme, tutorial_group, year_and_sem):
    tkStatus.set("Capturing Faces...")
    status_label.update()
    take_imgs.takeImages(name, student_id, programme, tutorial_group, year_and_sem)
    tkStatus.set("Faces Captured")
    status_label.update()


def save_student_details(name, student_id, programme, tutorial_group, year_and_sem):
    with open('student_database.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, student_id, programme, tutorial_group, year_and_sem])

    tkStatus.set("Student details saved.")
    status_label.update()

def take_imgs1(name_entry, student_id_entry, programme_entry, tutorial_group_entry, year_sem_entry, error_label):
    name = name_entry.get()
    student_id = student_id_entry.get()
    programme = programme_entry.get()
    tutorial_group = tutorial_group_entry.get()
    year_and_sem = year_sem_entry.get()

    if not (name and student_id and programme and tutorial_group and year_and_sem):
        error_label.config(text="Error: All fields are required!")
        return  # Don't proceed if any field is empty

    error_label.config(text="")  # Clear the error message
    tkStatus.set("")  # Clear any previous status message
    status_label.update()

    # Introduce a slight delay to show the "Updating Faces..." message
    tkStatus.set("Updating Faces...")
    status_label.update()
    root.update_idletasks()  # Force the update of the GUI to show the message

    # Call the function to save student details
    save_student_details(name, student_id, programme, tutorial_group, year_and_sem)

    t1 = threading.Thread(target=cfaces_call, args=(name, student_id, programme, tutorial_group, year_and_sem), daemon=True)
    t1.start()
def normalize_img():
    tkStatus.set("Normalizing Faces...")
    status_label.update()
    norm_img.normal_img()
    tkStatus.set("Image Normalized")
    status_label.update()


def norm_img1():
    t2 = threading.Thread(target=normalize_img, daemon=True)
    t2.start()

def rfaces_call():
    tkStatus.set("Recognizing Faces...")
    status_label.update()
    inference.recognize_attendance()
    tkStatus.set("Faces Recognized.")
    status_label.update()


def inference1():
    t4 = threading.Thread(target=rfaces_call, daemon=True)
    t4.start()

root = Tk()
root.title("Contactless Attendance System")
tkStatus = tk.StringVar()


def open_registration_screen():
    root.withdraw()  # Hide the main window
    registration_window = tk.Toplevel(root)
    registration_window.title("Registration")

    # Set the dimensions for the registration window
    # main_window_width = root.winfo_width()  # Get the main window's width
    # main_window_height = root.winfo_height()  # Get the main window's height
    main_window_width = 382
    main_window_height = 450
    registration_window.geometry(f"{main_window_width}x{main_window_height}")

    # Add entry widgets for user information
    name_label = tk.Label(registration_window, text="Name:")
    name_label.pack()
    name_entry = tk.Entry(registration_window)
    name_entry.pack()

    student_id_label = tk.Label(registration_window, text="Student ID:")
    student_id_label.pack()
    student_id_entry = tk.Entry(registration_window)
    student_id_entry.pack()

    programme_label = tk.Label(registration_window, text="Programme:")
    programme_label.pack()
    programme_combobox = ttk.Combobox(registration_window, values=["RDS", "REI", "RWS", "RSD"])
    programme_combobox.pack()

    # Create a Combobox for Tutorial Groups
    tutorial_group_label = tk.Label(registration_window, text="Tutorial Group:")
    tutorial_group_label.pack()
    tutorial_group_combobox = ttk.Combobox(registration_window, values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    tutorial_group_combobox.pack()

    # Create a Combobox for Year and Semester
    year_and_sem_label = tk.Label(registration_window, text="Year and Semester:")
    year_and_sem_label.pack()
    year_sem_combobox = ttk.Combobox(registration_window, values=["Year 1 Sem 1", "Year 1 Sem 2", "Year 1 Sem 3", "Year 2 Sem 1", "Year 2 Sem 2", "Year 2 Sem 3", "Year 3 Sem 1", "Year 3 Sem 2", "Year 3 Sem 3"])
    year_sem_combobox.pack()

    error_label = tk.Label(registration_window, text="", fg="red")
    error_label.pack()

    take_image_button = tk.Button(
        registration_window,
        text="TAKE IMAGE",
        command=lambda: take_imgs1(name_entry, student_id_entry, programme_combobox, tutorial_group_combobox, year_sem_combobox, error_label),
        width=42,  # Adjust the width as needed
        height=2,  # Adjust the height as needed
        bg='#3498db',
        fg='#ffffff',
        bd=2,
        relief=tk.FLAT,
        activebackground="Green",
        activeforeground="White",
    )
    take_image_button.pack(pady=8)

    normalize_button = tk.Button(
        registration_window,
        text="NORMALIZE IMAGE",
        command=norm_img1,
        width=42,  # Adjust the width as needed
        height=2,  # Adjust the height as needed
        bg='#3498db',
        fg='#ffffff',
        bd=2,
        relief=tk.FLAT,
        activebackground="Green",
        activeforeground="White",
    )
    normalize_button.pack(pady=8)

    def back_to_main_screen():
        registration_window.destroy()  # Close the registration window
        root.deiconify()  # Show the main window again

    # Back button to go back to the main screen
    back_button = tk.Button(
        registration_window,
        text="Back",
        command=back_to_main_screen,
        width=42,
        height=2,
        bg='red',
        fg='#ffffff',
        bd=2,
        relief=tk.FLAT,
        activebackground="Green",
        activeforeground="White",
    )
    back_button.pack(pady=8)

def check_student_information():
    root.withdraw()  # Hide the main window

    check_window = tk.Toplevel(root)
    check_window.title("Check Student Information")

    main_window_width = 382
    main_window_height = 300
    check_window.geometry(f"{main_window_width}x{main_window_height}")

    # Create an entry widget for student ID
    student_id_label = tk.Label(check_window, text="Enter Student ID:")
    student_id_label.pack()
    student_id_entry = tk.Entry(check_window)
    student_id_entry.pack()

    result_label = tk.Label(check_window, text="", fg="green")
    result_label.pack()

    def get_student_info():
        entered_student_id = student_id_entry.get()

        # Read the CSV file to search for student information
        with open('student_database.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[1] == entered_student_id:  # Assuming student ID is in the second column
                    student_info = f"Student ID: {row[1]}\nName: {row[0]}\nProgramme: {row[2]}\nTutorial Group: {row[3]}\nYear and Semester: {row[4]}"
                    result_label.config(text=student_info)
                    return

        result_label.config(text="Student not found")

    # Create a button to fetch student information
    fetch_info_button = tk.Button(
        check_window,
        text="Check",
        command=get_student_info,
        width=32,
        height=2,
        bg='#3498db',
        fg='#ffffff',
        bd=2,
        relief=tk.FLAT,
        activebackground="Green",
        activeforeground="White",
    )
    fetch_info_button.pack(pady=8)
    def back_to_main_screen():
        check_window.destroy()  # Close the check window
        root.deiconify()  # Show the main window again

    # Back button to go back to the main screen
    back_button = tk.Button(
        check_window,
        text="Back",
        command=back_to_main_screen,
        width=32,
        height=2,
        bg='red',
        fg='#ffffff',
        bd=2,
        relief=tk.FLAT,
        activebackground="Green",
        activeforeground="White",
    )
    back_button.pack(pady=8)


# ---------------main driver ------------------
# create a tkinter window

content_frame = tk.Frame(root)
content_frame.grid(row=1, column=0, padx=15, pady=8)

# Open window having dimension 100x100
# root.geometry('100x100')

# Create a Button
check_info_button = tk.Button(
    root,
    text="CHECK STUDENT INFORMATION",
    command=check_student_information,
    width=42,
    bg='#3498db',
    fg='#ffffff',
    bd=2,
    relief=tk.FLAT,
    activebackground="Green",
    activeforeground="White",
)
check_info_button.grid(
    padx=15,
    pady=8,
    ipadx=24,
    ipady=6,
    row=6,  # Adjust the row as needed
    column=0,
    columnspan=4,
    sticky=tk.W + tk.E + tk.N + tk.S,
)

registration_button = tk.Button(
    root,
    text="REGISTRATION",
    command=open_registration_screen,
    width=42,
    bg='#3498db',
    fg='#ffffff',
    bd=2,
    relief=tk.FLAT,
    activebackground="Green",
    activeforeground="White",
)
registration_button.grid(
    padx=15,
    pady=8,
    ipadx=24,
    ipady=6,
    row=5,  # Change to row=7
    column=0,
    columnspan=4,
    sticky=tk.W + tk.E + tk.N + tk.S,
)

btn4 = tk.Button(
    root,
    text='RECOGNIZE FACES',
    command=inference1,
    width=42,
    bg='#3498db',
    fg='#ffffff',
    bd=2,
    relief=tk.FLAT,
    activebackground="Green",
    activeforeground="White",
)

btn4.grid(
    padx=15,
    pady=8,
    ipadx=24,
    ipady=6,
    row=7,  # Change to row=7
    column=0,
    columnspan=4,
    sticky=tk.W + tk.E + tk.N + tk.S,
)

# Create a Button for viewing attendance (renamed to "View Attendance")
view_attendance_button = tk.Button(
    root,
    text="VIEW ATTENDANCE",
    command=view_attendance,  # Replace with the actual function to view attendance
    width=42,
    bg='#3498db',
    fg='#ffffff',
    bd=2,
    relief=tk.FLAT,
    activebackground="Green",
    activeforeground="White",
)
view_attendance_button.grid(
    padx=15,
    pady=8,
    ipadx=24,
    ipady=6,
    row=8,  # Adjust the row as needed
    column=0,
    columnspan=4,
    sticky=tk.W + tk.E + tk.N + tk.S,
)

btn5 = tk.Button(
    root,
    text='EXIT',
    command=root.destroy,
    width=42,
    bg='red',
    fg='#ffffff',
    bd=2,
    relief=tk.FLAT,
    activebackground="Green",
    activeforeground="White",
)
btn5.grid(
    padx=15,
    pady=8,
    ipadx=24,
    ipady=6,
    row=9,  # Change to row=8
    column=0,
    columnspan=4,
    sticky=tk.W + tk.E + tk.N + tk.S,
)

status_label = tk.Label(
    root,
    textvariable=tkStatus,
    bg='#eeeeee',
    anchor=tk.W,
    justify=tk.LEFT,
    relief=tk.FLAT,
    wraplength=350,
)
status_label.grid(
    padx=12,
    pady=(0, 12),
    ipadx=0,
    ipady=1,
    row=10,
    column=0,
    columnspan=4,
    sticky=tk.W + tk.E + tk.N + tk.S,
)


# Set the position of button on the top of window.

root.mainloop()
# mainMenu()
