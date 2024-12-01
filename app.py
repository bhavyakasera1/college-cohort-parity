import tkinter as tk
from tkinter import ttk, messagebox

if __name__ == '__main__':
    def calculate_cohorts():
        female_prop = float(female_entry.get())
        male_prop = float(male_entry.get())
        white_prop = float(white_entry.get())
        black_prop = float(black_entry.get())
        asian_prop = float(asian_entry.get())
        hispanic_prop = float(hispanic_entry.get())
        total_students = int(total_students_entry.get())
        students_to_admit = int(admit_students_entry.get())
    
    # GUI setup
    root = tk.Tk()
    root.title("Student Cohort Calculator")

    # Input frame
    input_frame = ttk.Frame(root, padding="10")
    input_frame.grid(row=0, column=0, sticky="W")

    # Gender inputs
    ttk.Label(input_frame, text="Gender Proportions:").grid(row=0, column=0, sticky="W")
    ttk.Label(input_frame, text="Female:").grid(row=1, column=0, sticky="W")
    female_entry = ttk.Entry(input_frame, width=10)
    female_entry.grid(row=1, column=1)

    ttk.Label(input_frame, text="Male:").grid(row=2, column=0, sticky="W")
    male_entry = ttk.Entry(input_frame, width=10)
    male_entry.grid(row=2, column=1)

    # Race inputs
    ttk.Label(input_frame, text="Race Proportions:").grid(row=3, column=0, sticky="W")
    ttk.Label(input_frame, text="White:").grid(row=4, column=0, sticky="W")
    white_entry = ttk.Entry(input_frame, width=10)
    white_entry.grid(row=4, column=1)

    ttk.Label(input_frame, text="Black:").grid(row=5, column=0, sticky="W")
    black_entry = ttk.Entry(input_frame, width=10)
    black_entry.grid(row=5, column=1)

    ttk.Label(input_frame, text="Asian:").grid(row=6, column=0, sticky="W")
    asian_entry = ttk.Entry(input_frame, width=10)
    asian_entry.grid(row=6, column=1)

    ttk.Label(input_frame, text="Hispanic:").grid(row=7, column=0, sticky="W")
    hispanic_entry = ttk.Entry(input_frame, width=10)
    hispanic_entry.grid(row=7, column=1)

    # Total students and students to admit
    ttk.Label(input_frame, text="Total Students in Simulated Data:").grid(row=8, column=0, sticky="W")
    total_students_entry = ttk.Entry(input_frame, width=10)
    total_students_entry.grid(row=8, column=1)

    ttk.Label(input_frame, text="Number of Students to Admit:").grid(row=9, column=0, sticky="W")
    admit_students_entry = ttk.Entry(input_frame, width=10)
    admit_students_entry.grid(row=9, column=1)

    # Calculate button
    calculate_button = ttk.Button(input_frame, text="Calculate Cohorts", command=calculate_cohorts)
    calculate_button.grid(row=10, column=0, columnspan=2, pady=10)

    # Output frame
    output_frame = ttk.Frame(root, padding="10")
    output_frame.grid(row=1, column=0, sticky="W")

    output_label = ttk.Label(output_frame, text="Results will appear here.", wraplength=400, justify="left")
    output_label.grid(row=0, column=0, sticky="W")

    # Run the app
    root.mainloop()







