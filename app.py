import tkinter as tk
from tkinter import ttk, messagebox
from simulated import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

if __name__ == '__main__':
    def calculate_cohorts(modified=False):

        female_prop = float(female_entry.get())
        male_prop = float(male_entry.get())
        white_prop = float(white_entry.get())
        black_prop = float(black_entry.get())
        asian_prop = float(asian_entry.get())
        hispanic_prop = float(hispanic_entry.get())
        other_prop = float(other_entry.get())
        total_students = int(total_students_entry.get())
        students_to_admit = int(admit_students_entry.get())

        gender_enrollment_rates = {
            'Male':male_prop,
            'Female':female_prop
        }

        race_enrollment_rates = {
            'White':white_prop,
            'Black':black_prop,
            'Asian':asian_prop,
            'Hispanic':hispanic_prop,
            'Other':other_prop
        }

        df = create_data(total_students)
        df = enrollment_without_parity(df, gender_enrollment_rates, race_enrollment_rates, students_to_admit)
        if not modified:
            plot_parity(df, plot_frame1, plot_frame2)
        else:
            df2 = create_data2(total_students)
            df2 = enrollment_with_parity(df, df2, gender_enrollment_rates, race_enrollment_rates, students_to_admit)
            plot_parity(df2, plot_frame1, plot_frame2)
        
    
    # GUI setup
    root = tk.Tk()
    root.title("Student Cohort Calculator")

    # Input frame
    input_frame = ttk.Frame(root, padding="10")
    input_frame.grid(row=0, column=0, sticky="W")

    # Gender inputs
    ttk.Label(input_frame, text="Gender Enrollment Rates:").grid(row=0, column=0, sticky="W")
    ttk.Label(input_frame, text="Female:").grid(row=1, column=0, sticky="W")
    female_entry = ttk.Entry(input_frame, width=10)
    female_entry.grid(row=1, column=1)

    ttk.Label(input_frame, text="Male:").grid(row=2, column=0, sticky="W")
    male_entry = ttk.Entry(input_frame, width=10)
    male_entry.grid(row=2, column=1)

    # Race inputs
    ttk.Label(input_frame, text="Race Enrollment Rates:").grid(row=3, column=0, sticky="W")
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

    ttk.Label(input_frame, text="Other:").grid(row=8, column=0, sticky="W")
    other_entry = ttk.Entry(input_frame, width=10)
    other_entry.grid(row=8, column=1)

    # Total students and students to admit
    ttk.Label(input_frame, text="Total Students in Simulated Data:").grid(row=9, column=0, sticky="W")
    total_students_entry = ttk.Entry(input_frame, width=10)
    total_students_entry.grid(row=9, column=1)

    ttk.Label(input_frame, text="Number of Students to Admit:").grid(row=10, column=0, sticky="W")
    admit_students_entry = ttk.Entry(input_frame, width=10)
    admit_students_entry.grid(row=10, column=1)

    # Buttons for calculations
    calculate_unmodified_button = ttk.Button(
        input_frame, text="Calculate (Unmodified)", command=lambda: calculate_cohorts(modified=False)
    )
    calculate_unmodified_button.grid(row=11, column=0, pady=10)

    calculate_modified_button = ttk.Button(
        input_frame, text="Calculate (Modified)", command=lambda: calculate_cohorts(modified=True)
    )
    calculate_modified_button.grid(row=12, column=1, pady=10)

    # Plot frame
    plot_frame1 = ttk.Frame(root, padding="10")
    plot_frame1.grid(row=1, column=0, sticky="W")

    plot_frame2 = ttk.Frame(root, padding="10")
    plot_frame2.grid(row=2, column=0, sticky="W")

    # output_label = ttk.Label(output_frame, text="Results will appear here.", wraplength=400, justify="left")
    # output_label.grid(row=0, column=0, sticky="W")

    # Run the app
    root.mainloop()








