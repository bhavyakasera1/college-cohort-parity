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
            'Male': male_prop,
            'Female': female_prop
        }

        race_enrollment_rates = {
            'White': white_prop,
            'Black': black_prop,
            'Asian': asian_prop,
            'Hispanic': hispanic_prop,
            'Other': other_prop
        }

        df = create_data(total_students)
        df = enrollment_without_parity(df, gender_enrollment_rates, race_enrollment_rates, students_to_admit)
        if not modified:
            plot_parity(df, plot_frame1, plot_frame2)
        else:
            df2 = create_data2(total_students)
            df2 = enrollment_with_parity(df, df2, gender_enrollment_rates, race_enrollment_rates, students_to_admit)
            plot_parity(df2, plot_frame1, plot_frame2)

    # GUI setup with custom styling
    root = tk.Tk()
    root.title("Student Cohort Calculator")
    root.configure(bg='#f0f0f0')

    # Custom style configuration
    style = ttk.Style()
    style.configure('TFrame', background='#f0f0f0')
    style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
    style.configure('TButton', font=('Arial', 10, 'bold'))
    style.configure('Header.TLabel', font=('Arial', 12, 'bold'))

    # Create main container that will hold both input and plot areas
    main_container = ttk.Frame(root)
    main_container.pack(fill="both", expand=True)

    # Create left panel for inputs
    left_panel = ttk.Frame(main_container)
    left_panel.pack(side="left", fill="both", expand=False, padx=10)

    # Create right panel for plots
    right_panel = ttk.Frame(main_container)
    right_panel.pack(side="right", fill="both", expand=True, padx=10)

    # Input section (left panel)
    # Title
    title_label = ttk.Label(left_panel, text="Student Cohort Calculator", style='Header.TLabel')
    title_label.pack(pady=(20, 20), padx=10, anchor="w")

    # Gender inputs section
    gender_frame = ttk.LabelFrame(left_panel, text="Gender Enrollment Rates", padding=(10, 5))
    gender_frame.pack(fill="x", padx=10, pady=5)

    for label_text, entry_var in [("Female:", "female_entry"), ("Male:", "male_entry")]:
        frame = ttk.Frame(gender_frame)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label_text).pack(side="left")
        globals()[entry_var] = ttk.Entry(frame, width=15)
        globals()[entry_var].pack(side="right", padx=5)

    # Race inputs section
    race_frame = ttk.LabelFrame(left_panel, text="Race Enrollment Rates", padding=(10, 5))
    race_frame.pack(fill="x", padx=10, pady=5)

    race_entries = [
        ("White:", "white_entry"),
        ("Black:", "black_entry"),
        ("Asian:", "asian_entry"),
        ("Hispanic:", "hispanic_entry"),
        ("Other:", "other_entry")
    ]

    for label_text, entry_var in race_entries:
        frame = ttk.Frame(race_frame)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label_text).pack(side="left")
        globals()[entry_var] = ttk.Entry(frame, width=15)
        globals()[entry_var].pack(side="right", padx=5)

    # Simulation parameters section
    params_frame = ttk.LabelFrame(left_panel, text="Simulation Parameters", padding=(10, 5))
    params_frame.pack(fill="x", padx=10, pady=5)

    # Total students
    total_frame = ttk.Frame(params_frame)
    total_frame.pack(fill="x", pady=2)
    ttk.Label(total_frame, text="Total Students:").pack(side="left")
    total_students_entry = ttk.Entry(total_frame, width=15)
    total_students_entry.pack(side="right", padx=5)

    # Students to admit
    admit_frame = ttk.Frame(params_frame)
    admit_frame.pack(fill="x", pady=2)
    ttk.Label(admit_frame, text="Students to Admit:").pack(side="left")
    admit_students_entry = ttk.Entry(admit_frame, width=15)
    admit_students_entry.pack(side="right", padx=5)

    # Buttons
    button_frame = ttk.Frame(left_panel)
    button_frame.pack(pady=20, padx=10)

    calculate_unmodified_button = ttk.Button(
        button_frame,
        text="Calculate (Unmodified)",
        command=lambda: calculate_cohorts(modified=False),
        padding=(10, 5)
    )
    calculate_unmodified_button.pack(side="left", padx=5)

    calculate_modified_button = ttk.Button(
        button_frame,
        text="Calculate (Modified)",
        command=lambda: calculate_cohorts(modified=True),
        padding=(10, 5)
    )
    calculate_modified_button.pack(side="left", padx=5)

    # Plot frames (right panel)
    plot_frame1 = ttk.Frame(right_panel)
    plot_frame1.pack(fill="both", expand=True, pady=10)

    plot_frame2 = ttk.Frame(right_panel)
    plot_frame2.pack(fill="both", expand=True, pady=10)

    # Set a minimum window size
    root.minsize(1000, 600)

    # Run the app
    root.mainloop()