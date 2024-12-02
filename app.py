import tkinter as tk
from tkinter import ttk, messagebox
from simulated import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

if __name__ == '__main__':
    def calculate_cohorts(modified=False):
        female_prop = female_slider.get()
        male_prop = male_slider.get()
        white_prop = white_slider.get()
        black_prop = black_slider.get()
        asian_prop = asian_slider.get()
        hispanic_prop = hispanic_prop_slider.get()
        other_prop = other_slider.get()
        total_students = int(total_students_slider.get())
        students_to_admit = int(admit_students_slider.get())

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

    def update_value_label(label, value):
        label.config(text=f"{value:.2f}")

    root = tk.Tk()
    root.title("Student Cohort Calculator")
    root.configure(bg='#f0f0f0')

    style = ttk.Style()
    style.configure('TFrame', background='#f0f0f0')
    style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
    style.configure('TButton', font=('Arial', 10, 'bold'))
    style.configure('Header.TLabel', font=('Arial', 12, 'bold'))

    main_container = ttk.Frame(root)
    main_container.pack(fill="both", expand=True)

    left_panel = ttk.Frame(main_container)
    left_panel.pack(side="left", fill="both", expand=False, padx=10)

    right_panel = ttk.Frame(main_container)
    right_panel.pack(side="right", fill="both", expand=True, padx=10)

    title_label = ttk.Label(left_panel, text="Student Cohort Calculator", style='Header.TLabel')
    title_label.pack(pady=(20, 20), padx=10, anchor="w")

    # Gender sliders section
    gender_frame = ttk.LabelFrame(left_panel, text="Gender Enrollment Rates", padding=(10, 5))
    gender_frame.pack(fill="x", padx=10, pady=5)

    female_frame = ttk.Frame(gender_frame)
    female_frame.pack(fill="x")
    ttk.Label(female_frame, text="Female:").pack(side="left")
    female_value = ttk.Label(female_frame, text="0.50")
    female_value.pack(side="right")
    female_slider = ttk.Scale(gender_frame, from_=0, to=1, orient="horizontal", 
                            command=lambda v: update_value_label(female_value, float(v)))
    female_slider.set(0.5)
    female_slider.pack(fill="x", padx=5, pady=2)

    male_frame = ttk.Frame(gender_frame)
    male_frame.pack(fill="x")
    ttk.Label(male_frame, text="Male:").pack(side="left")
    male_value = ttk.Label(male_frame, text="0.50")
    male_value.pack(side="right")
    male_slider = ttk.Scale(gender_frame, from_=0, to=1, orient="horizontal",
                          command=lambda v: update_value_label(male_value, float(v)))
    male_slider.set(0.5)
    male_slider.pack(fill="x", padx=5, pady=2)

    # Race sliders section
    race_frame = ttk.LabelFrame(left_panel, text="Race Enrollment Rates", padding=(10, 5))
    race_frame.pack(fill="x", padx=10, pady=5)

    race_sliders = []
    for label, var_name in [
        ("White:", "white_slider"),
        ("Black:", "black_slider"),
        ("Asian:", "asian_slider"),
        ("Hispanic:", "hispanic_prop_slider"),
        ("Other:", "other_slider")
    ]:
        frame = ttk.Frame(race_frame)
        frame.pack(fill="x")
        ttk.Label(frame, text=label).pack(side="left")
        value_label = ttk.Label(frame, text="0.20")
        value_label.pack(side="right")
        
        slider = ttk.Scale(race_frame, from_=0, to=1, orient="horizontal",
                          command=lambda v, l=value_label: update_value_label(l, float(v)))
        slider.set(0.2)
        slider.pack(fill="x", padx=5, pady=2)
        globals()[var_name] = slider
        race_sliders.append(slider)

    # Simulation parameters section
    params_frame = ttk.LabelFrame(left_panel, text="Simulation Parameters", padding=(10, 5))
    params_frame.pack(fill="x", padx=10, pady=5)

    total_frame = ttk.Frame(params_frame)
    total_frame.pack(fill="x")
    ttk.Label(total_frame, text="Total Students:").pack(side="left")
    total_value = ttk.Label(total_frame, text="1000")
    total_value.pack(side="right")
    total_students_slider = ttk.Scale(params_frame, from_=100, to=10000, orient="horizontal",
                                    command=lambda v: total_value.config(text=str(int(float(v)))))
    total_students_slider.set(1000)
    total_students_slider.pack(fill="x", padx=5, pady=2)

    admit_frame = ttk.Frame(params_frame)
    admit_frame.pack(fill="x")
    ttk.Label(admit_frame, text="Students to Admit:").pack(side="left")
    admit_value = ttk.Label(admit_frame, text="100")
    admit_value.pack(side="right")
    admit_students_slider = ttk.Scale(params_frame, from_=10, to=1000, orient="horizontal",
                                    command=lambda v: admit_value.config(text=str(int(float(v)))))
    admit_students_slider.set(100)
    admit_students_slider.pack(fill="x", padx=5, pady=2)

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

    plot_frame1 = ttk.Frame(right_panel)
    plot_frame1.pack(fill="both", expand=True, pady=10)

    plot_frame2 = ttk.Frame(right_panel)
    plot_frame2.pack(fill="both", expand=True, pady=10)

    root.minsize(1000, 600)
    root.mainloop()