import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

RACE_RATIOS = {
        'White': 0.5784,
        'Black': 0.1205,
        'Asian': 0.0592,
        'Hispanic': 0.1873,
        'Other': 1 - 0.5784 - 0.1205 - 0.0592 - 0.1873
    }

def create_data(num_students):
    # Generate the race column based on the defined ratios
    race_population = np.random.choice(
        list(RACE_RATIOS.keys()),
        size=num_students,
        p=list(RACE_RATIOS.values())
    )

    # Create the DataFrame
    data = {
        'student_id': range(1, num_students + 1),
        'gpa': np.random.normal(2.0, 4.0, num_students),
        'essay_score': np.random.normal(1, 5, num_students),
        'gender': np.random.choice(['Male', 'Female'], num_students),
        'race': race_population
    }
    df = pd.DataFrame(data)

    # Normalize GPA and GMAT scores to a range of 0 to 0.5
    df['normalized_gpa'] = (df['gpa'] - 2.0) / (4.0 - 2.0) * 0.5  # Normalized to [0, 0.5]
    df['normalized_essay_score'] = (df['essay_score'] - 1) / (5 - 1) * 0.5  # Normalized to [0, 0.5]

    # Create a ranking score
    df['ranking_score'] = df['normalized_gpa'] + df['normalized_essay_score']

    # Rank students based on the combined score in descending order
    df = df.sort_values(by='ranking_score', ascending=False)
    return df


def admit_students(df, target_admits = 1000):
    # Calculate number of admits needed for each race
    admits_by_race = {race: int(target_admits * ratio) for race, ratio in RACE_RATIOS.items()}

    # Create a list to hold selected students
    selected_students = []

    # Sample students based on race and gender
    for race, num_needed in admits_by_race.items():
        # Select students of this race
        race_group = df[df['race'] == race]

        # Calculate the number of admits needed from each gender within this race
        gender_counts = race_group['gender'].value_counts(normalize=True)

        for gender, proportion in gender_counts.items():
            num_gender_needed = int(num_needed * proportion)
            gender_group = race_group[race_group['gender'] == gender]

            if len(gender_group) > 0:
                # Select top-ranked students based on the ranking score
                sampled = gender_group.head(num_gender_needed)
                selected_students.append(sampled)

    # Combine the selected students into a single DataFrame
    if selected_students:
        admitted_df = pd.concat(selected_students)
    else:
        admitted_df = pd.DataFrame()  # Return empty if no students selected

    # If the total selected is less than target admits, fill the remaining with top-ranked from overall DataFrame
    if len(admitted_df) < target_admits:
        remaining_needed = target_admits - len(admitted_df)
        remaining_students = df.drop(admitted_df.index)  # Exclude already selected
        additional_samples = remaining_students.head(remaining_needed)  # Select top-ranked remaining
        admitted_df = pd.concat([admitted_df, additional_samples])

    return admitted_df

def enrollment_without_parity(df, gender_enrollment_rates, race_enrollment_rates, target_admits):

    admitted_students = admit_students(df, target_admits)
    # Initialize the enrollment column
    admitted_students['enrollment'] = 0.0

    # Assign enrollment probabilities based on gender and race
    for index, row in admitted_students.iterrows():
        gender = row['gender']
        race = row['race']
        gender_prob = gender_enrollment_rates[gender]
        race_prob = race_enrollment_rates[race]

        # Calculate the combined probability assuming independence
        combined_prob = gender_prob * race_prob

        # Assign enrollment status based on combined probability
        admitted_students.at[index, 'enrollment'] = 1 if np.random.rand() < combined_prob else 0
    df.loc[:,"admit"] = 0
    df.loc[:,"enrollment"] = 0

    df.loc[admitted_students.index, "admit"] = 1
    df.loc[admitted_students.index, "enrollment"] = admitted_students["enrollment"]
    return df


def bootstrap_enrollment_rates(df, group_column, num_iterations=100):
    rates = {}
    for group in df[group_column].unique():
        group_rates = []

        for _ in range(num_iterations):
            # Resample admitted students
            resampled_students = df.sample(n=len(df), replace=True)

            # Calculate enrollment rates for resampled data
            enrollment_count = resampled_students[resampled_students[group_column] == group]["enrollment"].sum()
            total_students = df[df[group_column] == group]["student_id"].count()
            enrollment_rate = enrollment_count / total_students if total_students > 0 else 0
            group_rates.append(enrollment_rate)

        # Store the percentiles for confidence intervals
        rates[group] = np.percentile(group_rates, [2.5, 50, 97.5])

    return rates

def plot_parity(df):
    # Calculate confidence intervals for gender enrollment rates
    cis = bootstrap_enrollment_rates(df, "gender")
    male_ci = cis["Male"]
    female_ci =  cis["Female"]


    # Calculate enrollment parity for gender
    enrollment_parity_gender = female_ci[1] / male_ci[1]  - 1 if male_ci[1] > 0 else 0  # Using median rates

    # Prepare data for visualization for gender
    enrollment_ci_gender_df = pd.DataFrame({
        'Gender': ['Female', 'Male'],
        'Enrollment Rate': [female_ci[1], male_ci[1]],  # Median enrollment rates
        'Lower CI': [female_ci[0], male_ci[0]],
        'Upper CI': [female_ci[2], male_ci[2]]
    })

    # Visualization for gender
    plt.figure(figsize=(8, 6))
    sns.barplot(data=enrollment_ci_gender_df, x='Gender', y='Enrollment Rate', color='blue', alpha=0.6, ci=None)
    plt.errorbar(x=enrollment_ci_gender_df['Gender'], y=enrollment_ci_gender_df['Enrollment Rate'],
                yerr=[enrollment_ci_gender_df['Enrollment Rate'] - enrollment_ci_gender_df['Lower CI'],
                        enrollment_ci_gender_df['Upper CI'] - enrollment_ci_gender_df['Enrollment Rate']],
                fmt='none', c='black', capsize=5)
    plt.title('Gender Enrollment Rates with 95% Confidence Intervals')
    plt.xlabel('Gender')
    plt.ylabel('Enrollment Rate')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.show()

    # Output the enrollment parity for gender
    print(f'Enrollment Parity (Male/Female): {enrollment_parity_gender:.2f}')

    # Calculate confidence intervals for racial groups enrollment rates
    racial_rates = bootstrap_enrollment_rates(df, "race")

    # Calculate enrollment parity for racial groups against "White"
    white_ci = racial_rates.get('White', None)
    parity_results = {}

    for race, ci in racial_rates.items():
        if white_ci is not None:
            parity = ci[1] / white_ci[1] - 1 if white_ci[1] > 0 else 0  # Using median rates
            parity_results[race] = {
                'Enrollment Rate': ci[1],
                'Lower CI': ci[0],
                'Upper CI': ci[2],
                'Enrollment Parity vs White': parity
            }

    # Prepare data for visualization for racial groups
    enrollment_ci_race_df = pd.DataFrame(parity_results).T
    enrollment_ci_race_df.reset_index(inplace=True)
    enrollment_ci_race_df.columns = ['Race', 'Enrollment Rate', 'Lower CI', 'Upper CI', 'Enrollment Parity vs White']

    # Visualization for racial groups
    plt.figure(figsize=(10, 6))
    sns.barplot(data=enrollment_ci_race_df, x='Race', y='Enrollment Rate', color='blue', alpha=0.6, ci=None)
    plt.errorbar(x=enrollment_ci_race_df['Race'], y=enrollment_ci_race_df['Enrollment Rate'],
                yerr=[enrollment_ci_race_df['Enrollment Rate'] - enrollment_ci_race_df['Lower CI'],
                        enrollment_ci_race_df['Upper CI'] - enrollment_ci_race_df['Enrollment Rate']],
                fmt='none', c='black', capsize=5)
    plt.title('Racial Enrollment Rates with 95% Confidence Intervals')
    plt.xlabel('Race')
    plt.ylabel('Enrollment Rate')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Assuming 'parity_results' is already defined and contains the enrollment parity results
    parity_values = []

    # Collect enrollment parity results for racial groups excluding White
    for race, result in parity_results.items():
        if race != 'White':
            parity_value = result["Enrollment Parity vs White"]
            parity_values.append(parity_value)

    # Calculate the average enrollment parity for non-White groups
    average_parity = np.mean(parity_values) if parity_values else None

    # Output the average enrollment parity
    if average_parity is not None:
        print(f'Average Enrollment Parity (Non-White vs White): {average_parity:.2f}')
    else:
        print('No non-White groups to calculate average enrollment parity.')

def bootstrap_for_parity(admitted_students):

    # Number of bootstrap samples
    n_iterations = 100
    bootstrap_samples = []

    # Step 1: Generate bootstrap samples and calculate joint distributions
    for _ in range(n_iterations):
        sample = admitted_students.sample(frac=1, replace=True)
        joint_distribution_sample = sample.groupby(by=['gender', 'race'])['enrollment'].mean().reset_index()
        bootstrap_samples.append(joint_distribution_sample)

    # Combine bootstrap samples into a single DataFrame
    return pd.concat(bootstrap_samples)

def compute_conditional_probabilities(bootstrapped_data, admitted_students):
    conditional_probabilities = []
    for race in admitted_students["race"].unique():
        # Calculate gender conditional probability for each race
        gender_prob = (bootstrapped_data.loc[bootstrapped_data["race"] == race]
                       .groupby("gender")["enrollment"].mean())
        total_enrollment = admitted_students.loc[admitted_students["race"] == race, 'enrollment'].mean()
        gender_conditional_prob = gender_prob / total_enrollment
        conditional_probabilities.append(gender_conditional_prob)

    return pd.concat(conditional_probabilities, axis=1).mean(axis=1)

def compute_race_conditional_probabilities(bootstrapped_data, admitted_students):
    conditional_probabilities = []
    for gender in admitted_students["gender"].unique():
        # Calculate race conditional probability for each gender
        race_prob = (bootstrapped_data.loc[bootstrapped_data["gender"] == gender]
                     .groupby("race")["enrollment"].mean())
        total_enrollment = admitted_students.loc[admitted_students["gender"] == gender, 'enrollment'].mean()
        race_conditional_prob = race_prob / total_enrollment
        conditional_probabilities.append(race_conditional_prob)

    return pd.concat(conditional_probabilities, axis=1).mean(axis=1)

def create_data2(num_students):
    # Generate the race column based on the defined ratios
    race_population = np.random.choice(
        list(RACE_RATIOS.keys()),
        size=num_students,
        p=list(RACE_RATIOS.values())
    )

    # Create the DataFrame
    data = {
        'student_id': range(1, num_students + 1),
        'gpa': np.random.normal(2.0, 4.0, num_students),
        'essay_score': np.random.normal(1, 5, num_students),
        'gender': np.random.choice(['Male', 'Female'], num_students),
        'race': race_population
    }
    df2 = pd.DataFrame(data)

    # Normalize GPA and GMAT scores to a range of 0 to 0.5
    df2['normalized_gpa'] = (df2['gpa'] - 2.0) / (4.0 - 2.0) * 0.5  # Normalized to [0, 0.5]
    df2['normalized_essay_score'] = (df2['essay_score'] - 1) / (5 - 1) * 0.5  # Normalized to [0, 0.5]

    # Create a ranking score
    df2['ranking_score'] = df2['normalized_gpa'] + df2['normalized_essay_score']

    # Rank students based on the combined score in descending order
    df2 = df2.sort_values(by='ranking_score', ascending=False)
    return df2

def admit_students_2(df, race_ratios_adjusted,gender_ratios_adjusted, target_admits = 500):
    # Calculate number of admits needed for each race
    admits_by_race = {race: int(target_admits * ratio) for race, ratio in race_ratios_adjusted.items()}

    # Create a list to hold selected students
    selected_students = []

    # Sample students based on race and gender
    for race, num_needed in admits_by_race.items():
        # Select students of this race
        race_group = df[df['race'] == race]

        for gender, proportion in gender_ratios_adjusted.items():
            num_gender_needed = int(num_needed * proportion)
            gender_group = race_group[race_group['gender'] == gender]

            if len(gender_group) > 0:
                # Select top-ranked students based on the ranking score
                sampled = gender_group.head(num_gender_needed)
                selected_students.append(sampled)

    # Combine the selected students into a single DataFrame
    if selected_students:
        admitted_df = pd.concat(selected_students)
    else:
        admitted_df = pd.DataFrame()  # Return empty if no students selected

    # If the total selected is less than target admits, fill the remaining with top-ranked from overall DataFrame
    if len(admitted_df) < target_admits:
        remaining_needed = target_admits - len(admitted_df)
        remaining_students = df.drop(admitted_df.index)  # Exclude already selected
        additional_samples = remaining_students.head(remaining_needed)  # Select top-ranked remaining
        admitted_df = pd.concat([admitted_df, additional_samples])

    return admitted_df

def enrollment_with_parity(df, df2, gender_enrollment_rates, race_enrollment_rates, target_admits):
    admitted_students = admit_students(df, target_admits)
    bootstrapped_joint_distribution = bootstrap_for_parity(admitted_students)
    # Compute the mean conditional probabilities for gender using bootstrapped samples
    gender_enrollment_prob = compute_conditional_probabilities(bootstrapped_joint_distribution, admitted_students)

    # Fix the calculation to ensure sum is done correctly
    total_prob = sum(gender_enrollment_prob)
    gender_enrollment_prob_fixed = {k: 1 / v / total_prob for k, v in gender_enrollment_prob.items()}
    # Compute the mean conditional probabilities for race using bootstrapped samples
    race_enrollment_prob = compute_race_conditional_probabilities(bootstrapped_joint_distribution, admitted_students)

    # Fix the calculation for race enrollment probabilities
    total_race_prob = sum(race_enrollment_prob)
    race_enrollment_prob_fixed = {k: RACE_RATIOS[k] / v / total_race_prob for k, v in race_enrollment_prob.items()}
    race_enrollment_prob_fixed_final = {k: v / sum(race_enrollment_prob_fixed.values()) for k, v in race_enrollment_prob_fixed.items()}

    admitted_students2 = admit_students_2(df2, race_enrollment_prob_fixed_final,gender_enrollment_prob_fixed, target_admits)
    # Initialize the enrollment column
    admitted_students2['enrollment'] = 0.0

    # Assign enrollment probabilities based on gender and race
    for index, row in admitted_students2.iterrows():
        gender = row['gender']
        race = row['race']
        gender_prob = gender_enrollment_rates[gender]
        race_prob = race_enrollment_rates[race]

        # Calculate the combined probability assuming independence
        combined_prob = gender_prob * race_prob

        # Assign enrollment status based on combined probability
        admitted_students2.at[index, 'enrollment'] = 1 if np.random.rand() < combined_prob else 0
    df2.loc[:,"admit"] = 0
    df2.loc[:,"enrollment"] = 0

    df2.loc[admitted_students2.index, "admit"] = 1
    df2.loc[admitted_students2.index, "enrollment"] = admitted_students2["enrollment"]
    return df2
        