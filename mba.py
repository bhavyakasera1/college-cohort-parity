import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv('data/MBA.csv')
    df = df.fillna(value = {"race": "not_mentioned", "gender": "not_mentioned", "admission": "Rejected"})
    df["admission"] = df["admission"].replace("Waitlist","Admit")
    return df

# def sample_by_combined_conditions(df, gender_col, race_col, gender_ratios, race_ratios):
#     sampled_df = pd.concat([
#         group.sample(frac=gender_ratios[group.iloc[0][gender_col]] * race_ratios[group.iloc[0][race_col]], random_state=42)
#         for _, group in df.groupby([gender_col, race_col])
#     ])
#     return sampled_df

def sample_by_combined_conditions(df, gender_col, race_col, gender_ratios, race_ratios):
    sampled_groups = []
    for _, group in df.groupby([gender_col, race_col]):
        gender = group.iloc[0][gender_col]
        race = group.iloc[0][race_col]

        # Get sampling fractions
        gender_ratio = gender_ratios.get(gender, 1)
        race_ratio = race_ratios.get(race, 1)
        combined_fraction = gender_ratio * race_ratio

        # Sample if fraction is greater than 0 and the group is not empty
        if combined_fraction > 0 and not group.empty:
            try:
                sampled_group = group.sample(frac=combined_fraction, random_state=42)
                sampled_groups.append(sampled_group)
            except ValueError:
                pass

    # Concatenate only non-empty groups
    if sampled_groups:
        return pd.concat(sampled_groups)
    else:
        # Return an empty DataFrame if no groups were sampled
        return pd.DataFrame(columns=df.columns)

def plot_distributions(sampling_ratio_sets, df):
    # Create plots for each set of sampling ratios
    fig, axes = plt.subplots(len(sampling_ratio_sets), 2, figsize=(16, 12))

    # Loop over each set of sampling ratios
    for i, ratio_set in enumerate(sampling_ratio_sets):
        gender_ratios = ratio_set['gender_ratios']
        race_ratios = ratio_set['race_ratios']

        # Filter the DataFrame where 'admission' == 'Admitted'
        admitted_df = df[df['admission'] == 'Admit']

        # Sample based on combined gender and race ratios
        sampled_df = sample_by_combined_conditions(admitted_df, 'gender', 'race', gender_ratios, race_ratios)

        # Create a contingency table for gender across races
        gender_race_table = pd.crosstab(sampled_df['gender'], sampled_df['race']).reset_index()
        melted_gender_race = gender_race_table.melt(id_vars='gender', var_name='race', value_name='count')

        # Create a contingency table for race across genders
        race_gender_table = pd.crosstab(sampled_df['race'], sampled_df['gender']).reset_index()
        melted_race_gender = race_gender_table.melt(id_vars='race', var_name='gender', value_name='count')

        # Plot 1: Gender composition across races
        sns.barplot(data=melted_gender_race, x='race', y='count', hue='gender', ax=axes[i, 0], palette='pastel')
        axes[i, 0].set_title(f'Sampling Set {i + 1} - Gender Composition Across Races')
        axes[i, 0].set_xlabel('Race')
        axes[i, 0].set_ylabel('Count')

        # Plot 2: Race distribution across genders
        sns.barplot(data=melted_race_gender, x='gender', y='count', hue='race', ax=axes[i, 1], palette='pastel')
        axes[i, 1].set_title(f'Sampling Set {i + 1} - Race Distribution Across Genders')
        axes[i, 1].set_xlabel('Gender')
        axes[i, 1].set_ylabel('Count')

    # Adjust layout
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    df = load_data()
    # gender_sampling_ratios = {'Male': 0.8, 'Female': 0.8}
    # race_sampling_ratios = {'White': 0.8, 'Black': 0.8, 'Asian': 0.8, 'Hispanic': 0.8, "Other" : 0.8, "not_mentioned" : 0.8}
    # sampled_df = sample_by_combined_conditions(df[df["admission"] == "Admit"], 'gender', 'race', gender_sampling_ratios, race_sampling_ratios)
    # df['enrollment_all_80'] = 0
    # df.loc[sampled_df.index, 'enrollment_all_80'] = 1
    # df[df["admission"] == "Admitted"]
    # df.groupby(by =["gender","enrollment_all_80"]).count()
    sampling_ratio_sets = [
        {'gender_ratios': {'Male': 0.8, 'Female': 0.8}, 'race_ratios': {'White': 0.8, 'Black': 0.8, 'Asian': 0.8, 'Hispanic': 0.8, 'Other': 0.8, 'not_mentioned': 0.8}},
        {'gender_ratios': {'Male': 0.8, 'Female': 0.7}, 'race_ratios': {'White': 0.8, 'Black': 0.7, 'Asian': 0.7, 'Hispanic': 0.7, 'Other': 0.7, 'not_mentioned': 0.8}},
        {'gender_ratios': {'Male': 0.8, 'Female': 0.6}, 'race_ratios': {'White': 0.8, 'Black': 0.6, 'Asian': 0.6, 'Hispanic': 0.6, 'Other': 0.6, 'not_mentioned': 0.8}},
    ]
    # Initialize a list to store final cohort sizes for each ratio set and compositions
    cohort_sizes = []
    cohort_compositions = []

    # Loop over each set of sampling ratios
    for i, ratio_set in enumerate(sampling_ratio_sets):
        gender_ratios = ratio_set['gender_ratios']
        race_ratios = ratio_set['race_ratios']

        # Filter the DataFrame where 'admission' == 'Admit'
        admitted_df = df[df['admission'] == 'Admit']

        # Sample based on combined gender and race ratios
        sampled_df = sample_by_combined_conditions(admitted_df, 'gender', 'race', gender_ratios, race_ratios)

        # Calculate the size of the sampled cohort
        cohort_size = len(sampled_df)
        cohort_sizes.append(cohort_size)

        # Calculate gender and race composition
        gender_counts = sampled_df['gender'].value_counts(normalize=True).to_dict()
        race_counts = sampled_df['race'].value_counts(normalize=True).to_dict()
        cohort_compositions.append((gender_counts, race_counts))