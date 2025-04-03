import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "C:/Users/vinyak/Downloads/myData.csv"
data = pd.read_csv(file_path)

def analyze_graph(symmetry=False, linearity=False):
    print(f"Graph Analysis: {'symmetric' if symmetry else 'asymmetric'}.")
    print(f"Graph Analysis: Relationship is {'linear' if linearity else 'non-linear'}.")

def age_distribution(df):
    plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black'), plt.title('Age Distribution'), plt.xlabel('Age'), plt.ylabel('Frequency'), plt.show()
    print("Inference: People aged 30-50 are most represented.")

def glucose_boxplot(df):
    sns.boxplot(x='Outcome', y='Glucose', data=df, palette='coolwarm'), plt.title('Glucose Levels by Diabetes Outcome'), plt.xlabel('Diabetes Outcome (0: No, 1: Yes)'), plt.ylabel('Glucose Level'), plt.show()
    print("Inference: Diabetics have higher glucose levels.")

def correlation_heatmap(df):
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5), plt.title('Correlation Heatmap'), plt.show()
    print("Inference: Glucose and BMI correlate with diabetes.")

def pregnancies_bar_chart(df):
    df.groupby('Pregnancies')['Outcome'].mean().plot(kind='bar', color='orange', edgecolor='black'), plt.title('Pregnancies vs. Diabetes Outcome'), plt.xlabel('Pregnancies'), plt.ylabel('Avg. Diabetes Outcome'), plt.show()
    print("Inference: Higher pregnancies relate to higher diabetes proportion.")

def bmi_distribution(df):
    plt.hist(df['BMI'], bins=10, color='green', edgecolor='black'), plt.title('BMI Distribution'), plt.xlabel('BMI'), plt.ylabel('Frequency'), plt.show()
    print("Inference: Most have BMI 25-40 (overweight/obese).")

def glucose_bp_scatter(df):
    plt.scatter(df['Glucose'], df['BloodPressure'], alpha=0.7, c=df['Outcome'], cmap='coolwarm'), plt.title('Glucose vs. Blood Pressure'), plt.xlabel('Glucose'), plt.ylabel('Blood Pressure'), plt.colorbar(label='Diabetes Outcome'), plt.show()
    print("Inference: High glucose and blood pressure indicate higher diabetes likelihood.")

def outcome_pie_chart(df):
    df['Outcome'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange']), plt.title('Outcome Distribution'), plt.ylabel(''), plt.show()
    print("Inference: About 35% have diabetes.")

def main():
    options = {
        1: (age_distribution, "Age Distribution (Histogram)"),
        2: (glucose_boxplot, "Glucose Levels by Diabetes Outcome (Box Plot)"),
        3: (correlation_heatmap, "Correlation Heatmap"),
        4: (pregnancies_bar_chart, "Pregnancies vs. Diabetes Outcome (Bar Chart)"),
        5: (bmi_distribution, "BMI Distribution (Histogram)"),
        6: (glucose_bp_scatter, "Glucose vs. Blood Pressure (Scatter Plot)"),
        7: (outcome_pie_chart, "Outcome Distribution (Pie Chart)")
    }

    while True:
        print("\nSelect a graph to visualize:")
        for key, (_, description) in options.items():
            print(f"\t{key}. {description}")

        choice = int(input("\nEnter choice (1-7): "))
        if choice in options:
            options[choice][0](data)
        else:
            print("Invalid choice.")

        if input("\nContinue? (y/yes): ").lower() not in ['y', 'yes']:
            print("Exiting.")
            break

if __name__ == "__main__":
    main()