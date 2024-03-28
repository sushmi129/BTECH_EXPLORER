from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import joblib

app = Flask(__name__, static_url_path="/static")


# Read the data from 'Data.csv' and filter it
df = pd.read_csv("Data.csv")
df_test = pd.read_csv("test.csv")
rf_model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

def find_colleges(rank, gender, category, branch, dataset):
    filtered_colleges = []

    for index, entry in dataset.iterrows():
        if entry['GENDER'] == gender and entry['CATEGORY'] == category and entry['Last Rank (2022)'] >= rank and entry['Branch'] == branch:
            filtered_colleges.append(entry['College Name'])

    return filtered_colleges[:10]


def find_colleges_by_district(district, dataset):
    unique_colleges = set()

    # Replace 'District' with the actual column name containing district information
    district_column_name = 'Place'  # Replace with the actual column name
    for index, entry in dataset.iterrows():
        if entry[district_column_name] == district:
            unique_colleges.add(entry['College Name'])

    return list(unique_colleges)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/college_predictor", methods=["GET", "POST"])
def college_predictor():
    if request.method == 'POST':
        rank = int(request.form['rank'])
        gender = request.form['gender']
        category = request.form['category']

        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        category_encoded = label_encoders['Category'].transform([category])[0]

        # Make prediction
        prediction = rf_model.predict([[rank, gender_encoded, category_encoded]])

        prediction_flat = prediction.flatten()
        # Find the records in the original dataset that are near the input rank
        nearby_records = df_test.iloc[(df_test['Rank'] - rank).abs().argsort()[:5]]
        college_list = nearby_records[['Rank', 'Branch', 'College']]


        # Extract predicted college and branch
        predicted_college = label_encoders['College'].inverse_transform([prediction_flat[0]])[0]
        predicted_branch = label_encoders['Branch'].inverse_transform([prediction_flat[1]])[0]

        return render_template('college_predictor.html', college=predicted_college, branch=predicted_branch, college_list = college_list)
    return render_template('college_predictor.html', college=None, branch=None, college_list = None)


@app.route("/probable_colleges", methods=["GET", "POST"])
def probable_colleges():
    if request.method == 'POST':
        rank_input = int(request.form['rank'])
        gender_input = request.form['gender']
        category_input = request.form['category']
        branch_input = request.form['branch']

        colleges_list = find_colleges(rank_input, gender_input, category_input, branch_input, df)
        return render_template('probable_colleges.html', colleges=colleges_list)
    return render_template('probable_colleges.html', colleges=None)


@app.route('/explore_colleges', methods=['GET', 'POST'])
def explore_colleges():
    district = request.form.get('district')
    colleges = find_colleges_by_district(district, df)
    return render_template('explore_colleges.html', colleges=colleges)


@app.route('/branch_cutoff', methods=['POST', 'GET'])
def branch_cutoff():
    if request.method == 'POST':
        college_name = request.form['college-code']
        gender = request.form['gender']
        category = request.form['category']

        college_column_name = next((col for col in df.columns if col.lower() == 'college code'), None)
        gender_column_name = next((col for col in df.columns if col.lower() == 'gender'), None)
        category_column_name = next((col for col in df.columns if col.lower() == 'category'), None)
        cutoff_column_name = next((col for col in df.columns if col.lower() == 'last rank (2022)'), None)

        if college_column_name is None or gender_column_name is None or category_column_name is None or cutoff_column_name is None:
            print("Column names not found. Please check your dataset and update the column names.")
            return

        # Filter the dataset based on input parameters
        filtered_data = df[(df[college_column_name] == college_name) &
                            (df[gender_column_name] == gender) &
                            (df[category_column_name] == category)]

        # Display cutoff ranks in the form of a bar graph with annotations
        plt.figure(figsize=(10, 6))
        bars = plt.bar(filtered_data['Branch'], filtered_data[cutoff_column_name])
        plt.xlabel('Branch')
        plt.ylabel('Cutoff Rank')
        plt.title(f'Cutoff Ranks for {college_name} ({gender}, {category})')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

        x = ['CSE', 'ECE', 'EEE', 'CIVIL', 'MECH']
        plt.xticks(range(len(x)), x)
        # Add annotations with cutoff ranks on the bars
        for bar, rank in zip(bars, filtered_data[cutoff_column_name]):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.1, str(rank), ha='center')

        # Save the plot to a BytesIO object
        plt.savefig("static/images/my_pic.png")
        return render_template('branch_cutoff.html', plot_data= "static/images/my_pic.png")

    # Render the template without plot data for initial page load
    return render_template('branch_cutoff.html', plot_data= None)


if __name__ == "__main__":
    app.run(debug=True)
