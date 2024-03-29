from flask import Flask, render_template, request, redirect
from load_data import *
import os


app = Flask(__name__)
os_path = os.getcwd()

@app.route('/')
def index():
    # Load the latest data under each category
    job = loadLatest()
    return render_template('home.html',jobs=job,categories=job.keys())

@app.route('/<name>')
def job_list(name):
    # Load full data
    jobDict = loadData()

    # Default to homepage if the name does not exist
    if name in jobDict.keys():
        # Get jobs of specified category
        jobs = jobDict[name]
        # Get number of jobs
        length = len(jobs['Title'])
        return render_template('job_listing.html',data=jobs,length=length,category=name)
    else:
        return redirect("/")

@app.route('/<folder>/<filename>')
def job_display(folder,filename):
    # Get the text file of specific job
    if os.path.exists(os.path.join(os_path,*['static','data',folder,filename+".txt"])):
        job = getJob(folder,filename)

        return render_template('job_detail.html',job=job)
    else:
        return redirect("/")


@app.route('/classify', methods=['GET','POST'])
def classify():
    # Classify the category based on input description
    if request.method == 'POST':
        # Read the content
        f_title = request.form['title']
        f_content = request.form['description']

        # Predict category based on the description
        y_pred = predictCategory(f_title+" "+f_content)

        # Set the predicted message
        predicted_message = "The category of this news is {}.".format(y_pred)

        # Return classification if form submitted
        return render_template('classify.html', predicted_message=predicted_message, title=f_title, description=f_content, predicted_category=y_pred)
    else:
        # Return empty form
        return render_template('classify.html')

@app.route('/submit', methods=['GET','POST'])
def submit():
    if request.method == 'POST':
        # Read the input of forms
        f_title = request.form['title'].strip()
        f_content = request.form['description'].strip()
        f_category = request.form.get('job-category')

        # If no category selected, auto predict the category
        if f_category == None:
            f_category = predictCategory(f_content)
        
        # Write the new job created into a text file under its category
        filename = writeToFile(f_title,f_content,f_category)

        # Load the newly created job
        job = getJob(f_category,filename)
        return render_template('job_detail.html',job=job)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500



