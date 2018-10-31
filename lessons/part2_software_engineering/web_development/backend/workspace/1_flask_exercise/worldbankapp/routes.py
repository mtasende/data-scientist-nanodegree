from worldbankapp import app

from flask import render_template

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')
    
@app.route('/project-one')
def project_one():
    return render_template('project_one.html')

# TODO: Add another route. You can use any names you want
@app.route('/my_page')
def my_page():
    return render_template('my_page.html')

# TODO: Start the web app per the instructions in the instructions.md file and make sure your new html file renders correctly.