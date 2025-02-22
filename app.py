from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/action', methods=['POST'])
def action():
    return render_template('home.html', message="Home Page. Howdy!")

@app.route('/action2', methods=['POST'])
def action2():
    return render_template('page2.html', message="Page 2! What's up")

if __name__ == '__main__':
    app.run(debug=True)