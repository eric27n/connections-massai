# IConNYC

This is the repository for **IConNYC** - **I**dentifying **Con**nections in the **N**ew **Y**ork times **C**onnections. It is an intelligent, NLP-powered solver for NYT Connections puzzles.

IConNYC replicates the experience of the original game with a sleek web interface and smart suggestions. Under the hood, it uses Flask for the backend and Word2Vec to analyze word meanings, group related words, and solve the puzzle with human-like intuition.

## How To Run

Run the following python command in terminal:

```
python -m flask --app app run
```

Installation:

```
pip install flask numpy scipy gensim
```

NOTE: You will need to have [`numberbatch-en-19.08.txt.gz`](https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz) downloaded to be able to properly run the Flask app. Place it in the same directory as `app.py`.

Our team won first place in the MassAI competition in Spring 25.

Made with ❤️ by

-   [Eric Nugnes](https://github.com/eric27n)
-   [Keshav Garg](https://github.com/keshavgarg616)
-   [Jonathan Liu](https://github.com/jonathanliu72)
-   [Aarohee Gondkar](https://github.com/aarohee-he)
-   [Dhruv Rohilla](https://github.com/dhruvrohilla19)
