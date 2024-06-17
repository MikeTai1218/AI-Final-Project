# AI-Final-Project

Diverse musical notation generation for piano with fine-tuned Long Short-Term Memory (LSTM)

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Result](#result)

## Installation

To set up this project locally, follow these steps:

1. Clone the repository:
    ```sh
    https://github.com/Mike1ife/AI-Final-Project.git
    ```
2. Navigate to the project directory:
    ```sh
    cd AI-Final-Project
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Data
We obtained [midi files](http://www.piano-midi.de/midi_files.htm) from famous musicians.
Note.csv contains notes of each file extracted from [Data](https://github.com/Mike1ife/AI-Final-Project/tree/main/Data)

## Usage
```sh
python main.py [--arg]
```
### Arguments
<pre>
-h                    show this help message and exit
--readcsv             read notes from csv file
--train               train your own model
</pre>

## Project Structure
- main.py: The main script for running the project.
- preprocess.py: Loads midi files and processes notes.
- model.py: Methods for training, evaluation and music generation.

### Result
https://github.com/Mike1ife/AI-Final-Project/assets/132564989/6567a581-107f-4070-9818-44482419a2db



