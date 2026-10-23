# Character Relationship Network for English Long Texts

An interactive reading-support tool developed by Group 65 for **AILT9019: AI Literacy II**.

## Project Overview

Character-dense novels can be difficult to follow. Readers may encounter dozens of characters, alternative names, titles, and indirect references, making it easy to forget who is who and how frequently different characters appear together. This problem is especially noticeable in novels such as *Pride and Prejudice*, *Jane Eyre*, and *War and Peace*.

This project transforms an English long-form text into an interactive character co-occurrence network. The application identifies person names, performs limited name normalization, counts how often pairs of characters occur near one another, and displays the result as a weighted network graph in a browser.

The intended users are undergraduate students studying English literature who need a quick visual aid while reading and preparing for class discussions.

## How It Works

The planned workflow is:

1. The user uploads an English `.txt` file or pastes English text into the application.
2. The system uses natural language processing to identify person names.
3. A conservative alias and title-cleaning step combines known variants such as `Elizabeth`, `Lizzy`, and `Miss Bennet`.
4. The system counts character co-occurrences by paragraph or sliding text window.
5. A weighted network graph is generated and displayed in the browser.
6. The user can inspect the graph by zooming, dragging nodes, hovering over relationships, and filtering weak edges.

In the graph, nodes represent detected characters. An edge connects two characters that appear near each other, while its weight or thickness represents their co-occurrence frequency.

> **Important:** Co-occurrence does not prove the real nature of a relationship. A strong edge does not necessarily mean that two characters are friends, relatives, enemies, or romantic partners.

## Planned Features

### Core features

- Upload an English plain-text file.
- Detect person names using spaCy or a similar local NLP library.
- Handle a small built-in alias table and common title words such as Mr., Mrs., Miss, and Dr.
- Count character co-occurrences by paragraph or sliding window.
- Build a weighted network using NetworkX.
- Display an interactive graph in a local browser-based application.
- Show clear validation messages for empty, unsupported, non-English, or unsuitable input.

### Extended features

- Paste text directly into the application.
- Manually add or remove detected names.
- Upload or edit an alias table.
- Adjust the window size and minimum edge weight.
- Export the graph as standalone HTML or PNG.
- Download node and edge data as CSV.

## Technology Stack

- **Python** for the application and processing pipeline
- **Streamlit or Gradio** for the local web interface
- **spaCy** for person-name recognition
- **NetworkX** for graph construction and analysis
- **PyVis or Plotly** for interactive browser-based visualization

The core version is designed to run locally and does not require an external API, user account, or cloud deployment.

## Evaluation

The application will be evaluated using two or three public-domain English texts with well-known character structures. A manually prepared list of major characters and expected relationships will be used as a reference.

Initial success criteria include:

- approximately 70% recall for the top ten main characters;
- visible edges between expected major character pairs;
- central characters appearing as highly connected or visually central nodes; and
- safe handling of empty files, non-text files, non-English input, and texts with too few character relationships.

## Scope and Limitations

The project is a reading aid rather than a literary-analysis authority. It does not:

- determine the semantic relationship between characters;
- generate character biographies or plot summaries;
- replace close reading or academic judgment;
- guarantee perfect person-name recognition;
- perform fully automatic alias clustering in the first version;
- detect the copyright status of uploaded material; or
- upload, store, or redistribute user text in the core local version.

Because named-entity recognition and alias normalization can make mistakes, detected names and network connections should always be interpreted critically.

## Privacy and Responsible Use

The core application processes text locally. Uploaded content is not intended to be sent to an external service or retained by the application. Users remain responsible for ensuring that they have permission to process the text they provide, particularly copyrighted or sensitive material.

## Running the Application

The application is currently under development. Once the first working version is added, the expected setup will be:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The application will then open as a local browser-based web app. It is not planned as a native desktop or mobile application.

## Repository Structure

```text
AILT9019/
├── proposal/      Project proposal documents
├── README.md      Project overview and usage guide
└── .gitignore     Files excluded from version control
```

The repository structure will be expanded as the source code, tests, sample data, and final project materials are developed.

## Project Status

**In development.** The revised proposal has been completed, and the next stage is to implement and test the minimum end-to-end prototype.
