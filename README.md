# QuizCreator

Upload your PDF study notes, pick a topic, and get multiple-choice questions generated from your own material.

## How it works

1. **Ingest:** uploaded PDFs are loaded page by page with LangChain (`tasks/task_3`).
2. **Embed:** pages are split into 1,000-character chunks, embedded with Vertex AI `textembedding-gecko`, and stored in ChromaDB (`tasks/task_4`, `tasks/task_5`).
3. **Generate:** for the chosen topic, relevant chunks are retrieved and Gemini writes each question, its answer choices, and an explanation as JSON (`tasks/task_8`).
4. **Quiz:** a Streamlit app shows one question at a time and checks your answers (`tasks/task_9`, `tasks/task_10`).

## Run it

You need a Google Cloud project with Vertex AI enabled.

```bash
pip install -r requirements.txt
gcloud auth application-default login
cd tasks/task_10
streamlit run task_10.py
```

Set your own GCP project ID in `embed_config` in `task_10.py`.

---

Built on the [RadicalX](https://github.com/radicalxdev) Quizify starter (MIT).
