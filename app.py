import streamlit as st
from inference import MODEL_FILES, load_artifacts, classify

st.set_page_config(page_title="AI vs. Human Text Detector", page_icon="📝")
st.title("AI vs. Human Text Detector")
st.caption("An individual machine learning project by Joshua-Bernard Takere")
st.write("Compare how three trained classifiers label a passage of text.")

@st.cache_resource
def get_artifacts(name):
    return load_artifacts(name)

name = st.selectbox("Choose a model", list(MODEL_FILES))
text = st.text_area("Enter a passage", height=200)
if st.button("Classify text", type="primary"):
    try:
        model, vectorizer = get_artifacts(name)
        label, probability = classify(text, model, vectorizer)
        st.success(f"Prediction: {label}")
        if probability is not None:
            st.write(f"Model probability for this label: {probability:.1%}")
        else:
            st.caption("This LinearSVC model returns a label without a probability.")
    except ValueError as exc:
        st.warning(str(exc))
    except FileNotFoundError:
        st.error("Model files are missing. Restore the files in the models folder.")
st.caption("Experimental predictions can be wrong. They are not proof of authorship; model probabilities are not verified accuracy.")
