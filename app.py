import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import docx2txt
import re
import tempfile
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data with error handling
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except:
            st.warning("Could not download punkt_tab. Using alternative tokenization.")
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

# Download required NLTK resources
download_nltk_data()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy English model not found. Please run: python -m spacy download en_core_web_sm")
    st.stop()

# Function to analyze resume against job description
def analyze_resume(resume_text, jd_text, resume_name):
    # Preprocess texts
    resume_processed = preprocess_text(resume_text)
    jd_processed = preprocess_text(jd_text)
    
    # Calculate similarity using TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, jd_processed])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Convert to percentage (0-100)
    relevance_score = min(100, max(0, int(similarity * 100)))
    
    # Extract skills from JD and resume
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    
    # Find matched and missing skills
    matched_skills = list(set(jd_skills) & set(resume_skills))
    missing_skills = list(set(jd_skills) - set(resume_skills))
    
    # Determine verdict
    if relevance_score >= 70:
        verdict = "High"
    elif relevance_score >= 40:
        verdict = "Medium"
    else:
        verdict = "Low"
    
    # Generate suggestions
    suggestions = generate_suggestions(missing_skills, relevance_score)
    
    return {
        'resume_name': resume_name,
        'relevance_score': relevance_score,
        'verdict': verdict,
        'matched_skills': matched_skills[:10],  # Show top 10
        'missing_elements': missing_skills[:5],  # Show top 5 missing
        'suggestions': suggestions
    }

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        # Use simple regex-based tokenization as fallback
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        words = [word for word in words if word not in stop_words]
    except:
        # Fallback tokenization if NLTK fails
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    
    return ' '.join(words)

def extract_skills(text):
    # Simple skill extraction using common tech skills
    tech_skills = [
        'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 
        'vue', 'node', 'express', 'django', 'flask', 'machine learning', 'deep learning',
        'data analysis', 'tableau', 'power bi', 'excel', 'aws', 'azure', 'google cloud',
        'docker', 'kubernetes', 'ci/cd', 'devops', 'agile', 'scrum', 'project management',
        'communication', 'leadership', 'problem solving', 'teamwork'
    ]
    
    found_skills = []
    for skill in tech_skills:
        if skill in text.lower():
            found_skills.append(skill)
    
    return found_skills

def generate_suggestions(missing_skills, score):
    suggestions = []
    
    if score < 40:
        suggestions.append("Consider gaining more relevant experience in the required field")
        suggestions.append("Highlight transferable skills that may be relevant to this role")
    
    if missing_skills:
        suggestions.append(f"Consider developing these skills: {', '.join(missing_skills[:3])}")
    
    if score < 60:
        suggestions.append("Tailor your resume to include more keywords from the job description")
        suggestions.append("Quantify your achievements with metrics and numbers")
    
    if not suggestions:
        suggestions.append("Your resume is well-aligned with the job requirements")
    
    return suggestions

# Set up page configuration
st.set_page_config(
    page_title="Automated Resume Relevance Check System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'resumes' not in st.session_state:
    st.session_state.resumes = []
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'results' not in st.session_state:
    st.session_state.results = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .score-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .score-medium {
        color: #F9A825;
        font-weight: bold;
    }
    .score-low {
        color: #C62828;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">Automated Resume Relevance Check System</h1>', unsafe_allow_html=True)
st.markdown("### AI-powered resume evaluation against job descriptions")

# Sidebar for navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3209/3209095.png", width=80)
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a section", 
                               ["Home", "Upload Documents", "Analysis Results", "Dashboard"])

# Home page
if app_mode == "Home":
    st.markdown("""
    <div class="highlight">
    <h3>Welcome to the Automated Resume Relevance Check System</h3>
    <p>This tool helps recruiters and placement teams quickly evaluate resumes against job descriptions, 
    providing relevance scores and improvement feedback.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <h4>🚀 Fast Evaluation</h4>
            <p>Process hundreds of resumes in minutes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h4>🎯 Accurate Matching</h4>
            <p>AI-powered semantic analysis combined with keyword matching</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center;">
            <h4>💡 Actionable Insights</h4>
            <p>Get detailed feedback on missing skills and qualifications</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### How it works:")
    
    steps = [
        ("1. Upload Job Description", "Paste or upload the job description you want to evaluate against"),
        ("2. Upload Resumes", "Upload multiple resumes in PDF or DOCX format"),
        ("3. Analyze", "The system processes documents and calculates relevance scores"),
        ("4. Review Results", "See relevance scores, missing elements, and improvement suggestions")
    ]
    
    for i, (title, description) in enumerate(steps):
        if i % 2 == 0:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.info(f"**{title}**")
            with col2:
                st.write(description)
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(description)
            with col2:
                st.info(f"**{title}**")
    
    st.markdown("---")
    st.markdown("### Supported Formats")
    st.write("- Job Descriptions: Text input, PDF, DOCX")
    st.write("- Resumes: PDF, DOCX")

# Upload documents page
elif app_mode == "Upload Documents":
    st.markdown('<h2 class="sub-header">Upload Job Description and Resumes</h2>', unsafe_allow_html=True)
    
    # Job Description Upload
    st.markdown("#### Job Description")
    jd_option = st.radio("Choose input method for Job Description:", 
                        ("Text Input", "File Upload"))
    
    if jd_option == "Text Input":
        jd_text = st.text_area("Paste Job Description here", height=200)
        if jd_text:
            st.session_state.job_description = jd_text
    else:
        jd_file = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])
        if jd_file is not None:
            # Extract text from uploaded JD file
            if jd_file.type == "application/pdf":
                with fitz.open(stream=jd_file.read(), filetype="pdf") as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    st.session_state.job_description = text
            elif jd_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = docx2txt.process(jd_file)
                st.session_state.job_description = text
            
            st.success("Job Description uploaded successfully!")
    
    if st.session_state.job_description:
        with st.expander("View Job Description"):
            st.text(st.session_state.job_description)
    
    st.markdown("---")
    
    # Resume Upload
    st.markdown("#### Resumes")
    uploaded_resumes = st.file_uploader("Upload Resumes (PDF or DOCX)", 
                                      type=["pdf", "docx"], 
                                      accept_multiple_files=True)
    
    if uploaded_resumes:
        for resume in uploaded_resumes:
            if resume not in [r['file'] for r in st.session_state.resumes]:
                # Extract text from resume
                if resume.type == "application/pdf":
                    with fitz.open(stream=resume.read(), filetype="pdf") as doc:
                        text = ""
                        for page in doc:
                            text += page.get_text()
                        # Reset file pointer for potential future use
                        resume.seek(0)
                elif resume.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = docx2txt.process(resume)
                    # Reset file pointer
                    resume.seek(0)
                
                st.session_state.resumes.append({
                    'file': resume,
                    'name': resume.name,
                    'text': text
                })
        
        st.success(f"{len(uploaded_resumes)} resume(s) uploaded successfully!")
        
        # Display uploaded resumes
        st.markdown("**Uploaded Resumes:**")
        for i, resume in enumerate(st.session_state.resumes):
            st.write(f"{i+1}. {resume['name']}")
    
    # Analyze button
    if st.session_state.job_description and st.session_state.resumes:
        if st.button("Analyze Resumes", type="primary"):
            st.session_state.analysis_done = True
            st.session_state.results = []
            
            # Process each resume
            for resume in st.session_state.resumes:
                result = analyze_resume(resume['text'], st.session_state.job_description, resume['name'])
                st.session_state.results.append(result)
            
            st.success("Analysis completed!")
            # Use the current method to rerun the app
            st.rerun()

# Analysis Results page
elif app_mode == "Analysis Results":
    if not st.session_state.analysis_done:
        st.warning("Please upload documents and run analysis first.")
        st.stop()
    
    st.markdown('<h2 class="sub-header">Analysis Results</h2>', unsafe_allow_html=True)
    
    # Summary statistics
    scores = [result['relevance_score'] for result in st.session_state.results]
    avg_score = np.mean(scores) if scores else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Resumes", len(st.session_state.results))
    with col2:
        st.metric("Average Score", f"{avg_score:.1f}%")
    with col3:
        high_fit = len([r for r in st.session_state.results if r['verdict'] == 'High'])
        st.metric("High Fit Resumes", high_fit)
    
    # Filter options
    st.markdown("### Filter Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider("Minimum Relevance Score", 0, 100, 0)
    with col2:
        verdict_filter = st.selectbox("Verdict", ["All", "High", "Medium", "Low"])
    with col3:
        sort_by = st.selectbox("Sort By", ["Relevance Score", "Name"])
    
    # Filter results
    filtered_results = st.session_state.results.copy()
    filtered_results = [r for r in filtered_results if r['relevance_score'] >= min_score]
    
    if verdict_filter != "All":
        filtered_results = [r for r in filtered_results if r['verdict'] == verdict_filter]
    
    if sort_by == "Relevance Score":
        filtered_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    else:
        filtered_results.sort(key=lambda x: x['resume_name'])
    
    # Display results
    for result in filtered_results:
        with st.expander(f"{result['resume_name']} - Score: {result['relevance_score']}% - {result['verdict']} Fit"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Relevance Score**")
                # Display score with color coding
                score = result['relevance_score']
                if score >= 70:
                    st.markdown(f'<p class="score-high">{score}%</p>', unsafe_allow_html=True)
                elif score >= 40:
                    st.markdown(f'<p class="score-medium">{score}%</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="score-low">{score}%</p>', unsafe_allow_html=True)
                
                # Progress bar
                st.progress(score/100)
                
                st.markdown("**Verdict**")
                verdict = result['verdict']
                if verdict == "High":
                    st.success(verdict)
                elif verdict == "Medium":
                    st.warning(verdict)
                else:
                    st.error(verdict)
            
            with col2:
                st.markdown("**Matched Skills**")
                if result['matched_skills']:
                    for skill in result['matched_skills']:
                        st.markdown(f"- {skill}")
                else:
                    st.write("No skills matched")
            
            st.markdown("**Missing Elements**")
            if result['missing_elements']:
                for element in result['missing_elements']:
                    st.markdown(f"- {element}")
            else:
                st.info("No major missing elements")
            
            st.markdown("**Suggestions for Improvement**")
            if result['suggestions']:
                for suggestion in result['suggestions']:
                    st.markdown(f"- {suggestion}")
            else:
                st.info("No specific suggestions")

# Dashboard page
elif app_mode == "Dashboard":
    if not st.session_state.analysis_done:
        st.warning("Please upload documents and run analysis first.")
        st.stop()
    
    st.markdown('<h2 class="sub-header">Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Prepare data for visualizations
    df = pd.DataFrame(st.session_state.results)
    
    # Score distribution chart
    st.markdown("### Score Distribution")
    fig = px.histogram(df, x="relevance_score", nbins=10, 
                      title="Distribution of Relevance Scores")
    st.plotly_chart(fig, use_container_width=True)
    
    # Verdict distribution
    st.markdown("### Fit Verdict Distribution")
    verdict_counts = df['verdict'].value_counts().reset_index()
    verdict_counts.columns = ['Verdict', 'Count']
    
    fig = px.pie(verdict_counts, values='Count', names='Verdict', 
                title="Proportion of High, Medium, and Low Fit Resumes")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top missing skills
    st.markdown("### Common Missing Skills")
    all_missing = []
    for missing_list in df['missing_elements']:
        all_missing.extend(missing_list)
    
    if all_missing:
        # Count occurrences of each missing skill
        missing_counter = Counter(all_missing)
        top_missing = pd.DataFrame(missing_counter.most_common(10), columns=['Skill', 'Count'])
        
        fig = px.bar(top_missing, x='Count', y='Skill', orientation='h',
                    title="Top 10 Most Frequently Missing Skills")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing skills detected across resumes")
    
    # Score vs resume length (approximate)
    st.markdown("### Score vs Resume Content Length")
    df['resume_length'] = df['resume_name'].apply(lambda x: next((r['text'] for r in st.session_state.resumes if r['name'] == x), ""))
    df['resume_length'] = df['resume_length'].apply(len)
    
    fig = px.scatter(df, x='resume_length', y='relevance_score', 
                    hover_data=['resume_name'],
                    title="Relevance Score vs Resume Length")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    # This is already running as a Streamlit app
    pass