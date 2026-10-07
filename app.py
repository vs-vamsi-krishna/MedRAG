import os
import re
import pandas as pd
import streamlit as st

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from google import genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MedRAG | FDA Drug Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */

.stApp {
    background: linear-gradient(
        135deg,
        #f7fbff 0%,
        #eef6ff 50%,
        #f8fbff 100%
    );
}


/* Hide Streamlit default elements */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* Sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #ffffff 0%,
        #f7fbff 100%
    );

    border-right: 1px solid #dbe7f3;
}

[data-testid="stSidebarContent"] {
    padding: 1rem;
}


/* Sidebar collapse button */

button[kind="header"] {
    color: #1e40af !important;
}


/* Dashboard cards */

.dashboard-card {
    background: rgba(255,255,255,0.95);
    border: 1px solid #dce7f2;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 16px;

    box-shadow:
        0 5px 20px rgba(30, 80, 130, 0.07);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}

.dashboard-card:hover {
    transform: translateY(-3px);

    box-shadow:
        0 10px 28px rgba(30, 80, 130, 0.13);
}


/* Dashboard title */

.card-title {
    font-size: 17px;
    font-weight: 700;
    color: #172554;
    margin-bottom: 4px;
}

.card-subtitle {
    font-size: 11px;
    color: #64748b;
    margin-bottom: 15px;
}


/* Record count */

.stat-number {
    font-size: 30px;
    font-weight: 800;
    color: #2563eb;
    line-height: 1.1;
}

.stat-label {
    font-size: 11px;
    color: #64748b;
    margin-top: 4px;
    margin-bottom: 13px;
}


/* Mini information */

.mini-line {
    font-size: 11px;
    color: #475569;
    padding: 5px 0;
}


/* Pipeline */

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 9px 0;
}

.pipeline-icon {
    width: 32px;
    height: 32px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: #eff6ff;
    border-radius: 10px;

    font-size: 16px;
}

.pipeline-step b {
    display: block;
    font-size: 11px;
    color: #1e293b;
}

.pipeline-step small {
    display: block;
    font-size: 9px;
    color: #94a3b8;
    margin-top: 2px;
}

.pipeline-arrow {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin: -2px 0;
}


/* System status */

.status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 8px 0;

    border-bottom: 1px solid #edf2f7;

    font-size: 10px;
    color: #475569;
}

.status-row:last-child {
    border-bottom: none;
}

.status-ok {
    color: #16a34a;
    font-weight: 600;
}


/* Sidebar buttons */

.stButton > button {
    border-radius: 10px !important;
    border: 1px solid #dbe5ef !important;
    background: white !important;
    color: #334155 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    border-color: #60a5fa !important;
    color: #2563eb !important;
    transform: translateY(-1px);
}


/* Main hero */

.hero {
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        #eff7ff 100%
    );

    border: 1px solid #dce9f5;

    border-radius: 24px;

    padding: 32px;

    margin-bottom: 24px;

    box-shadow:
        0 10px 35px rgba(30, 80, 130, 0.08);

    animation: fadeUp 0.6s ease;
}


/* Hero icon */

.hero-icon {
    font-size: 42px;
    display: inline-block;

    animation:
        floatMedicine 3s ease-in-out infinite;
}


/* Hero title */

.hero-title {
    font-size: 34px;
    font-weight: 800;
    color: #172554;

    margin-top: 6px;
    margin-bottom: 7px;
}

.hero-subtitle {
    font-size: 14px;
    color: #64748b;
    max-width: 750px;
    line-height: 1.6;
}


/* Status badge */

.status-badge {
    display: inline-flex;
    align-items: center;

    gap: 6px;

    padding: 6px 11px;

    border-radius: 20px;

    background: #ecfdf5;
    color: #15803d;

    border: 1px solid #bbf7d0;

    font-size: 11px;
    font-weight: 600;

    margin-top: 14px;
}


/* Chat messages */

[data-testid="stChatMessage"] {
    border-radius: 17px;
    border: 1px solid #e1eaf3;

    padding: 14px 18px;

    box-shadow:
        0 4px 16px rgba(30, 80, 130, 0.05);

    margin-bottom: 12px;
}


/* User message */

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-user"]
) {
    background: #eff6ff;
}


/* Assistant message */

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-assistant"]
) {
    background: #ffffff;
}


/* Quick question buttons */

.quick-title {
    font-size: 13px;
    font-weight: 700;
    color: #334155;

    margin-top: 8px;
    margin-bottom: 10px;
}


/* Source card */

.source-card {
    background: #f8fbff;

    border: 1px solid #dce8f4;

    border-radius: 14px;

    padding: 14px;

    margin-bottom: 10px;

    transition: all 0.2s ease;
}

.source-card:hover {
    border-color: #93c5fd;
    transform: translateY(-1px);
}

.source-title {
    font-size: 13px;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 6px;
}

.source-text {
    font-size: 11px;
    color: #475569;
    line-height: 1.5;
}


/* Section title */

.section-title {
    font-size: 20px;
    font-weight: 750;
    color: #172554;

    margin-top: 18px;
    margin-bottom: 12px;
}


/* Footer */

.app-footer {
    text-align: center;

    color: #94a3b8;

    font-size: 10px;

    padding: 25px 0 10px;
}


/* Animations */

@keyframes fadeUp {

    from {
        opacity: 0;
        transform: translateY(10px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


@keyframes floatMedicine {

    0%, 100% {
        transform: translateY(0px) rotate(-3deg);
    }

    50% {
        transform: translateY(-8px) rotate(3deg);
    }
}


/* Spinner */

.stSpinner > div {
    border-top-color: #2563eb !important;
}


/* Chat input */

[data-testid="stChatInput"] {
    border-radius: 16px;
}


/* Responsive */

@media (max-width: 900px) {

    .hero-title {
        font-size: 27px;
    }

    .hero {
        padding: 24px;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# ============================================================
# LOAD FDA DATA
# ============================================================

@st.cache_data
def load_fda_data():

    applications_path = os.path.join(
        DATA_PATH,
        "Applications.txt"
    )

    products_path = os.path.join(
        DATA_PATH,
        "Products.txt"
    )

    marketing_path = os.path.join(
        DATA_PATH,
        "MarketingStatus.txt"
    )

    marketing_lookup_path = os.path.join(
        DATA_PATH,
        "MarketingStatus_Lookup.txt"
    )


    applications = pd.read_csv(
        applications_path,
        sep="\t",
        dtype=str,
        encoding="latin1"
    )


    products = pd.read_csv(
        products_path,
        sep="\t",
        dtype=str,
        encoding="latin1"
    )


    marketing_status = pd.read_csv(
        marketing_path,
        sep="\t",
        dtype=str,
        encoding="latin1"
    )


    marketing_lookup = pd.read_csv(
        marketing_lookup_path,
        sep="\t",
        dtype=str,
        encoding="latin1"
    )


    # Applications + Products

    app_products = pd.merge(
        applications,
        products,
        on="ApplNo",
        how="left"
    )


    # Add marketing status

    app_products_status = pd.merge(
        app_products,
        marketing_status,
        on=["ApplNo", "ProductNo"],
        how="left"
    )


    # Add marketing status description

    final_data = pd.merge(
        app_products_status,
        marketing_lookup,
        on="MarketingStatusID",
        how="left"
    )


    # Select required fields

    rag_data = final_data[
        [
            "ApplNo",
            "ApplType",
            "SponsorName",
            "ProductNo",
            "Form",
            "Strength",
            "DrugName",
            "ActiveIngredient",
            "MarketingStatusDescription"
        ]
    ].copy()


    rag_data = rag_data.fillna("Not available")


    return rag_data


# ============================================================
# CREATE DOCUMENT TEXT
# ============================================================

@st.cache_data
def create_documents_data(rag_data):

    def create_document(row):

        return f"""
Application Number: {row['ApplNo']}
Application Type: {row['ApplType']}
Sponsor: {row['SponsorName']}

Product Number: {row['ProductNo']}
Drug Name: {row['DrugName']}
Active Ingredient: {row['ActiveIngredient']}
Strength: {row['Strength']}
Dosage Form: {row['Form']}
Marketing Status: {row['MarketingStatusDescription']}
""".strip()


    rag_data = rag_data.copy()

    rag_data["document_text"] = rag_data.apply(
        create_document,
        axis=1
    )


    return rag_data


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD CHROMA VECTOR DATABASE
# ============================================================

@st.cache_resource
def load_vectorstore(_embeddings):

    vectorstore = Chroma(
        collection_name="medrag",
        embedding_function=_embeddings,
        persist_directory=CHROMA_PATH
    )

    return vectorstore


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def load_gemini():

    try:
        api_key = st.secrets["GEMINI_API_KEY"]

        return genai.Client(
            api_key=api_key
        )

    except Exception:
        return None

# ============================================================
# STRUCTURED RETRIEVAL
# ============================================================

def retrieve_documents(
    query,
    rag_data,
    vectorstore,
    k=5
):

    query_upper = query.upper().strip()


    # --------------------------------------------------------
    # 1. Application number
    # --------------------------------------------------------

    application_numbers = re.findall(
        r"\b\d{6}\b",
        query_upper
    )


    for appl_no in application_numbers:

        results = rag_data[
            rag_data["ApplNo"]
            .astype(str)
            .str.upper()
            == appl_no
        ]


        if not results.empty:

            return results.head(k)


    # --------------------------------------------------------
    # 2. Exact drug name
    # --------------------------------------------------------

    drug_names = (
        rag_data["DrugName"]
        .dropna()
        .astype(str)
        .unique()
    )


    for drug in drug_names:

        drug_upper = drug.upper().strip()


        if (
            drug_upper != "NOT AVAILABLE"
            and drug_upper in query_upper
        ):

            results = rag_data[
                rag_data["DrugName"]
                .astype(str)
                .str.upper()
                == drug_upper
            ]


            if not results.empty:

                return results.head(k)


    # --------------------------------------------------------
    # 3. Structured multi-condition search
    # --------------------------------------------------------

    conditions = []


    # Application type

    application_types = [
        "NDA",
        "ANDA",
        "BLA"
    ]


    for app_type in application_types:

        if re.search(
            rf"\b{app_type}\b",
            query_upper
        ):

            conditions.append(
                rag_data["ApplType"]
                .astype(str)
                .str.upper()
                == app_type
            )

            break


    # --------------------------------------------------------
    # Sponsor
    # --------------------------------------------------------

    sponsors = (
        rag_data["SponsorName"]
        .dropna()
        .astype(str)
        .unique()
    )


    for sponsor in sponsors:

        sponsor_upper = sponsor.upper().strip()


        if (
            sponsor_upper != "NOT AVAILABLE"
            and sponsor_upper in query_upper
        ):

            conditions.append(
                rag_data["SponsorName"]
                .astype(str)
                .str.upper()
                == sponsor_upper
            )

            break


    # --------------------------------------------------------
    # Strength
    # --------------------------------------------------------

    strength_matches = re.findall(
        r"\b\d+(?:\.\d+)?\s*%",
        query_upper
    )


    if strength_matches:

        strength = (
            strength_matches[0]
            .replace(" ", "")
        )


        conditions.append(
            rag_data["Strength"]
            .astype(str)
            .str.upper()
            .str.replace(" ", "", regex=False)
            == strength
        )


    # --------------------------------------------------------
    # Form / dosage form
    # --------------------------------------------------------

    form_keywords = [
        "SOLUTION",
        "TABLET",
        "CAPSULE",
        "INJECTION",
        "CREAM",
        "OINTMENT",
        "DROPS",
        "OPHTHALMIC",
        "ORAL",
        "TOPICAL",
        "SPRAY",
        "SUSPENSION"
    ]


    for form in form_keywords:

        if form in query_upper:

            conditions.append(
                rag_data["Form"]
                .astype(str)
                .str.upper()
                .str.contains(
                    form,
                    na=False
                )
            )

            break


    # --------------------------------------------------------
    # Active ingredient
    # --------------------------------------------------------

    ingredients = (
        rag_data["ActiveIngredient"]
        .dropna()
        .astype(str)
        .unique()
    )


    for ingredient in ingredients:

        ingredient_upper = ingredient.upper().strip()


        if (
            ingredient_upper != "NOT AVAILABLE"
            and ingredient_upper in query_upper
        ):

            conditions.append(
                rag_data["ActiveIngredient"]
                .astype(str)
                .str.upper()
                == ingredient_upper
            )

            break


    # --------------------------------------------------------
    # Apply structured filters
    # --------------------------------------------------------

    if conditions:

        mask = conditions[0]

        for condition in conditions[1:]:

            mask = mask & condition


        results = rag_data[mask]


        if not results.empty:

            return results.head(k)


    # --------------------------------------------------------
    # 4. Semantic search fallback
    # --------------------------------------------------------

    semantic_results = vectorstore.similarity_search(
        query,
        k=k
    )


    # Convert LangChain documents back into dataframe-like
    # records for the UI

    rows = []


    for doc in semantic_results:

        rows.append({

            "ApplNo":
                doc.metadata.get(
                    "ApplNo",
                    "Not available"
                ),

            "DrugName":
                doc.metadata.get(
                    "DrugName",
                    "Not available"
                ),

            "ProductNo":
                doc.metadata.get(
                    "ProductNo",
                    "Not available"
                ),

            "ApplType":
                doc.metadata.get(
                    "ApplType",
                    "Not available"
                ),

            "SponsorName":
                doc.metadata.get(
                    "SponsorName",
                    "Not available"
                ),

            "Strength":
                doc.metadata.get(
                    "Strength",
                    "Not available"
                ),

            "Form":
                doc.metadata.get(
                    "Form",
                    "Not available"
                ),

            "MarketingStatusDescription":
                doc.metadata.get(
                    "MarketingStatus",
                    "Not available"
                ),

            "document_text":
                doc.page_content
        })


    return pd.DataFrame(rows)


# ============================================================
# GENERATE GEMINI ANSWER
# ============================================================

def generate_answer(
    query,
    rag_data,
    vectorstore,
    gemini_client
):

    if gemini_client is None:

        return (
            "Gemini API key is not configured. "
            "Please set the GEMINI_API_KEY environment variable."
        ), pd.DataFrame()


    retrieved_data = retrieve_documents(
        query,
        rag_data,
        vectorstore,
        k=5
    )


    if retrieved_data.empty:

        return (
            "I couldn't find relevant information "
            "in the FDA dataset."
        ), retrieved_data


    contexts = []


    for _, row in retrieved_data.iterrows():

        contexts.append(
            row["document_text"]
        )


    context = "\n\n---\n\n".join(
        contexts
    )


    prompt = f"""
You are MedRAG, an AI assistant for FDA drug information.

Your job is to answer questions using ONLY the FDA
information provided in the context.

IMPORTANT RULES:

1. Use only the provided context.
2. Do not invent information.
3. Answer every part of the user's question.
4. If the requested information is not available,
   clearly say that it is not available in the provided
   FDA records.
5. Keep the answer clear and concise.
6. When useful, mention the application number,
   drug name, sponsor, strength, dosage form,
   and marketing status.
7. Do not mention retrieval, embeddings, vector databases,
   or internal system details unless the user asks.

FDA CONTEXT:

{context}


USER QUESTION:

{query}


ANSWER:
"""


    try:

        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )


        return (
            response.text,
            retrieved_data
        )


    except Exception as e:

        return (
            f"Sorry, I couldn't generate the answer.\n\n"
            f"Error: {str(e)}",
            retrieved_data
        )


# ============================================================
# LOAD EVERYTHING
# ============================================================

with st.spinner("Loading MedRAG..."):

    rag_data = load_fda_data()

    rag_data = create_documents_data(
        rag_data
    )

    embeddings = load_embeddings()

    vectorstore = load_vectorstore(
        embeddings
    )

    gemini_client = load_gemini()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="
        text-align:center;
        padding:8px 0 18px 0;
    ">

        <div style="
            font-size:34px;
            animation:floatMedicine 3s ease-in-out infinite;
        ">
            💊
        </div>

        <div style="
            font-size:21px;
            font-weight:800;
            color:#172554;
        ">
            MedRAG
        </div>

        <div style="
            font-size:10px;
            color:#64748b;
            margin-top:3px;
        ">
            FDA Drug Intelligence
        </div>

    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # DASHBOARD CARD 1
    # ========================================================

    st.markdown(f"""
    <div class="dashboard-card">

        <div class="card-title">
            📊 Knowledge Base
        </div>

        <div class="card-subtitle">
            FDA drug information
        </div>

        <div class="stat-number">
            {len(rag_data):,}
        </div>

        <div class="stat-label">
            FDA Drug Records
        </div>

        <div class="mini-line">
            🔎 Searchable drug information
        </div>

        <div class="mini-line">
            📋 Applications & products
        </div>

        <div class="mini-line">
            🏷️ Marketing status data
        </div>

    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # DASHBOARD CARD 2
    # ========================================================

    st.markdown("""
    <div class="dashboard-card">

        <div class="card-title">
            🔗 RAG Pipeline
        </div>

        <div class="card-subtitle">
            How MedRAG finds answers
        </div>


        <div class="pipeline-step">

            <div class="pipeline-icon">
                📄
            </div>

            <div>
                <b>FDA Dataset</b>
                <small>Structured records</small>
            </div>

        </div>


        <div class="pipeline-arrow">
            ↓
        </div>


        <div class="pipeline-step">

            <div class="pipeline-icon">
                🔎
            </div>

            <div>
                <b>Structured Retrieval</b>
                <small>Exact matching</small>
            </div>

        </div>


        <div class="pipeline-arrow">
            ↓
        </div>


        <div class="pipeline-step">

            <div class="pipeline-icon">
                🧠
            </div>

            <div>
                <b>Vector Search</b>
                <small>Semantic retrieval</small>
            </div>

        </div>


        <div class="pipeline-arrow">
            ↓
        </div>


        <div class="pipeline-step">

            <div class="pipeline-icon">
                🤖
            </div>

            <div>
                <b>Gemini AI</b>
                <small>Grounded generation</small>
            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # DASHBOARD CARD 3
    # ========================================================

    gemini_status = (
        "● Active"
        if gemini_client
        else "● Not configured"
    )

    gemini_class = (
        "status-ok"
        if gemini_client
        else "status-warning"
    )


    st.markdown(f"""
    <div class="dashboard-card">

        <div class="card-title">
            ⚡ System Status
        </div>

        <div class="card-subtitle">
            MedRAG components
        </div>


        <div class="status-row">

            <span>
                FDA Dataset
            </span>

            <span class="status-ok">
                ● Connected
            </span>

        </div>


        <div class="status-row">

            <span>
                Embeddings
            </span>

            <span class="status-ok">
                ● Ready
            </span>

        </div>


        <div class="status-row">

            <span>
                ChromaDB
            </span>

            <span class="status-ok">
                ● Ready
            </span>

        </div>


        <div class="status-row">

            <span>
                Gemini AI
            </span>

            <span class="{gemini_class}">
                {gemini_status}
            </span>

        </div>

    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # QUICK QUESTIONS
    # ========================================================

    st.markdown(
        '<div class="quick-title">💡 Quick Questions</div>',
        unsafe_allow_html=True
    )


    quick_questions = [

        "What is the status of PAREDRINE?",

        "Find an NDA from PHARMICS",

        "Tell me about application 000004",

        "Find drugs with 1% strength",

        "Which products are discontinued?"

    ]


    for question in quick_questions:

        if st.button(
            question,
            use_container_width=True
        ):

            st.session_state["selected_question"] = question


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)


    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# INITIALIZE CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# HANDLE QUICK QUESTION
# ============================================================

if "selected_question" in st.session_state:

    selected_question = (
        st.session_state.pop(
            "selected_question"
        )
    )

    st.session_state.messages.append({

        "role": "user",

        "content": selected_question

    })


    with st.spinner(
        "Searching FDA records..."
    ):

        answer, sources = generate_answer(
            selected_question,
            rag_data,
            vectorstore,
            gemini_client
        )


    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": sources

    })

    st.rerun()


# ============================================================
# MAIN HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-icon">
        💊
    </div>

    <div class="hero-title">
        MedRAG
    </div>

    <div class="hero-subtitle">

        Ask questions about FDA drug records and get
        grounded answers from the MedRAG knowledge base.

        Search applications, products, sponsors,
        strengths, dosage forms and marketing status.

    </div>

    <div class="status-badge">
        🟢 RAG System Online
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# CHAT TITLE
# ============================================================

st.markdown(
    '<div class="section-title">💬 Ask MedRAG</div>',
    unsafe_allow_html=True
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    with st.chat_message(role):

        st.markdown(
            message["content"]
        )


        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        if (
            role == "assistant"
            and message.get("sources") is not None
            and not message["sources"].empty
        ):

            sources = message["sources"]


            with st.expander(
                f"📚 FDA Sources Used ({len(sources)})"
            ):

                for index, (_, row) in enumerate(
                    sources.iterrows(),
                    start=1
                ):

                    st.markdown(f"""
                    <div class="source-card">

                        <div class="source-title">
                            Source {index}
                        </div>

                        <div class="source-text">

                            <b>Drug:</b>
                            {row.get("DrugName", "Not available")}

                            &nbsp;&nbsp;|&nbsp;&nbsp;

                            <b>Application:</b>
                            {row.get("ApplNo", "Not available")}

                            <br><br>

                            <b>Sponsor:</b>
                            {row.get("SponsorName", "Not available")}

                            <br>

                            <b>Strength:</b>
                            {row.get("Strength", "Not available")}

                            <br>

                            <b>Dosage Form:</b>
                            {row.get("Form", "Not available")}

                            <br>

                            <b>Marketing Status:</b>
                            {row.get(
                                "MarketingStatusDescription",
                                "Not available"
                            )}

                        </div>

                    </div>
                    """, unsafe_allow_html=True)


# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Ask about an FDA drug, application, sponsor, strength..."
)


if user_query:

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": user_query

    })


    with st.chat_message("user"):

        st.markdown(user_query)


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching FDA records..."
        ):

            answer, sources = generate_answer(
                user_query,
                rag_data,
                vectorstore,
                gemini_client
            )


        st.markdown(answer)


        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if (
            sources is not None
            and not sources.empty
        ):

            with st.expander(
                f"📚 FDA Sources Used ({len(sources)})"
            ):

                for index, (_, row) in enumerate(
                    sources.iterrows(),
                    start=1
                ):

                    st.markdown(f"""
                    <div class="source-card">

                        <div class="source-title">
                            Source {index}
                        </div>

                        <div class="source-text">

                            <b>Drug:</b>
                            {row.get(
                                "DrugName",
                                "Not available"
                            )}

                            &nbsp;&nbsp;|&nbsp;&nbsp;

                            <b>Application:</b>
                            {row.get(
                                "ApplNo",
                                "Not available"
                            )}

                            <br><br>

                            <b>Sponsor:</b>
                            {row.get(
                                "SponsorName",
                                "Not available"
                            )}

                            <br>

                            <b>Strength:</b>
                            {row.get(
                                "Strength",
                                "Not available"
                            )}

                            <br>

                            <b>Dosage Form:</b>
                            {row.get(
                                "Form",
                                "Not available"
                            )}

                            <br>

                            <b>Marketing Status:</b>
                            {row.get(
                                "MarketingStatusDescription",
                                "Not available"
                            )}

                        </div>

                    </div>
                    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": sources

    })


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="app-footer">

    MedRAG • FDA Drug Intelligence •
    Retrieval-Augmented Generation

    <br>

    Answers are generated from the available FDA dataset.

</div>
""", unsafe_allow_html=True)