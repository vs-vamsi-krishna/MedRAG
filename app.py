import os
import re

import pandas as pd
import streamlit as st

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
    initial_sidebar_state="expanded",
)


# ============================================================
# DARK THEME + CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html,
    body,
    [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 75% 10%,
                rgba(37, 99, 235, 0.10),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #080d17 0%,
                #0b1220 45%,
                #080d17 100%
            );

        color: #f8fafc;
    }

    /* Remove Streamlit default header/footer */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* Main content */

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #101827 0%,
                #0b1220 100%
            );

        border-right: 1px solid #243044;
    }

    [data-testid="stSidebarContent"] {
        padding: 1rem;
    }

    /* Sidebar collapse button */

    [data-testid="stSidebar"] button {
        color: #e2e8f0 !important;
    }

    /* ========================================================
       SIDEBAR BRAND
       ======================================================== */

    .brand-container {
        padding: 10px 5px 25px 5px;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        font-size: 42px;
        animation: medicineFloat 3s ease-in-out infinite;
    }

    .brand-title {
        font-size: 25px;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1;
    }

    .brand-subtitle {
        margin-top: 6px;
        font-size: 11px;
        color: #94a3b8;
    }


    /* ========================================================
       SIDEBAR NAV
       ======================================================== */

    .nav-item {
        display: flex;
        align-items: center;
        gap: 12px;

        padding: 13px 14px;
        margin: 6px 0;

        border-radius: 12px;

        color: #cbd5e1;
        font-size: 14px;
        font-weight: 500;

        transition: all 0.2s ease;
    }

    .nav-item:hover {
        background: #182338;
        color: #ffffff;
        transform: translateX(3px);
    }

    .nav-item.active {
        background:
            linear-gradient(
                135deg,
                #1d4ed8,
                #2563eb
            );

        color: white;

        box-shadow:
            0 8px 25px rgba(37, 99, 235, 0.25);
    }

    .nav-icon {
        font-size: 20px;
        width: 25px;
        text-align: center;
    }


    /* ========================================================
       CONNECTION CARD
       ======================================================== */

    .connection-card {
        margin-top: 22px;

        padding: 18px;

        border-radius: 16px;

        background:
            linear-gradient(
                145deg,
                #111b2d,
                #0d1625
            );

        border: 1px solid #26344a;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.20);
    }

    .connection-title {
        color: #f8fafc;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 16px;
    }

    .connection-row {
        display: flex;
        align-items: center;
        gap: 10px;

        color: #cbd5e1;

        font-size: 12px;

        padding: 8px 0;
    }

    .green-dot {
        width: 9px;
        height: 9px;

        border-radius: 50%;

        background: #22c55e;

        box-shadow:
            0 0 10px rgba(34, 197, 94, 0.6);
    }

    .connection-status {
        margin-left: auto;
        color: #22c55e;
        font-size: 11px;
        font-weight: 600;
    }


    /* ========================================================
       MAIN TOPBAR
       ======================================================== */

    .topbar {
        display: flex;
        justify-content: flex-end;
        align-items: center;

        gap: 15px;

        margin-bottom: 8px;
    }

    .topbar-icon {
        width: 38px;
        height: 38px;

        border-radius: 10px;

        display: flex;
        align-items: center;
        justify-content: center;

        background: #111a2a;
        border: 1px solid #27354a;

        color: #e2e8f0;

        font-size: 18px;

        text-decoration: none;

        transition: all 0.2s ease;
    }

    .topbar-icon:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
        color: white;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        padding: 30px 8px 25px 8px;
        animation: fadeUp 0.6s ease;
    }

    .hero-row {
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .hero-icon {
        font-size: 48px;

        animation:
            medicineFloat 3s ease-in-out infinite;
    }

    .hero-title {
        color: #f8fafc;

        font-size: 40px;

        font-weight: 800;

        letter-spacing: -1px;

        margin: 0;
    }

    .hero-subtitle {
        margin-top: 7px;

        color: #94a3b8;

        font-size: 15px;

        line-height: 1.6;

        max-width: 800px;
    }

    .online-badge {
        display: inline-flex;

        align-items: center;

        gap: 8px;

        margin-top: 15px;

        padding: 7px 13px;

        border-radius: 20px;

        background: rgba(34, 197, 94, 0.10);

        border: 1px solid rgba(34, 197, 94, 0.25);

        color: #4ade80;

        font-size: 11px;

        font-weight: 600;
    }


    /* ========================================================
       SECTION TITLE
       ======================================================== */

    .section-title {
        color: #f8fafc;

        font-size: 25px;

        font-weight: 750;

        margin-top: 15px;

        margin-bottom: 15px;
    }

    .section-subtitle {
        color: #64748b;

        font-size: 12px;

        margin-top: -8px;

        margin-bottom: 18px;
    }


    /* ========================================================
       USER / ASSISTANT CHAT CARDS
       ======================================================== */

    [data-testid="stChatMessage"] {
        border-radius: 16px !important;

        margin-bottom: 14px !important;

        padding: 14px 18px !important;

        border: 1px solid #26354b !important;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.16) !important;
    }

    /* User */

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {
        background:
            linear-gradient(
                135deg,
                #111c2e,
                #101a2a
            ) !important;
    }

    /* Assistant */

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-assistant"]
    ) {
        background:
            linear-gradient(
                135deg,
                #111a29,
                #0d1624
            ) !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: #f1f5f9 !important;
    }


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    [data-testid="stChatInput"] {
        background: #0e1726 !important;

        border: 1px solid #30415b !important;

        border-radius: 15px !important;

        box-shadow:
            0 10px 35px rgba(0, 0, 0, 0.25) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;

        color: #f8fafc !important;

        font-size: 14px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }

    [data-testid="stChatInput"] button {
        background: #2563eb !important;

        border-radius: 10px !important;

        color: white !important;
    }


    /* ========================================================
       QUICK QUESTION BUTTONS
       ======================================================== */

    .quick-title {
        color: #cbd5e1;

        font-size: 13px;

        font-weight: 700;

        margin-top: 10px;

        margin-bottom: 10px;
    }

    .stButton > button {
        background: #111a2a !important;

        color: #cbd5e1 !important;

        border: 1px solid #26354a !important;

        border-radius: 10px !important;

        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #172338 !important;

        color: #ffffff !important;

        border-color: #3b82f6 !important;

        transform: translateY(-2px);
    }


    /* ========================================================
       SOURCE CARDS
       ======================================================== */

    .source-card {
        background:
            linear-gradient(
                145deg,
                #101a2a,
                #0d1624
            );

        border: 1px solid #27364c;

        border-radius: 13px;

        padding: 14px;

        margin-bottom: 10px;

        transition: all 0.2s ease;
    }

    .source-card:hover {
        border-color: #3b82f6;

        transform: translateY(-2px);
    }

    .source-title {
        color: #60a5fa;

        font-size: 13px;

        font-weight: 700;

        margin-bottom: 8px;
    }

    .source-text {
        color: #cbd5e1;

        font-size: 11px;

        line-height: 1.7;
    }

    .source-text b {
        color: #f8fafc;
    }


    /* ========================================================
       SIDEBAR CLEAR BUTTON
       ======================================================== */

    .clear-label {
        color: #64748b;

        font-size: 10px;

        text-align: center;

        margin-top: 10px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .app-footer {
        text-align: center;

        color: #475569;

        font-size: 10px;

        padding: 30px 0 10px;
    }


    /* ========================================================
       ANIMATIONS
       ======================================================== */

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes medicineFloat {
        0%, 100% {
            transform: translateY(0px) rotate(-3deg);
        }

        50% {
            transform: translateY(-7px) rotate(3deg);
        }
    }


    /* ========================================================
       RESPONSIVE
       ======================================================== */

    @media (max-width: 900px) {

        .hero-title {
            font-size: 30px;
        }

        .hero-subtitle {
            font-size: 13px;
        }

        .hero-icon {
            font-size: 38px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
)

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "chroma_db",
)


# ============================================================
# LOAD FDA DATA
# ============================================================

@st.cache_data
def load_fda_data():

    applications_path = os.path.join(
        DATA_PATH,
        "Applications.txt",
    )

    products_path = os.path.join(
        DATA_PATH,
        "Products.txt",
    )

    marketing_path = os.path.join(
        DATA_PATH,
        "MarketingStatus.txt",
    )

    marketing_lookup_path = os.path.join(
        DATA_PATH,
        "MarketingStatus_Lookup.txt",
    )

    applications = pd.read_csv(
        applications_path,
        sep="\t",
        dtype=str,
        encoding="latin1",
    )

    products = pd.read_csv(
        products_path,
        sep="\t",
        dtype=str,
        encoding="latin1",
    )

    marketing_status = pd.read_csv(
        marketing_path,
        sep="\t",
        dtype=str,
        encoding="latin1",
    )

    marketing_lookup = pd.read_csv(
        marketing_lookup_path,
        sep="\t",
        dtype=str,
        encoding="latin1",
    )

    # Applications + Products

    app_products = pd.merge(
        applications,
        products,
        on="ApplNo",
        how="left",
    )

    # Add marketing status

    app_products_status = pd.merge(
        app_products,
        marketing_status,
        on=["ApplNo", "ProductNo"],
        how="left",
    )

    # Add marketing status description

    final_data = pd.merge(
        app_products_status,
        marketing_lookup,
        on="MarketingStatusID",
        how="left",
    )

    # Select fields used by MedRAG

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
            "MarketingStatusDescription",
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
        axis=1,
    )

    return rag_data


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


# ============================================================
# CHROMA
# ============================================================

@st.cache_resource
def load_vectorstore(_embeddings):

    vectorstore = Chroma(
        collection_name="medrag",
        embedding_function=_embeddings,
        persist_directory=CHROMA_PATH,
    )

    return vectorstore


# ============================================================
# GEMINI
# ============================================================

@st.cache_resource
def load_gemini():

    try:

        api_key = st.secrets["GEMINI_API_KEY"]

        return genai.Client(
            api_key=api_key,
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
    k=5,
):

    query_upper = query.upper().strip()

    # --------------------------------------------------------
    # 1. Application number
    # --------------------------------------------------------

    application_numbers = re.findall(
        r"\b\d{6}\b",
        query_upper,
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
        "BLA",
    ]

    for app_type in application_types:

        if re.search(
            rf"\b{app_type}\b",
            query_upper,
        ):

            conditions.append(
                rag_data["ApplType"]
                .astype(str)
                .str.upper()
                == app_type
            )

            break


    # Sponsor

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


    # Strength

    strength_matches = re.findall(
        r"\b\d+(?:\.\d+)?\s*%",
        query_upper,
    )

    if strength_matches:

        strength = strength_matches[0].replace(
            " ",
            "",
        )

        conditions.append(
            rag_data["Strength"]
            .astype(str)
            .str.upper()
            .str.replace(
                " ",
                "",
                regex=False,
            )
            == strength
        )


    # Dosage form

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
        "SUSPENSION",
    ]

    for form in form_keywords:

        if form in query_upper:

            conditions.append(
                rag_data["Form"]
                .astype(str)
                .str.upper()
                .str.contains(
                    form,
                    na=False,
                )
            )

            break


    # Active ingredient

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
        k=k,
    )

    rows = []

    for doc in semantic_results:

        rows.append(
            {
                "ApplNo": doc.metadata.get(
                    "ApplNo",
                    "Not available",
                ),

                "DrugName": doc.metadata.get(
                    "DrugName",
                    "Not available",
                ),

                "ProductNo": doc.metadata.get(
                    "ProductNo",
                    "Not available",
                ),

                "ApplType": doc.metadata.get(
                    "ApplType",
                    "Not available",
                ),

                "SponsorName": doc.metadata.get(
                    "SponsorName",
                    "Not available",
                ),

                "Strength": doc.metadata.get(
                    "Strength",
                    "Not available",
                ),

                "Form": doc.metadata.get(
                    "Form",
                    "Not available",
                ),

                "MarketingStatusDescription": doc.metadata.get(
                    "MarketingStatus",
                    "Not available",
                ),

                "document_text": doc.page_content,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# GENERATE GEMINI ANSWER
# ============================================================

def generate_answer(
    query,
    rag_data,
    vectorstore,
    gemini_client,
):

    if gemini_client is None:

        return (
            "Gemini API key is not configured. "
            "Please configure GEMINI_API_KEY in Streamlit Secrets."
        ), pd.DataFrame()


    retrieved_data = retrieve_documents(
        query,
        rag_data,
        vectorstore,
        k=5,
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

Answer the user's question using ONLY the FDA information
provided in the context.

IMPORTANT RULES:

1. Use only the provided context.
2. Do not invent information.
3. Answer every part of the user's question.
4. If information is not available, clearly say so.
5. Keep the answer clear and concise.
6. When useful, mention application number, drug name,
   sponsor, strength, dosage form, and marketing status.
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
            contents=prompt,
        )

        return (
            response.text,
            retrieved_data,
        )

    except Exception as e:

        return (
            "Sorry, I couldn't generate the answer.\n\n"
            f"Error: {str(e)}",
            retrieved_data,
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
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # Brand

    st.markdown(
        """
        <div class="brand-container">

            <div class="brand-row">

                <div class="brand-icon">
                    💊
                </div>

                <div>

                    <div class="brand-title">
                        MedRAG
                    </div>

                    <div class="brand-subtitle">
                        FDA Drug Information Assistant
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # Navigation

    st.markdown(
        """
        <div class="nav-item active">
            <span class="nav-icon">💬</span>
            <span>Ask MedRAG</span>
        </div>

        <div class="nav-item">
            <span class="nav-icon">⚡</span>
            <span>System Status</span>
        </div>

        <div class="nav-item">
            <span class="nav-icon">🗄️</span>
            <span>FDA Dataset</span>
        </div>

        <div class="nav-item">
            <span class="nav-icon">🔗</span>
            <span>Embeddings</span>
        </div>

        <div class="nav-item">
            <span class="nav-icon">ℹ️</span>
            <span>About</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # Connection status

    gemini_connected = gemini_client is not None

    gemini_status = (
        "Connected"
        if gemini_connected
        else "Not configured"
    )

    gemini_color = (
        "#22c55e"
        if gemini_connected
        else "#ef4444"
    )


    st.markdown(
        f"""
        <div class="connection-card">

            <div class="connection-title">
                🟢 Connection Status
            </div>

            <div class="connection-row">
                <span>🗄️</span>
                <span>ChromaDB</span>
                <span class="connection-status">
                    🟢 Connected
                </span>
            </div>

            <div class="connection-row">
                <span>🧠</span>
                <span>Embeddings</span>
                <span class="connection-status">
                    🟢 Ready
                </span>
            </div>

            <div class="connection-row">
                <span>🤖</span>
                <span>Gemini AI</span>
                <span
                    class="connection-status"
                    style="color:{gemini_color};"
                >
                    {"🟢" if gemini_connected else "🔴"}
                    {gemini_status}
                </span>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # Dataset information

    st.markdown(
        f"""
        <div class="connection-card">

            <div class="connection-title">
                🗄️ FDA Dataset
            </div>

            <div
                style="
                    color:#60a5fa;
                    font-size:28px;
                    font-weight:800;
                "
            >
                {len(rag_data):,}
            </div>

            <div
                style="
                    color:#64748b;
                    font-size:10px;
                    margin-top:3px;
                "
            >
                searchable drug records
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # Clear conversation

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# TOP RIGHT CONTROLS
# ============================================================

st.markdown(
    """
    <div class="topbar">

        <div class="topbar-icon">
            ☀️
        </div>

        <a
            class="topbar-icon"
            href="https://github.com/vs-vamsi-krishna/MedRAG"
            target="_blank"
        >
            ◉
        </a>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-row">

            <div class="hero-icon">
                💬
            </div>

            <div>

                <div class="hero-title">
                    Ask MedRAG
                </div>

                <div class="hero-subtitle">
                    Get information about FDA drugs,
                    applications, sponsors, strengths,
                    dosage forms, and marketing status.
                </div>

                <div class="online-badge">
                    <span>●</span>
                    MedRAG System Online
                </div>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# QUICK QUESTIONS
# ============================================================

st.markdown(
    """
    <div class="quick-title">
        💡 Quick Questions
    </div>
    """,
    unsafe_allow_html=True,
)


quick_questions = [
    "What is the status of PAREDRINE?",
    "Find an NDA from PHARMICS",
    "Tell me about application 000004",
    "Find drugs with 1% strength",
    "Which products are discontinued?",
]


cols = st.columns(2)

for index, question in enumerate(quick_questions):

    with cols[index % 2]:

        if st.button(
            question,
            use_container_width=True,
            key=f"quick_{index}",
        ):

            st.session_state.selected_question = question


# ============================================================
# HANDLE QUICK QUESTION
# ============================================================

if "selected_question" in st.session_state:

    selected_question = st.session_state.pop(
        "selected_question"
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": selected_question,
        }
    )

    with st.spinner(
        "Searching FDA records..."
    ):

        answer, sources = generate_answer(
            selected_question,
            rag_data,
            vectorstore,
            gemini_client,
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )

    st.rerun()


# ============================================================
# CHAT SECTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        💬 Conversation
    </div>

    <div class="section-subtitle">
        Ask questions about FDA drug records.
        MedRAG answers using the available dataset.
    </div>
    """,
    unsafe_allow_html=True,
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


        # Sources

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
                    start=1,
                ):

                    drug_name = row.get(
                        "DrugName",
                        "Not available",
                    )

                    appl_no = row.get(
                        "ApplNo",
                        "Not available",
                    )

                    sponsor = row.get(
                        "SponsorName",
                        "Not available",
                    )

                    strength = row.get(
                        "Strength",
                        "Not available",
                    )

                    form = row.get(
                        "Form",
                        "Not available",
                    )

                    status = row.get(
                        "MarketingStatusDescription",
                        "Not available",
                    )

                    st.markdown(
                        f"""
                        <div class="source-card">

                            <div class="source-title">
                                Source {index}
                            </div>

                            <div class="source-text">

                                <b>Drug:</b>
                                {drug_name}

                                <br>

                                <b>Application:</b>
                                {appl_no}

                                <br>

                                <b>Sponsor:</b>
                                {sponsor}

                                <br>

                                <b>Strength:</b>
                                {strength}

                                <br>

                                <b>Dosage Form:</b>
                                {form}

                                <br>

                                <b>Marketing Status:</b>
                                {status}

                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Ask about an FDA drug, application, sponsor, strength..."
)


if user_query:

    # Add user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )


    # Generate response

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching FDA records..."
        ):

            answer, sources = generate_answer(
                user_query,
                rag_data,
                vectorstore,
                gemini_client,
            )

        st.markdown(answer)


        # Sources

        if (
            sources is not None
            and not sources.empty
        ):

            with st.expander(
                f"📚 FDA Sources Used ({len(sources)})"
            ):

                for index, (_, row) in enumerate(
                    sources.iterrows(),
                    start=1,
                ):

                    drug_name = row.get(
                        "DrugName",
                        "Not available",
                    )

                    appl_no = row.get(
                        "ApplNo",
                        "Not available",
                    )

                    sponsor = row.get(
                        "SponsorName",
                        "Not available",
                    )

                    strength = row.get(
                        "Strength",
                        "Not available",
                    )

                    form = row.get(
                        "Form",
                        "Not available",
                    )

                    status = row.get(
                        "MarketingStatusDescription",
                        "Not available",
                    )

                    st.markdown(
                        f"""
                        <div class="source-card">

                            <div class="source-title">
                                Source {index}
                            </div>

                            <div class="source-text">

                                <b>Drug:</b>
                                {drug_name}

                                <br>

                                <b>Application:</b>
                                {appl_no}

                                <br>

                                <b>Sponsor:</b>
                                {sponsor}

                                <br>

                                <b>Strength:</b>
                                {strength}

                                <br>

                                <b>Dosage Form:</b>
                                {form}

                                <br>

                                <b>Marketing Status:</b>
                                {status}

                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


    # Save assistant response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">

        MedRAG • FDA Drug Intelligence •
        Retrieval-Augmented Generation

        <br><br>

        Answers are generated from the available FDA dataset.

    </div>
    """,
    unsafe_allow_html=True,
)