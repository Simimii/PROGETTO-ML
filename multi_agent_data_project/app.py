# app.py
import streamlit as st
from pathlib import Path
import sys
import shutil
import re 
import base64
import time
import pandas as pd 
import traceback 

# ---  Streamlit Configuration ---
st.set_page_config(
    page_title="REPLY Data Agents LUISS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configurazione Iniziale del Path e Import dell'Orchestratore ---
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from agents.smart_data_analyst import run_orchestrator
except ImportError as e_import:
    st.error(f"Fatal Error: Could not import 'run_orchestrator'. Details: {e_import}\n"
             f"Current Project Root: {PROJECT_ROOT}\n"
             f"Python Sys Path: {sys.path}\n"
             f"Please ensure 'agents/__init__.py' exists and 'agents/smart_data_analyst.py' is correct.")
    st.stop()
except Exception as e_generic:
    st.error(f"Fatal Error during initial setup or import: {e_generic}")
    st.stop()

# --- Definizione della Directory Temporanea per Output Agenti e Funzione di Pulizia ---
TEMP_AGENT_OUTPUTS_DIR = PROJECT_ROOT / "temp_agent_outputs"

def clear_temp_outputs_on_startup():
    feedback_messages = []
    if not TEMP_AGENT_OUTPUTS_DIR.exists():
        try:
            TEMP_AGENT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            feedback_messages.append(f"Directory 'temp_agent_outputs' creata: {TEMP_AGENT_OUTPUTS_DIR}")
        except Exception as e_mkdir:
            feedback_messages.append(f"FATAL ERROR creazione 'temp_agent_outputs' {TEMP_AGENT_OUTPUTS_DIR}: {e_mkdir}")
            return feedback_messages

    feedback_messages.append(f"Check temporary directory for output agents: {TEMP_AGENT_OUTPUTS_DIR}...")
    deleted_files_count = 0
    deleted_dirs_count = 0
    cleanup_errors = []
    try:
        for item in TEMP_AGENT_OUTPUTS_DIR.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                    deleted_files_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_dirs_count += 1
            except Exception as e_cleanup_item:
                cleanup_errors.append(f"Error {item.name}: {e_cleanup_item}")
    except Exception as e_iter:
         cleanup_errors.append(f"Error {TEMP_AGENT_OUTPUTS_DIR}: {e_iter}")

    if cleanup_errors:
        for err in cleanup_errors:
            feedback_messages.append(f"CLEANUP WARNING: {err}")
    if deleted_files_count > 0 or deleted_dirs_count > 0:
        feedback_messages.append(f"Pulizia output temporanei completata: {deleted_files_count} file(s) e {deleted_dirs_count} dir(s) rimossi.")
    else:
        feedback_messages.append("'temp_agent_outputs' già pulita o vuota.")
    return feedback_messages

if 'cleaned_on_this_session_startup' not in st.session_state:
    st.session_state.cleanup_feedback_messages_for_sidebar = clear_temp_outputs_on_startup()
    st.session_state.cleaned_on_this_session_startup = True

# --- CSS Personalizzato ---
PRIMARY_COLOR = "#181e25"    # Sfondo app scuro
SECONDARY_COLOR = "#05d34a"  # Verde Reply
TEXT_COLOR_ON_DARK_BG = "#EAEAEA" # Testo chiaro per sfondi scuri
TEXT_COLOR_ON_LIGHT_BG = "#181e25" # Testo scuro per sfondi chiari (es. sidebar)
ASSISTANT_MSG_BG = "#2E3440" # Sfondo messaggi assistente
USER_MSG_BG = "#05d34a"      # Sfondo messaggi utente (verde Reply)
INPUT_AREA_BG = "#222831"    # Sfondo per l'area di input
SIDEBAR_BG_COLOR = "#FFFFFF"   # Sidebar bianca
SIDEBAR_TEXT_COLOR = TEXT_COLOR_ON_LIGHT_BG # Testo scuro per sidebar

st.markdown(f"""
<style>
    /* Sfondo generale dell'app */
    .stApp {{
        background-color: {PRIMARY_COLOR};
    }}
    /* Colore testo generale nel container principale */
    .main .block-container > div:first-child {{ /* Target il primo div nel block-container del main */
        color: {SECONDARY_COLOR};
    }}
    /* Header e Titoli nel main container */
    .main .block-container h1, .main .block-container h2, .main .block-container h3 {{
        color: {SECONDARY_COLOR} !important;
        text-align: center;
    }}
     .main .st-emotion-cache-1wda3go h1, /* Target h1 dentro il contenitore del testo principale */
    .main div[data-testid="stVerticalBlock"] h1 {{ /* Altro tentativo di targettare h1 nel main */
        color: {SECONDARY_COLOR} !important; /* Verde Reply */
        text-align: center !important; /* Assicura centratura */
    }}
    /* Testo Markdown nel main container */
    .main .block-container .stMarkdown p, .main .block-container .stMarkdown li {{
        color: {SECONDARY_COLOR};
    }}
    h1 {{
        color: {SECONDARY_COLOR} !important;
        text-align: center !important;
    }}

    /* Contenitore Messaggi Chat */
    div[data-testid="stChatMessage"] > div[class*="message-container"] {{ /* Selettore aggiornato per Streamlit >= 1.30 circa */
        border-radius: 10px;
        padding: 0.8em 1em;
        margin-bottom: 0.8em;
        border: 1px solid #4A4A4A;
    }}
    /* Messaggi Utente */
    div[data-testid="stChatMessage"][aria-label*="user"] > div[class*="message-container"] {{
        background-color: {USER_MSG_BG};
    }}
    div[data-testid="stChatMessage"][aria-label*="user"] div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stChatMessage"][aria-label*="user"] div[data-testid="stMarkdownContainer"] li {{
        color: {TEXT_COLOR_ON_LIGHT_BG} !important; /* Testo scuro su sfondo verde */
    }}
    /* Messaggi Assistente */
    div[data-testid="stChatMessage"][aria-label*="assistant"] > div[class*="message-container"] {{
        background-color: {ASSISTANT_MSG_BG};
    }}
    div[data-testid="stChatMessage"][aria-label*="assistant"] div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stChatMessage"][aria-label*="assistant"] div[data-testid="stMarkdownContainer"] li {{
        color: {TEXT_COLOR_ON_DARK_BG};
    }}

    /* Input Text Box */
    div[data-testid="stChatInput"] {{
        background-color: {PRIMARY_COLOR};
        border-top: 2px solid {SECONDARY_COLOR};
        padding-top: 10px;
    }}
    div[data-testid="stChatInput"] div[data-baseweb="base-input"] > textarea {{
        border-left: none !important;
        border-radius: 10px !important; /* Angoli smussati */
        background-color: {INPUT_AREA_BG} !important;
        color: {TEXT_COLOR_ON_DARK_BG} !important;
    }}
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: #7A828F !important;
    }}
    /* Bottone Invio */
    button[data-testid="stChatInputSubmitButton"] {{
        background-color: {SECONDARY_COLOR} !important;
        color: {PRIMARY_COLOR} !important;
        border-radius: 8px !important;
        border: none !important;
    }}
    button[data-testid="stChatInputSubmitButton"]:hover {{
        filter: brightness(0.9);
    }}
    button[data-testid="stChatInputSubmitButton"] svg {{
        fill: {PRIMARY_COLOR} !important;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] > div:first-child {{ /* Target il container principale della sidebar */
        background-color: {SIDEBAR_BG_COLOR} !important;
    }}
    /* Testo e titoli nella sidebar */
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stTextLabel, /* Etichette widgets come st.radio */
    section[data-testid="stSidebar"] .stSubheader,
    section[data-testid="stSidebar"] .stButton > button {{ /* Testo del bottone */
        color: {SIDEBAR_TEXT_COLOR} !important;
    }}
    /* Specifico per radio items (opzioni di navigazione) */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div {{
        color: {SIDEBAR_TEXT_COLOR} !important; /* Testo dell'opzione radio */
        font-size: 1.05em;
    }}
    /* Evidenziazione dell'opzione radio selezionata */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label input:checked + div {{
        color: {SECONDARY_COLOR} !important; /* Verde Reply per l'opzione attiva */
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

# --- Navigazione Sidebar e Funzioni Pagina ---

def get_image_as_base64_str_util(path: Path) -> str | None:
    if not path.is_file():
        print(f"WARNING: Logo file not found at: {path}")
        return None
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"ERROR loading image {path.name}: {e}")
        return None

def display_main_header():
    reply_logo_b64 = get_image_as_base64_str_util(ASSETS_DIR / "REPLY_logo.png")
    luiss_logo_b64 = get_image_as_base64_str_util(ASSETS_DIR / "LUISS_logo.png")

    col1, col_spacer, col2 = st.columns([2,1,2])
    with col1:
        if reply_logo_b64:
            st.image(f"data:image/png;base64,{reply_logo_b64}", width=220)
        else:
            st.markdown("<h2 style='text-align: left; margin-top: 10px;'>REPLY</h2>", unsafe_allow_html=True)
    with col2:
        if luiss_logo_b64:
            st.image(f"data:image/png;base64,{luiss_logo_b64}", width=170)
        else:
            st.markdown("<h2 style='text-align: right; margin-top: 10px;'>LUISS</h2>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)


def chat_agent_page_func():
   
    st.title("3SM's Multi-Agent System")

    if "chat_page_messages" not in st.session_state:
        st.session_state.chat_page_messages = [
            {"role": "assistant", "content": "What analysis would you like to perform today?", "image_path": None}
        ]

    chat_display_area = st.container()
    with chat_display_area:
        for message in st.session_state.chat_page_messages:
            avatar_icon = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar_icon):
                st.markdown(message["content"])
                if message.get("image_path"):
                    img_path_obj_history = Path(message["image_path"])
                    if img_path_obj_history.is_file():
                        try:
                            st.image(str(img_path_obj_history), use_container_width=True, caption=f"Generated Chart: {img_path_obj_history.name}")
                        except Exception as e_img_hist:
                            st.caption(f"(Error displaying saved image {img_path_obj_history.name}: {e_img_hist})")

    if user_query_prompt := st.chat_input("Start with a question!"):
        st.session_state.chat_page_messages.append({"role": "user", "content": user_query_prompt, "image_path": None})
        with chat_display_area:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_query_prompt)

        with chat_display_area:
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                response_placeholder.markdown("Thinking... ⚙️")

                assistant_text_for_markdown = "Failed to elaborate"
                actual_image_path_to_display = None

                try:
                    orchestrator_output_str = run_orchestrator(user_query_prompt)
                    print(f"DEBUG APP: Orchestrator Raw Output String: '{orchestrator_output_str}'")

                    assistant_text_for_markdown = orchestrator_output_str
                    
                    path_pattern_options = [
                        r"A chart has been generated and saved at:\s*(?:`|\*\*)?([A-Za-z]:[\\/][^<>:\"|?*\s]+?\.(?:png|jpg|jpeg|gif|bmp|svg))(?:`|\*\*)?",
                        r"You can view the chart at the following location:\s*(?:`|\*\*)?([A-Za-z]:[\\/][^<>:\"|?*\s]+?\.(?:png|jpg|jpeg|gif|bmp|svg))(?:`|\*\*)?",
                        r"The chart has been saved at the following path:\s*(?:`|\*\*)?([A-Za-z]:[\\/][^<>:\"|?*\s]+?\.(?:png|jpg|jpeg|gif|bmp|svg))(?:`|\*\*)?",
                        r"The chart is available at:\s*(?:`|\*\*)?([A-Za-z]:[\\/][^<>:\"|?*\s]+?\.(?:png|jpg|jpeg|gif|bmp|svg))(?:`|\*\*)?",
                        r"chart.*? at:\s*(?:`|\*\*)?([A-Za-z]:[\\/][^<>:\"|?*\s]+?\.(?:png|jpg|jpeg|gif|bmp|svg))(?:`|\*\*)?"
                    ]

                    match = None
                    active_pattern_for_substitution = None

                    for p_pattern in path_pattern_options:
                        match = re.search(p_pattern, orchestrator_output_str, re.IGNORECASE)
                        if match:
                            print(f"DEBUG APP: Regex Matched with pattern: {p_pattern}")
                            active_pattern_for_substitution = p_pattern
                            break
                    
                    print(f"DEBUG APP: Final Regex Match Object for chart path: {match}")

                    if match and active_pattern_for_substitution:
                        extracted_path_str = match.group(1).strip()
                        image_file_path_obj = Path(extracted_path_str.replace("\\", "/"))
                        resolved_image_path = image_file_path_obj.resolve() 
                        
                        print(f"DEBUG APP: Extracted Image Path Object: {image_file_path_obj}")
                        print(f"DEBUG APP: Resolved Image Path for is_file() check: {resolved_image_path}")

                        if resolved_image_path.is_file():
                            actual_image_path_to_display = str(resolved_image_path)
                            print(f"DEBUG APP: Image file IS VALID and found: {actual_image_path_to_display}")
                            
                            assistant_text_for_markdown = re.sub(
                                active_pattern_for_substitution,
                                "\n*The generated chart is displayed below.*",
                                assistant_text_for_markdown,
                                flags=re.IGNORECASE
                            ).strip()
                        else:
                            actual_image_path_to_display = None
                            st.sidebar.warning(f"App Warning: Image mentioned ({resolved_image_path.name}) NOT FOUND at: {resolved_image_path}")
                            print(f"DEBUG APP: Image file NOT found at: {resolved_image_path}")
                            
                            charts_dir_content = []
                            try:
                                charts_dir_path_to_check = resolved_image_path.parent 
                                if charts_dir_path_to_check.exists() and charts_dir_path_to_check.is_dir():
                                    charts_dir_content = [f.name for f in charts_dir_path_to_check.iterdir()]
                                    print(f"DEBUG APP: Content of charts directory ({charts_dir_path_to_check}): {charts_dir_content}")
                                    if resolved_image_path.name in charts_dir_content:
                                        print(f"DEBUG APP: File name {resolved_image_path.name} IS in charts directory listing, but is_file() failed. Possible permission/timing/caching issue or subtle path difference?")
                                    else:
                                        print(f"DEBUG APP: File name {resolved_image_path.name} IS NOT in charts directory listing.")
                                else:
                                    print(f"DEBUG APP: Charts directory {charts_dir_path_to_check} does not exist or is not a directory.")
                            except Exception as e_listdir:
                                print(f"DEBUG APP: Error listing charts directory: {e_listdir}")
                    else:
                        actual_image_path_to_display = None
                        print("DEBUG APP: No chart path pattern found in SDA response or active_pattern_for_substitution is None.")

                except Exception as e_orch:
                    assistant_text_for_markdown = f"I'm sorry, a critical error occurred while processing your request: {type(e_orch).__name__} - {str(e_orch)}"
                    st.sidebar.error(f"Orchestrator Error Detail in App: {traceback.format_exc(limit=5)}")
                    print(f"ERROR APP: Orchestrator exception: {type(e_orch).__name__} - {e_orch}")
                    traceback.print_exc()

                response_placeholder.markdown(assistant_text_for_markdown)

                if actual_image_path_to_display:
                    print(f"DEBUG APP: Attempting to display image with st.image: {actual_image_path_to_display}")
                    try:
                        st.image(actual_image_path_to_display, use_container_width=True, caption="Generated Chart")
                    except Exception as e_st_image:
                        st.error(f"Error displaying the image: {e_st_image}")
                        print(f"ERROR APP: st.image failed: {e_st_image}")
                        traceback.print_exc()
                else:
                    print("DEBUG APP: No valid image path to display with st.image.")

                st.session_state.chat_page_messages.append({
                    "role": "assistant",
                    "content": assistant_text_for_markdown,
                    "image_path": actual_image_path_to_display
                })
        st.rerun()

def explore_datasets_page_func():
    display_main_header()
    st.title("Explore Available Datasets")
    st.markdown("Select a dataset to visualize insights.")

    DATA_DIR = PROJECT_ROOT / "data"
    dataset_files_map = {
        "Accessi Utenti": "EntryAccessoAmministrati.csv",
        "Accredito Stipendi": "EntryAccreditoStipendi.csv",
        "Pendolarismo": "EntryPendolarismo.csv",
        "Personale Amministrato": "EntryAmministrati.csv"
    }
    available_in_dir = [f.name for f in DATA_DIR.iterdir() if f.is_file() and f.name.endswith(".csv")]
    display_options = {name: fname for name, fname in dataset_files_map.items() if fname in available_in_dir}

    if not display_options:
        st.warning(f"Nessun dataset trovato in '{DATA_DIR}'. Verifica la presenza dei file CSV.")
        return

    selected_name = st.selectbox("Select a dataset:", options=list(display_options.keys()))

    if selected_name:
        csv_file = display_options[selected_name]
        file_path = DATA_DIR / csv_file
        st.markdown(f"### Details: `{csv_file}`")
        try:
            with st.spinner(f"Charging {csv_file}..."):
                # Aggiunta di encoding='utf-8' per maggiore robustezza
                df = pd.read_csv(file_path, encoding='utf-8')
            st.success(f"Dataset '{csv_file}' successfully retrieved!")
            st.markdown(f"**Dimensions:** {df.shape[0]} rows, {df.shape[1]} colomns.")

            if st.checkbox("Show first rows", value=True, key=f"head_{csv_file}"): st.dataframe(df.head())
            if st.checkbox("Columns information", key=f"info_{csv_file}"):
                # df.info() stampa su stdout, quindi catturiamo l'output
                from io import StringIO
                buffer = StringIO()
                df.info(buf=buffer, verbose=True)
                info_str = buffer.getvalue()
                st.text_area("Output df.info():", value=info_str, height=300)
            if st.checkbox("Descriptive statistics", key=f"desc_{csv_file}"): st.dataframe(df.describe(include='all'))
            if st.checkbox("Data types per column", key=f"dtype_{csv_file}"): st.write(df.dtypes)
        except FileNotFoundError:
            st.error(f"File {csv_file} non trovato al percorso: {file_path}")
        except pd.errors.EmptyDataError:
            st.error(f"Il file {csv_file} è vuoto o non contiene dati.")
        except Exception as e:
            st.error(f"Errore caricando o processando {csv_file}: {e}")
            traceback.print_exc()


def settings_page_func():
    display_main_header()
    st.title("Settings")
    st.markdown("Configure options.")

    if st.button("Clean Directory Temp Agents Output "):
        feedback = clear_temp_outputs_on_startup()
        for msg in feedback:
            if "ERROR" in msg or "WARNING" in msg: st.warning(msg)
            else: st.success(msg)

    st.markdown("---")
    st.subheader("Information")
    st.info("""
    - **Project:**Multi-Agent System for Data Analysis
    - **Course:** Machine Learning, LUISS Guido Carli
    - **Collaboration:** Reply
    """)

# --- Sidebar per Navigazione e Log Pulizia ---
with st.sidebar:
    st.markdown("## Menu")

    cleanup_msgs_sidebar = st.session_state.get('cleanup_feedback_messages_for_sidebar', [])
    if cleanup_msgs_sidebar:
        with st.expander("Log di Avvio", expanded=False):
            for msg in cleanup_msgs_sidebar:
                if "FATAL ERROR" in msg: st.error(msg)
                elif "ERROR" in msg or "ATTENZIONE" in msg or "WARNING" in msg: st.warning(msg)
                else: st.info(msg)
        st.session_state.cleanup_feedback_messages_for_sidebar = []

    page_options = {
        "💬 Chat with agent": chat_agent_page_func,
        "📊 Explore Dataset": explore_datasets_page_func,
        "⚙️ Setting": settings_page_func
    }
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = list(page_options.keys())[0]

    st.session_state.selected_page = st.radio(
        "Select Page:",
        options=list(page_options.keys()),
        index=list(page_options.keys()).index(st.session_state.selected_page),
        label_visibility='collapsed'
    )
    st.markdown("---")
    st.caption("Powered by Reply & LUISS AI Agents")

# --- Esegui la Funzione della Pagina Selezionata ---
if st.session_state.selected_page in page_options:
    page_function_to_call = page_options[st.session_state.selected_page]
    page_function_to_call()
else:
    st.error("Pagina not found!")
    chat_agent_page_func()