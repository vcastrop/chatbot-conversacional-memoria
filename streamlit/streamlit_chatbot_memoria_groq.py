import os
import streamlit as st
from groq import Groq

st.set_page_config(page_title="💬 Chatbot con Memoria (Groq + Streamlit)", page_icon="💬", layout="centered")

st.title("💬 Chatbot Conversacional con Memoria")
st.caption("Stateful app con `st.session_state` + API de Groq (`llama3-8b-8192`)")

# ---------- Utilidades ----------
DEFAULT_SYSTEM = (
    "Eres un asistente útil y conciso. Responde en español por defecto. "
    "Si el usuario pide código, respóndelo en bloques bien comentados."
)

def get_api_key() -> str | None:
    # Prioridad a st.secrets (recomendado), luego variable de entorno.
    key = st.secrets.get("GROQ_API_KEY", None)
    if not key:
        key = os.getenv("GROQ_API_KEY")
    return key

@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)

def ensure_memory(system_prompt: str):
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
    if "model_name" not in st.session_state:
        st.session_state.model_name = "llama3-8b-8192"

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Opciones")
    api_key = get_api_key()
    if not api_key:
        st.error("No encuentro la clave. Añádela en `.streamlit/secrets.toml` como `GROQ_API_KEY` o exporta la variable de entorno `GROQ_API_KEY`.")
    model_name = st.selectbox(
        "Modelo (Groq)",
        options=["llama3-8b-8192"],
        index=0,
        help="Modelo recomendado para chat general con buen contexto.",
    )
    temperature = st.slider("Creatividad (temperature)", 0.0, 1.5, 0.3, 0.1)
    max_ctx = st.number_input("Límites de memoria (últimos mensajes)", min_value=4, max_value=64, value=24, step=2,
                              help="Para evitar prompts muy largos, solo enviamos los últimos N turnos (incl. system).")
    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM, height=120)

    col1, col2 = st.columns(2)
    with col1:
        clear_btn = st.button("🧹 Borrar memoria", help="Reinicia el historial de la sesión.")
    with col2:
        export_btn = st.button("⬇️ Exportar chat", help="Descarga el historial como texto.")

# ---------- Estado ----------
ensure_memory(system_prompt)
if clear_btn:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
    st.success("Memoria borrada.")

# Sincroniza el modelo si cambia
if model_name != st.session_state.model_name:
    st.session_state.model_name = model_name

# ---------- Mostrar historial ----------
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if export_btn:
    # Exportar como texto simple
    lines = []
    for m in st.session_state.messages:
        if m["role"] == "system":
            continue
        lines.append(f"{m['role'].upper()}: {m['content']}")
    st.download_button("Descargar conversación", "\n\n".join(lines), file_name="chat_groq.txt", mime="text/plain")

# ---------- Chat input ----------
user_input = st.chat_input("Escribe tu mensaje…")
if user_input and api_key:
    # 1) Añadimos el mensaje del usuario a la memoria
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Preparamos el contexto (últimos N mensajes)
    context = st.session_state.messages[-int(max_ctx):]

    # 3) Llamamos a la API de Groq
    try:
        client = get_client(api_key)
        completion = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=context,
            temperature=float(temperature),
        )
        assistant_reply = completion.choices[0].message.content
    except Exception as e:
        assistant_reply = f"Lo siento, ocurrió un error al llamar a Groq: `{e}`"

    # 4) Añadimos la respuesta a memoria y la mostramos
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

st.markdown("---")
st.markdown(
    "### Cómo funciona\n"
    "- **Memoria de sesión**: `st.session_state` guarda el historial mientras la app está abierta.\n"
    "- **Seguridad**: La clave va en `st.secrets` y **no** en el código.\n"
    "- **Estado**: En cada turno se envían los mensajes recientes al modelo (`llama3-8b-8192`).\n"
    "- **UI de chat**: Se usa `st.chat_message` y `st.chat_input`."
)
