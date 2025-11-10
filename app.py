import streamlit as st
from assistant import answer  # reutiliza toda tu lógica

st.set_page_config(page_title="Asistente Médico Inteligente", page_icon="🩺")
st.title("🩺 Asistente Médico con Neo4j + Ollama")
st.write("Ingresá tus síntomas o consultas médicas para obtener un diagnóstico sugerido.")

question = st.text_input("💬 Escribí tu pregunta:")

if st.button("Consultar"):
    if question.strip():
        with st.spinner("Generando respuesta..."):
            try:
                cypher_used, ans = answer(question)
            except Exception as e:
                cypher_used, ans = "", f"Error al procesar la consulta: {e}"
        st.markdown("### 🧠 Respuesta")
        st.write(ans or "Sin respuesta generada.")
        st.markdown("### 🧩 Cypher generado")
        st.code(cypher_used or "(No disponible)", language="cypher")
    else:
        st.warning("Por favor escribí una pregunta antes de consultar.")
