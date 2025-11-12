# streamlit-IA

Proyecto Streamlit: Asistente Médico con Neo4j + Ollama

Instrucciones rápidas para desplegar en Streamlit Cloud:

1) Asegurate de incluir `requirements.txt` en la raíz (ya creado).

2) Añadí los secrets (Settings → Secrets) en la app de Streamlit Cloud:

- NEO4J_URL = "neo4j+s://<tu-host>"
- NEO4J_USER = "<usuario>"
- NEO4J_PASSWORD = "<password>"
- LLM_NAME = "<nombre_modelo>"  # opcional

3) Comprobá con el health-check localmente si querés:

```powershell
pip install -r requirements.txt
python .\check_neo4j.py
```

4) Si usás una instancia Neo4j local (localhost), recordá que Streamlit Cloud NO podrá conectarse a ella; debés usar una instancia con IP/host pública o Neo4j Aura.

5) Si tenés problemas con dependencias que no existan en PyPI (por ejemplo adaptadores no publicados), agregá instrucciones para instalarlas desde GitHub o considerá usar otro LLM (OpenAI) para despliegue.

Si querés, hago yo los cambios para que las credenciales se lean desde `st.secrets` explícitamente en el app y te dejo instrucciones paso a paso para configurar la app en Streamlit Cloud.
