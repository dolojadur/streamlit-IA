# streamlit-IA

Proyecto Streamlit: Asistente Médico con Neo4j + Ollama

## Configuración de Credenciales

### Para desarrollo local:

1) Copia el archivo `.env.example` a `.env`:
```bash
cp .env.example .env
```

2) Edita `.env` con tus credenciales reales de Neo4j:
```
NEO4J_URL=neo4j+s://tu-instancia.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=tu-password-real
NEO4J_DATABASE=neo4j
LLM_NAME=gemma3:4b
```

3) Instala dependencias:
```bash
pip install -r requirements.txt
```

4) Comprueba la conexión a Neo4j:
```bash
python check_neo4j.py
```

### Para desplegar en Streamlit Cloud:

1) Asegurate de incluir `requirements.txt` en la raíz (ya creado).

2) Añadí los secrets (Settings → Secrets) en la app de Streamlit Cloud:

```
NEO4J_URL = "neo4j+s://tu-instancia.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu-password-real"
NEO4J_DATABASE = "neo4j"
LLM_NAME = "gemma3:4b"
```

3) Si usás una instancia Neo4j local (localhost), recordá que Streamlit Cloud NO podrá conectarse a ella; debés usar una instancia con IP/host pública o Neo4j Aura.

4) Si tenés problemas con dependencias que no existan en PyPI (por ejemplo adaptadores no publicados), agregá instrucciones para instalarlas desde GitHub o considerá usar otro LLM (OpenAI) para despliegue.

## Seguridad

⚠️ **IMPORTANTE**: Nunca incluyas credenciales directamente en el código. Este proyecto ahora usa variables de entorno y Streamlit secrets para mantener las credenciales seguras.
