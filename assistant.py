# assistant.py — LangChain 1.0.x + Neo4j + Ollama (REPL interactivo + sanitizado)
from typing import Any, Dict, List, Tuple, Optional
import re

from langchain_ollama import OllamaLLM
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

# --- Config directa ---
NEO4J_URL = "neo4j+s://c63dbf3f.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "jRcnixjwaho44020Iic2ZK9D2jZXc1hOsrzRCDteJBA"  
NEO4J_DATABASE = "neo4j"
LLM_NAME = "gemma3:4b"                 

# ---------- Conectar al grafo ----------
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

# ---------- LLM ----------
llm = OllamaLLM(model=LLM_NAME)

# ---------- Prompt seguro (solo {question}) ----------
CYPHER_SYSTEM_HINT = CYPHER_SYSTEM_HINT = """
Eres un generador de Cypher para un grafo médico preventivo en español.

Usa exclusivamente estas etiquetas y relaciones:
- Nodos: Sintoma(nombre), Enfermedad(nombre), Regla(codigo, descripcion, score_base), Recomendacion(descripcion)
- Relaciones: 
  (Sintoma)-[:INDICA_POTENCIALMENTE]->(Enfermedad)
  (Regla)-[:CONDICION]->(Sintoma)
  (Regla)-[:SUGIERE]->(Enfermedad)
  (Enfermedad)-[:RECOMIENDA]->(Recomendacion)   // si existen
  (Regla)-[:PROPONE]->(Recomendacion)           // si existen

Reglas:
- SOLO MATCH / OPTIONAL MATCH / WHERE / RETURN (no usar CREATE, MERGE, DELETE, SET, CALL).
- No incluyas propiedades dentro de las relaciones; si necesitás filtrar, usá WHERE.
- Compará strings con toLower(...) y comillas simples.

Ejemplos:
1) Tos + Fiebre -> enfermedad sugerida por una misma regla:
   MATCH (r:Regla)-[:CONDICION]->(s1:Sintoma),
         (r)-[:CONDICION]->(s2:Sintoma),
         (r)-[:SUGIERE]->(e:Enfermedad)
   WHERE toLower(s1.nombre)='tos' AND toLower(s2.nombre)='fiebre'
   RETURN r.codigo AS regla, r.descripcion AS descripcion, e.nombre AS enfermedad

2) Estornudos y congestión nasal -> enfermedades en la intersección:
   MATCH (s1:Sintoma)-[:INDICA_POTENCIALMENTE]->(e:Enfermedad),
         (s2:Sintoma)-[:INDICA_POTENCIALMENTE]->(e)
   WHERE toLower(s1.nombre)='estornudos'
     AND toLower(s2.nombre) IN ['congestion nasal','congestión nasal','rinorrea']
   RETURN DISTINCT e.nombre AS enfermedad

Devolvé SOLO el Cypher correcto que responde a:
Pregunta: {question}
""".strip()

cypher_prompt = PromptTemplate.from_template(CYPHER_SYSTEM_HINT)

# --- Cadena QA sobre grafo (con intermediate steps) ---
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
    # validate_cypher=True,  # opcional
)

def _normalize_query_text(q: str) -> str:
    ql = q.lower()
    # mapeos básicos
    synonyms = {
        "mocos": ["congestion nasal", "congestión nasal", "rinorrea", "secrecion nasal", "secreción nasal"],
        "resfrio": ["resfriado"],
        "resfrío": ["resfriado"],
        "dolor de cabeza": ["cefalea"],
    }
    extra = []
    for k, vs in synonyms.items():
        if k in ql:
            extra.extend(vs)
    if extra:
        return f"{q}\n(Sinónimos a considerar: {', '.join(set(extra))})"
    return q


# ===================== Helpers de robustez =====================

def _strip_code_fences(txt: str) -> str:
    if not txt:
        return txt
    txt = txt.strip()
    # elimina ```...``` (con o sin lenguaje)
    if txt.startswith("```") and txt.endswith("```"):
        txt = txt[3:-3].strip()
    # elimina línea inicial 'cypher'
    if txt.lower().startswith("cypher"):
        # puede venir como 'cypher\nMATCH ...'
        parts = txt.split("\n", 1)
        if len(parts) == 2:
            txt = parts[1].strip()
        else:
            txt = ""
    return txt

def _sanitize_cypher(cypher: str, question: str = "") -> str:
    """Corrige errores típicos del LLM y adapta casos frecuentes."""
    if not cypher:
        return cypher

    def _strip_code_fences(txt: str) -> str:
        txt = txt.strip()
        if txt.startswith("```") and txt.endswith("```"):
            txt = txt[3:-3].strip()
        if txt.lower().startswith("cypher"):
            parts = txt.split("\n", 1)
            txt = parts[1].strip() if len(parts) == 2 else ""
        return txt

    fixed = _strip_code_fences(cypher)

    # Normalizar comillas “inteligentes” y tildes problemáticas
    fixed = (fixed.replace("’", "'")
                  .replace("‘", "'")
                  .replace("“", '"')
                  .replace("”", '"')
                  .replace("congestión nasal", "congestion nasal"))

    # Quitar mapas de propiedades en relaciones: -[:REL {…}]->
    fixed = re.sub(r"(\-\s*\:\s*[A-Za-z_][A-Za-z0-9_]*\s*)\{[^}]*\}", r"\1", fixed)

    qlow = question.lower()

    # ===== Caso 1: "estornudos" + "congestión nasal" -> intersección de ambas hacia la MISMA enfermedad
    if ("estornudos" in qlow and ("congestion" in qlow or "congestión" in qlow)) or \
       ("'estornudos'" in fixed and ("'congestion nasal'" in fixed or "'congestión nasal'" in fixed)):
        fixed = (
            "MATCH (s1:Sintoma)-[:INDICA_POTENCIALMENTE]->(e:Enfermedad), "
            "      (s2:Sintoma)-[:INDICA_POTENCIALMENTE]->(e)\n"
            "WHERE toLower(s1.nombre) = 'estornudos' "
            "  AND toLower(s2.nombre) IN ['congestion nasal','congestión nasal']\n"
            "RETURN DISTINCT e.nombre AS enfermedad"
        )
        return fixed

    # ===== Caso 2: tos + fiebre -> usa s1/s2 y Regla->SUGIERE->Enfermedad
    if (("tos" in qlow and "fiebre" in qlow) or ("'tos'" in fixed and "'fiebre'" in fixed)):
        # Asegurar dos condiciones y enlace a enfermedad
        if "(r)-[:CONDICION]->(s1:Sintoma)" not in fixed or "(r)-[:CONDICION]->(s2:Sintoma)" not in fixed:
            fixed = (
                "MATCH (r:Regla)-[:CONDICION]->(s1:Sintoma), "
                "(r)-[:CONDICION]->(s2:Sintoma), (r)-[:SUGIERE]->(e:Enfermedad)\n"
                "WHERE toLower(s1.nombre) = 'tos' AND toLower(s2.nombre) = 'fiebre'\n"
                "RETURN r.codigo AS regla, r.descripcion AS descripcion, e.nombre AS enfermedad"
            )
        else:
            fixed = fixed.replace(
                "toLower(s1.nombre) = 'tos' AND toLower(s1.nombre) = 'fiebre'",
                "toLower(s1.nombre) = 'tos' AND toLower(s2.nombre) = 'fiebre'"
            )
            if "(r)-[:SUGIERE]->(e:Enfermedad)" not in fixed:
                # inserta el link a enfermedad antes del RETURN
                if "RETURN" in fixed:
                    fixed = fixed.replace(
                        "RETURN",
                        ", (r)-[:SUGIERE]->(e:Enfermedad)\nRETURN"
                    )
                else:
                    fixed += "\nMATCH (r)-[:SUGIERE]->(e:Enfermedad)"
            # RETURN estándar
            if "RETURN" not in fixed.upper():
                fixed += "\nRETURN r.codigo AS regla, r.descripcion AS descripcion, e.nombre AS enfermedad"
        return fixed
    
    qlow = question.lower()

# Tos + mocos -> intersección usando sinónimos de mocos
    if ("tos" in qlow and ("mocos" in qlow or "secrecion nasal" in qlow or "secreción nasal" in qlow)):
        fixed = (
            "MATCH (s1:Sintoma)-[:INDICA_POTENCIALMENTE]->(e:Enfermedad), "
            "      (s2:Sintoma)-[:INDICA_POTENCIALMENTE]->(e)\n"
            "WHERE toLower(s1.nombre)='tos' "
            "  AND toLower(s2.nombre) IN ['congestion nasal','congestión nasal','rinorrea','mocos','secrecion nasal','secreción nasal']\n"
            "RETURN DISTINCT e.nombre AS enfermedad"
        )
        return fixed

    # ===== Caso 3: el LLM usó e1/e2 pero retorna e.nombre -> armonizar alias
    if "RETURN" in fixed and " e.nombre" in fixed and "e:" not in fixed:
        if "e1:" in fixed:
            fixed = fixed.replace("RETURN e.nombre", "RETURN e1.nombre")
        elif "e2:" in fixed:
            fixed = fixed.replace("RETURN e.nombre", "RETURN e2.nombre")
        # asegurar alias de salida
        if "AS enfermedad" not in fixed:
            fixed = fixed.replace("RETURN ", "RETURN ").rstrip(";")
            if "RETURN e1.nombre" in fixed:
                fixed = fixed.replace("RETURN e1.nombre", "RETURN e1.nombre AS enfermedad")
            elif "RETURN e2.nombre" in fixed:
                fixed = fixed.replace("RETURN e2.nombre", "RETURN e2.nombre AS enfermedad")

    # ===== Caso 4: consulta por “reglas para gripe”
    if ("regla" in qlow and "gripe" in qlow) or ("'gripe'" in fixed and ":Regla" in fixed):
        fixed = (
            "MATCH (r:Regla)-[:SUGIERE]->(e:Enfermedad {nombre:'Gripe'})\n"
            "OPTIONAL MATCH (r)-[:CONDICION]->(s:Sintoma)\n"
            "RETURN r.codigo AS regla, r.descripcion AS descripcion, "
            "collect(DISTINCT s.nombre) AS sintomas_condicionados"
        )
        return fixed

    # Asegurar que exista algún RETURN razonable
    if "RETURN" not in fixed.upper():
        if "(e:Enfermedad)" in fixed or "->(e:Enfermedad)" in fixed:
            fixed += "\nRETURN DISTINCT e.nombre AS enfermedad"
        else:
            fixed += "\nRETURN 1 AS ok"

    return fixed.strip()


def _extract_steps(data: Dict[str, Any]) -> Tuple[str | None, Any]:
    """Soporta intermediate_steps como dict o list; devuelve (cypher, context)."""
    cypher = None
    context = None
    steps = data.get("intermediate_steps")

    if isinstance(steps, dict):
        cypher = steps.get("cypher")
        context = steps.get("context")
    elif isinstance(steps, list):
        # Buscar el último dict que tenga 'cypher'
        for item in reversed(steps):
            if isinstance(item, dict) and "cypher" in item:
                cypher = item.get("cypher")
                context = item.get("context")
                break
        # Si no hay dict, buscar un string que parezca consulta
        if cypher is None:
            for item in reversed(steps):
                if isinstance(item, str) and "MATCH" in item:
                    cypher = item
                    break
    return cypher, context

def _llm_generate_cypher(question: str) -> str:
    """Llama directamente al generador de Cypher del chain como fallback."""
    raw = chain.cypher_generation_chain.invoke({"question": question, "query": question})
    text = raw.get("text") if isinstance(raw, dict) and "text" in raw else str(raw)
    return _sanitize_cypher(text, question)

def _run_cypher(cypher: str) -> List[Dict[str, Any]]:
    try:
        return graph.query(cypher) or []
    except Exception:
        return []

def _format_answer_from_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No se encontraron resultados en el grafo para esa consulta."

    # Intentamos inferir diagnóstico y regla desde las columnas que ya devuelve tu Cypher
    # columnas típicas: regla, descripcion, enfermedad
    enf = None
    regla = None
    desc = None
    for r in rows:
        enf = r.get("enfermedad") or enf
        regla = r.get("regla") or regla
        desc = r.get("descripcion") or desc

    # Texto base
    partes = []
    if enf:
        partes.append(f"Posible diagnóstico: **{enf}**.")
    if desc:
        partes.append(f"Regla aplicada: _{desc}_ ({regla})." if regla else f"Regla aplicada: _{desc}_.")

    # Buscar recomendaciones en el grafo (si existieran)
    recomendaciones: List[str] = []
    try:
        if regla:
            q1 = (
                "OPTIONAL MATCH (r:Regla {codigo:$regla})-[:PROPONE]->(rec:Recomendacion) "
                "RETURN collect(distinct rec.descripcion) AS recs"
            )
            rec_rows = graph.query(q1, params={"regla": regla}) or []
            if rec_rows and rec_rows[0].get("recs"):
                recomendaciones.extend([x for x in rec_rows[0]["recs"] if x])

        if enf:
            q2 = (
                "OPTIONAL MATCH (e:Enfermedad {nombre:$enf})-[:RECOMIENDA]->(rec:Recomendacion) "
                "RETURN collect(distinct rec.descripcion) AS recs"
            )
            rec_rows2 = graph.query(q2, params={"enf": enf}) or []
            if rec_rows2 and rec_rows2[0].get("recs"):
                recomendaciones.extend([x for x in rec_rows2[0]["recs"] if x])
    except Exception:
        pass

    # Fallback si el grafo aún no tiene esas relaciones
    if not recomendaciones and enf:
        fallback = {
            "Gripe": [
                "Reposo e hidratación",
                "Antitérmicos/analgésicos de venta libre si no hay contraindicaciones",
                "Consultar clínica médica si fiebre alta >48 h o empeora",
            ],
            "Alergia Estacional": [
                "Evitar alérgenos y ventilar hogares fuera de picos de polen",
                "Lavar fosas nasales con solución salina",
                "Antihistamínicos según indicación profesional",
            ],
        }
        recomendaciones = fallback.get(enf, [
            "Reposo relativo y buena hidratación",
            "Controlar la evolución de los síntomas 24–48 h",
            "Consultar profesional si aparecen signos de alarma",
        ])

    if recomendaciones:
        partes.append("Recomendaciones:\n- " + "\n- ".join(recomendaciones))

    # Si por algún motivo no pudimos inferir nada, caemos al listado genérico
    if not partes:
        keys = rows[0].keys()
        listado = []
        for r in rows[:10]:
            vals = [f"{k}={r.get(k)}" for k in keys]
            listado.append(" • " + ", ".join(vals))
        return "Resultados:\n" + "\n".join(listado)

    return "\n".join(partes)

# ===================== Flujo principal =====================

def answer(question: str) -> Tuple[str, str]:
    """Devuelve (cypher_usado, respuesta)"""
    # 1) Intento con el chain completo
    try:
        user_q = _normalize_query_text(question)
        out = chain.invoke({"query": user_q, "question": user_q})
    except Exception:
        out = None

    cypher, context = (None, None)
    answer_txt = ""

    if isinstance(out, dict):
        cypher, context = _extract_steps(out)
        answer = out.get("result", "")
    else:
        answer = str(out)

    # 2) Fallback si no hay cypher o la respuesta es pobre
    if not cypher or answer.strip().lower() in {"", "i don't know the answer.", "i don't know the answer", "lo siento, no sé la respuesta."}:
        cypher = _llm_generate_cypher(question)
        rows = _run_cypher(cypher)
        if rows:
            answer_txt = _format_answer_from_rows(rows)
        elif not answer_txt:
            answer_txt = "No se encontró información suficiente para responder."
    else:
        # Si vino cypher del chain, intentar ejecutarlo para enriquecer
        rows = _run_cypher(_sanitize_cypher(cypher, question))
        if rows and ("Resultados:" not in answer_txt):
            answer_txt += "\n\n" + _format_answer_from_rows(rows)

    return _sanitize_cypher(cypher or "", question), answer_txt

def cli():
    print("Conectado a Neo4j. Modo interactivo (Ctrl+C para salir)\n")
    try:
        while True:
            q = input("Tu pregunta> ").strip()
            if not q:
                continue
            cypher_used, ans = answer(q)
            print("\n--- Cypher usado ---")
            print(cypher_used or "(no disponible)")
            print("\n=== Respuesta ===")
            print(ans or "(sin respuesta)")
            print("\n" + "=" * 60 + "\n")
    except KeyboardInterrupt:
        print("\nChau!")

# --- Ejecutar ---
if __name__ == "__main__":
    try:
        print("Esquema inferido:\n", graph.schema)
    except Exception:
        pass
    cli()
