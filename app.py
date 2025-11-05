# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# ---------- Tipografía Poppins vía CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
html, body, [class*="css"]  {
  font-family: 'Poppins', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Blackjack ML – Demo", page_icon="🃏", layout="wide")

# ---------- Transformadores que usaste en el notebook ----------
from sklearn.base import BaseEstimator, TransformerMixin

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)

class BlackjackFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_ = X.copy()
        # Valor óptimo de la mano del jugador (A=1 u 11)
        def hand_value(cards):
            card_list = [c.strip().upper() for c in str(cards).split(",") if c.strip()]
            values = []
            for c in card_list:
                if c in ["J","Q","K"]:
                    values.append(10)
                elif c == "A":
                    values.append(11)
                else:
                    try:
                        values.append(int(c))
                    except ValueError:
                        values.append(0)
            total = sum(values)
            aces = card_list.count("A")
            while total > 21 and aces > 0:
                total -= 10
                aces -= 1
            return total

        X_["player_total"] = X_["player_cards"].apply(hand_value)
        X_["player_aces"]  = X_["player_cards"].apply(lambda s: str(s).upper().split(",").count("A"))

        def dealer_value(cards):
            first = str(cards).split(",")[0].strip().upper()
            if first in ["J","Q","K"]:
                return 10
            elif first == "A":
                return 11
            else:
                try:
                    return int(first)
                except ValueError:
                    return 0
        X_["dealer_visible"] = X_["dealer_cards"].apply(dealer_value)
        return X_

# ---------- Carga del modelo ----------
@st.cache_resource
def load_model():
    return joblib.load("models/blackjack_action_model.joblib")

model = load_model()

# ---------- Utils ----------
ACTIONS = ["hit","stand","double","split"]

def recommend_action(model, player_cards, dealer_cards, step=1, extra_cols=None):
    """
    Construye un mini DataFrame con las columnas mínimas esperadas por tu pipeline:
      - 'player_cards'  (str CSV, ej: "A, 6")
      - 'dealer_cards'  (str CSV, ej: "K, 8")  -> tu extractor mira la primera
      - 'step'
      - + columnas que luego DropColumns eliminará (si faltan, igual no molesta)
    """
    row = {
        "player_cards": player_cards,
        "dealer_cards": dealer_cards,
        "step": step,
        # columnas que suelen estar pero DropColumns tirará:
        "game_id": 1,
        "round_id": 1,
        "hand_number": 1,
        "bet_mode": "flat",
        "strategy_used": "unknown",
    }
    if extra_cols:
        row.update(extra_cols)

    X = pd.DataFrame([row])
    pred = model.predict(X)[0]
    return pred

# Sencillo motor de Blackjack para la pestaña "Jugar"
import random
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["♠","♥","♦","♣"]

def new_shoe(num_decks=4):
    shoe = []
    for _ in range(num_decks):
        for r in RANKS:
            for s in SUITS:
                shoe.append((r,s))
    random.shuffle(shoe)
    return shoe

def card_to_str(card_tuple):
    return card_tuple[0]

def add_card(list_str, card_tuple):
    s = card_to_str(card_tuple)
    return (list_str + ", " + s) if list_str.strip() else s

def hand_value(cards_str):
    # igual a lo del extractor para coherencia
    cards = [c.strip().upper() for c in cards_str.split(",") if c.strip()]
    vals = []
    for c in cards:
        if c in ["J","Q","K"]:
            vals.append(10)
        elif c == "A":
            vals.append(11)
        else:
            try: vals.append(int(c))
            except: vals.append(0)
    total = sum(vals)
    aces = cards.count("A")
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total

# ---------- Sidebar ----------
st.sidebar.title("🃏 Blackjack ML – Demo")
st.sidebar.write("Explorá los datos, mirá resultados del modelo y jugá contra la política aprendida.")
num_decks = st.sidebar.selectbox("N° de mazos", [1,2,4,6,8], index=2)

# ---------- State para la pestaña de juego ----------
if "shoe" not in st.session_state:
    st.session_state.shoe = new_shoe(num_decks)
if "player_cards" not in st.session_state:
    st.session_state.player_cards = ""
if "dealer_cards" not in st.session_state:
    st.session_state.dealer_cards = ""
if "step" not in st.session_state:
    st.session_state.step = 1
if "round_over" not in st.session_state:
    st.session_state.round_over = True

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📊 Exploración", "✅ Resultados del modelo", "🎮 Jugar vs el modelo"])

# === Tab 1: Exploración ===
with tab1:
    st.subheader("Exploración del dataset")
    default_file = "data/Simulacion_BJ.csv"
    uploaded = st.file_uploader("Subí tu CSV (opcional). Si no, intento cargar data/Simulacion_BJ.csv", type=["csv"])
    try:
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_csv(default_file)
        st.write("Vista rápida:")
        st.dataframe(df.head(50), use_container_width=True)

        # Filtros básicos
        cols = st.multiselect("Columnas a mostrar", df.columns.tolist(), default=df.columns.tolist()[:10])
        st.dataframe(df[cols].head(200), use_container_width=True)

        # Conteo de acciones si existe 'action'
        if "action" in df.columns:
            st.write("Distribución de acciones:")
            st.bar_chart(df["action"].value_counts())
    except Exception as e:
        st.info("No se pudo cargar un dataset por defecto. Subí uno arriba.")
        st.caption(str(e))

# === Tab 2: Resultados ===
with tab2:
    st.subheader("Aplicar el modelo y revisar resultados")
    st.write("Podés pasar tu dataset por el pipeline y ver las acciones predichas.")
    data_file = st.file_uploader("Dataset para predecir", type=["csv"], key="pred_uploader")

    if data_file:
        dfx = pd.read_csv(data_file)
        try:
            preds = model.predict(dfx)
            dfx_out = dfx.copy()
            dfx_out["predicted_action"] = preds
            st.success(f"OK. Filas: {len(dfx_out)}")
            st.dataframe(dfx_out.head(200), use_container_width=True)

            # Descarga
            buffer = BytesIO()
            dfx_out.to_csv(buffer, index=False)
            st.download_button("Descargar CSV con predicciones", data=buffer.getvalue(),
                               file_name="predicciones_blackjack.csv", mime="text/csv")
        except Exception as e:
            st.error("El dataset no tiene las columnas esperadas para el pipeline.")
            st.code(str(e))

    st.caption("Recordá que tu pipeline internamente calcula features como player_total, player_aces y dealer_visible, y descarta columnas no usadas.")

# === Tab 3: Jugar vs el modelo ===
with tab3:
    st.subheader("Simulador de mano – consulta al modelo")
    colA, colB = st.columns(2)
    with colA:
        if st.session_state.round_over:
            if st.button("🂠 Repartir"):
                st.session_state.shoe = new_shoe(num_decks)
                st.session_state.player_cards = ""
                st.session_state.dealer_cards = ""
                st.session_state.step = 1
                st.session_state.round_over = False

                # Dar 2 al jugador y 2 al dealer
                for _ in range(2):
                    st.session_state.player_cards = add_card(st.session_state.player_cards, st.session_state.shoe.pop())
                    st.session_state.dealer_cards = add_card(st.session_state.dealer_cards, st.session_state.shoe.pop())
        else:
            st.write("**Tus cartas**:", st.session_state.player_cards, " | Total:", hand_value(st.session_state.player_cards))
            st.write("**Dealer (visible)**:", st.session_state.dealer_cards.split(",")[0])

            c1, c2, c3 = st.columns(3)
            if c1.button("🤖 Recomendar acción"):
                rec = recommend_action(
                    model,
                    st.session_state.player_cards,
                    st.session_state.dealer_cards,
                    step=st.session_state.step
                )
                st.toast(f"Modelo sugiere: {rec.upper()}")
                st.session_state.last_rec = rec

            # Aplicar acción (simple): hit / stand
            rec_to_apply = st.selectbox("Acción a aplicar", ACTIONS, index=ACTIONS.index(st.session_state.get("last_rec","hit")))
            if c2.button("Aplicar acción"):
                if rec_to_apply == "hit" or rec_to_apply == "double":
                    # (si es double no duplicamos apuesta aquí; el objetivo es didáctico)
                    st.session_state.player_cards = add_card(st.session_state.player_cards, st.session_state.shoe.pop())
                    st.session_state.step += 1
                    if hand_value(st.session_state.player_cards) > 21:
                        st.error("¡Te pasaste! Pierdes la mano.")
                        st.session_state.round_over = True
                elif rec_to_apply == "stand":
                    # "Jugar" dealer hasta 17+
                    def dealer_total(cards_str):
                        return hand_value(cards_str)
                    while dealer_total(st.session_state.dealer_cards) < 17 and len(st.session_state.shoe) > 0:
                        st.session_state.dealer_cards = add_card(st.session_state.dealer_cards, st.session_state.shoe.pop())

                    # Resultado
                    p = hand_value(st.session_state.player_cards)
                    d = hand_value(st.session_state.dealer_cards)
                    st.write(f"Dealer: {st.session_state.dealer_cards} (total {d})")
                    if d > 21 or p > d:
                        st.success("¡Ganaste!")
                    elif p < d:
                        st.error("Perdiste 😢")
                    else:
                        st.info("Empate (push).")
                    st.session_state.round_over = True
                elif rec_to_apply == "split":
                    st.info("Split no está implementado en esta versión demo (mecánica de múltiples manos).")

            if c3.button("🔄 Nueva mano"):
                st.session_state.round_over = True

    with colB:
        st.caption("Vista rápida del estado")
        st.metric("Paso", st.session_state.step)
        st.write("Jugador:", st.session_state.player_cards, " | Total:", hand_value(st.session_state.player_cards) if st.session_state.player_cards else "-")
        st.write("Dealer:", st.session_state.dealer_cards, " | Visible:", st.session_state.dealer_cards.split(",")[0] if st.session_state.dealer_cards else "-")
        if "last_rec" in st.session_state:
            st.write("Última recomendación del modelo:", st.session_state.last_rec.upper())
