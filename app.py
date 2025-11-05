# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Configuración de página (antes de cualquier st.*) ---
st.set_page_config(page_title="Blackjack ML – Jugar", page_icon="🃏", layout="wide")

# --- Estilos: tipografía Poppins + fondo claro + ocultar sidebar ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }
:root { --bg: #f6f7fb; }
section.main > div { padding-top: 1.2rem; }
.block-container { padding-top: 1rem; }
.stApp { background: var(--bg); }
[data-testid="stSidebar"] { display: none; }  /* oculta sidebar */
h1, h2, h3 { letter-spacing: .2px; }
.card {
  background: #ffffff; border-radius: 16px; padding: 1.1rem 1.25rem;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}
.btn-row button { width: 100%; }
.grid { display: grid; gap: 1rem; }
.table { background: #0c5e2c; color: #fff; border-radius: 18px; padding: 18px; }
.badge { background:#ffffff22; padding: 2px 8px; border-radius: 8px; margin-left: 8px; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Utilidades para encontrar columnas esperadas ----------
def _find_column_transformer(pipe: Pipeline) -> ColumnTransformer | None:
    if isinstance(pipe, ColumnTransformer):
        return pipe
    if hasattr(pipe, "named_steps"):
        for _, step in pipe.named_steps.items():
            if isinstance(step, ColumnTransformer):
                return step
            if isinstance(step, Pipeline):
                inner = _find_column_transformer(step)
                if inner is not None:
                    return inner
    return None

def expected_columns_from_ct(ct: ColumnTransformer) -> list[str]:
    cols = []
    for name, trans, cols_spec in getattr(ct, "transformers_", []):
        if cols_spec == "drop" or cols_spec is None:
            continue
        if isinstance(cols_spec, (list, tuple)):
            cols.extend([c for c in cols_spec if isinstance(c, str)])
    seen, unique_cols = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c); unique_cols.append(c)
    return unique_cols

def ensure_expected_columns(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in expected:
        if col not in df.columns:
            if col in ("player_cards", "dealer_cards"):
                df[col] = ""
            elif col in ("step", "round_id", "hand_number"):
                df[col] = 1
            elif col in ("game_id",):
                df[col] = 0
            elif col in ("bet_mode", "strategy_used"):
                df[col] = "unknown"
            else:
                df[col] = np.nan
    return df

# ---------- Clases usadas en el pipeline (necesarias para deserializar joblib) ----------
from sklearn.base import BaseEstimator, TransformerMixin

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []
    def fit(self, X, y=None): return self
    def transform(self, X): return X.drop(columns=self.columns_to_drop)

class BlackjackFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_ = X.copy()
        def hand_value(cards):
            card_list = [c.strip().upper() for c in str(cards).split(",") if c.strip()]
            values = []
            for c in card_list:
                if c in ["J","Q","K"]: values.append(10)
                elif c == "A": values.append(11)
                else:
                    try: values.append(int(c))
                    except ValueError: values.append(0)
            total = sum(values); aces = card_list.count("A")
            while total > 21 and aces > 0:
                total -= 10; aces -= 1
            return total
        X_["player_total"] = X_["player_cards"].apply(hand_value)
        X_["player_aces"]  = X_["player_cards"].apply(lambda s: str(s).upper().split(",").count("A"))
        def dealer_value(cards):
            first = str(cards).split(",")[0].strip().upper()
            if first in ["J","Q","K"]: return 10
            if first == "A": return 11
            try: return int(first)
            except ValueError: return 0
        X_["dealer_visible"] = X_["dealer_cards"].apply(dealer_value)
        return X_

# ---------- Carga del modelo ----------
@st.cache_resource
def load_model():
    m = joblib.load("models/blackjack_action_model.joblib")
    ct = _find_column_transformer(m)
    expected = expected_columns_from_ct(ct) if ct is not None else []
    return m, expected

model, expected_cols = load_model()

# ---------- Recomendación ----------
ACTIONS = ["hit","stand","double","split"]

def recommend_action(model, player_cards, dealer_cards, step=1, extra_cols=None):
    row = {
        "player_cards": player_cards, "dealer_cards": dealer_cards, "step": step,
        "game_id": 1, "round_id": 1, "hand_number": 1, "bet_mode": "flat", "strategy_used": "unknown",
    }
    if extra_cols: row.update(extra_cols)
    X = pd.DataFrame([row])
    try:
        if expected_cols: X = ensure_expected_columns(X, expected_cols)
        return model.predict(X)[0]
    except ValueError as e:
        st.error("Faltan columnas para el pipeline. Ajustá ensure_expected_columns().")
        st.code(str(e))
        raise

# ---------- Motor sencillo de Blackjack ----------
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

def card_to_str(card_tuple): return card_tuple[0]
def add_card(list_str, card_tuple):
    s = card_to_str(card_tuple)
    return (list_str + ", " + s) if list_str.strip() else s

def hand_value(cards_str):
    cards = [c.strip().upper() for c in cards_str.split(",") if c.strip()]
    vals = []
    for c in cards:
        if c in ["J","Q","K"]: vals.append(10)
        elif c == "A": vals.append(11)
        else:
            try: vals.append(int(c))
            except: vals.append(0)
    total = sum(vals); aces = cards.count("A")
    while total > 21 and aces > 0:
        total -= 10; aces -= 1
    return total

# ---------- Estado ----------
if "shoe" not in st.session_state: st.session_state.shoe = new_shoe(4)
if "num_decks" not in st.session_state: st.session_state.num_decks = 4
if "player_cards" not in st.session_state: st.session_state.player_cards = ""
if "dealer_cards" not in st.session_state: st.session_state.dealer_cards = ""
if "step" not in st.session_state: st.session_state.step = 1
if "round_over" not in st.session_state: st.session_state.round_over = True
if "last_rec" not in st.session_state: st.session_state.last_rec = "hit"

# ---------- UI ----------
st.markdown("<h1>🃏 Blackjack ML — Jugar contra el modelo</h1>", unsafe_allow_html=True)
with st.container():
    c_top = st.columns([1,1,2,2,2])
    with c_top[0]:
        st.number_input("N° de mazos", min_value=1, max_value=8, step=1,
                        value=st.session_state.num_decks, key="num_decks")
    with c_top[1]:
        if st.button("🂠 Repartir nueva mano"):
            st.session_state.shoe = new_shoe(st.session_state.num_decks)
            st.session_state.player_cards = ""
            st.session_state.dealer_cards = ""
            st.session_state.step = 1
            st.session_state.round_over = False
            for _ in range(2):
                st.session_state.player_cards = add_card(st.session_state.player_cards, st.session_state.shoe.pop())
                st.session_state.dealer_cards = add_card(st.session_state.dealer_cards, st.session_state.shoe.pop())

# Mesa simple (foco en “jugar”)
st.markdown('<div class="table">', unsafe_allow_html=True)
st.markdown("### Dealer <span class='badge'>(segundo naipe oculto hasta plantarse)</span>", unsafe_allow_html=True)
if st.session_state.dealer_cards:
    dealer_show = st.session_state.dealer_cards.split(",")[0]
    st.write(f"**Visible:** {dealer_show}")
else:
    st.write("—")

st.markdown("---")
st.markdown("### Jugador")
if st.session_state.player_cards:
    st.write(f"Cartas: {st.session_state.player_cards}  |  **Total**: {hand_value(st.session_state.player_cards)}")
else:
    st.write("—")
st.markdown('</div>', unsafe_allow_html=True)

# Controles de juego
colA, colB, colC, colD = st.columns([1,1,2,2])
with colA:
    if st.button("🤖 Recomendar"):
        if not st.session_state.round_over:
            rec = recommend_action(model, st.session_state.player_cards, st.session_state.dealer_cards, step=st.session_state.step)
            st.session_state.last_rec = rec
            st.toast(f"Modelo sugiere: {rec.upper()}")
        else:
            st.info("Repartí una mano para consultar al modelo.")

with colB:
    accion = st.selectbox("Acción a aplicar", ACTIONS, index=ACTIONS.index(st.session_state.last_rec))

with colC:
    if st.button("Aplicar acción"):
        if st.session_state.round_over:
            st.info("Primero repartí una mano."); st.stop()
        if accion in ("hit","double"):
            st.session_state.player_cards = add_card(st.session_state.player_cards, st.session_state.shoe.pop())
            st.session_state.step += 1
            if hand_value(st.session_state.player_cards) > 21:
                st.error("¡Te pasaste! Pierdes la mano.")
                st.session_state.round_over = True
        elif accion == "stand":
            # Juega el dealer hasta 17+
            def dealer_total(s): return hand_value(s)
            while dealer_total(st.session_state.dealer_cards) < 17 and len(st.session_state.shoe) > 0:
                st.session_state.dealer_cards = add_card(st.session_state.dealer_cards, st.session_state.shoe.pop())
            p = hand_value(st.session_state.player_cards)
            d = hand_value(st.session_state.dealer_cards)
            st.write(f"Dealer: {st.session_state.dealer_cards} (total {d})")
            if d > 21 or p > d: st.success("¡Ganaste!")
            elif p < d:        st.error("Perdiste 😢")
            else:              st.info("Empate (push).")
            st.session_state.round_over = True
        elif accion == "split":
            st.info("Split no implementado en esta demo.")

with colD:
    if st.button("🔄 Nueva mano"):
        st.session_state.round_over = True
        st.session_state.player_cards = ""
        st.session_state.dealer_cards = ""
        st.session_state.step = 1

# Panel de estado
st.markdown('<div class="card">', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.metric("Paso", st.session_state.step)
c2.write(f"**Jugador:** {st.session_state.player_cards or '-'}")
c3.write(f"**Dealer:** {st.session_state.dealer_cards or '-'}")
if st.session_state.last_rec:
    st.caption(f"Última recomendación del modelo: **{st.session_state.last_rec.upper()}**")
st.markdown('</div>', unsafe_allow_html=True)
