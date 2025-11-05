# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------
# Config básica
# -------------------------------
st.set_page_config(page_title="Blackjack ML – Jugar", page_icon="🃏", layout="wide")

# Tipografía + estilos (oscuro + paño + cartas)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
:root, .stApp, .main, body { background-color: #0f1116 !important; color: #e8e8ea !important; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }

.h1title { display:flex; gap:.6rem; align-items:center; }
.h1title img { width:32px; height:32px; }
.panel { background:#141823; border:1px solid #242b3a; border-radius:18px; padding:1.2rem; }
.infochip { font-size:.9rem; opacity:.9; }

.tablefelt {
  margin-top: .6rem;
  background: radial-gradient(1200px 600px at 50% 60%, #2c6b3f 0%, #19492c 55%, #133a24 100%);
  border: 1px solid rgba(0,0,0,.2);
  box-shadow: 0 12px 30px rgba(0,0,0,.35) inset, 0 8px 24px rgba(0,0,0,.35);
  border-radius: 22px;
  padding: 22px;
}

.row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.label { width:110px; color:#d7ecd9; opacity:.9; }
.cards { display:flex; gap:10px; flex-wrap:wrap; }

.cardface {
  width: 56px; height: 78px;
  background: #fff;
  border-radius: 10px;
  border: 1px solid #c9c9c9;
  display:flex; align-items:center; justify-content:center;
  font-weight:600; font-size:18px;
  box-shadow: 0 2px 8px rgba(0,0,0,.25);
  color:#111;
}
.cardface.red { color:#b3122f; }
.cardface.back {
  background: repeating-linear-gradient(45deg, #1b2340, #1b2340 6px, #2b3a74 6px, #2b3a74 12px);
  border-color: #253264;
}

.controls .stButton>button, .stSelectbox [data-baseweb="select"] {
  border-radius: 12px; border: 1px solid #2a2f3a; box-shadow: none;
}
hr { border-color:#2a2f3a; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Helpers de pipeline
# -------------------------------
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
    for _, _, cols_spec in getattr(ct, "transformers_", []):
        if cols_spec in ("drop", None): continue
        if isinstance(cols_spec, (list, tuple)):
            cols.extend([c for c in cols_spec if isinstance(c, str)])
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def ensure_expected_columns(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in expected:
        if col not in df.columns:
            if col in ("player_cards", "dealer_cards"): df[col] = ""
            elif col in ("step", "round_id", "hand_number"): df[col] = 1
            elif col in ("game_id",): df[col] = 0
            elif col in ("bet_mode", "strategy_used"): df[col] = "unknown"
            else: df[col] = np.nan
    return df

# -------------------------------
# Transformadores usados
# -------------------------------
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None): self.columns_to_drop = columns_to_drop or []
    def fit(self, X, y=None): return self
    def transform(self, X): return X.drop(columns=self.columns_to_drop)

class BlackjackFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_ = X.copy()
        def hand_value(cards):
            card_list = [c.strip().upper() for c in str(cards).split(",") if c.strip()]
            vals = []
            for c in card_list:
                if c in ["J","Q","K"]: vals.append(10)
                elif c == "A": vals.append(11)
                else:
                    try: vals.append(int(c))
                    except: vals.append(0)
            total = sum(vals); aces = card_list.count("A")
            while total > 21 and aces > 0:
                total -= 10; aces -= 1
            return total
        X_["player_total"] = X_["player_cards"].apply(hand_value)
        X_["player_aces"]  = X_["player_cards"].apply(lambda s: str(s).upper().split(",").count("A"))
        def dealer_value(cards):
            first = str(cards).split(",")[0].strip().upper()
            if not first: return 0
            if first in ["J","Q","K"]: return 10
            if first == "A": return 11
            try: return int(first)
            except: return 0
        X_["dealer_visible"] = X_["dealer_cards"].apply(dealer_value)
        return X_

# -------------------------------
# Modelo
# -------------------------------
@st.cache_resource
def load_model():
    m = joblib.load("models/blackjack_action_model.joblib")
    ct = _find_column_transformer(m)
    expected = expected_columns_from_ct(ct) if ct is not None else []
    return m, expected

model, expected_cols = load_model()

# -------------------------------
# Juego
# -------------------------------
ACTIONS = ["hit", "stand", "double", "split"]
RANKS  = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS  = ["♠","♥","♦","♣"]

def new_shoe(num_decks=4):
    shoe = []
    for _ in range(num_decks):
        for r in RANKS:
            for s in SUITS:
                shoe.append((r,s))
    random.shuffle(shoe); return shoe

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

def recommend_action(model, player_cards, dealer_cards, step=1, extra_cols=None):
    row = {
        "player_cards": player_cards, "dealer_cards": dealer_cards, "step": step,
        "game_id": 1, "round_id": 1, "hand_number": 1, "bet_mode": "flat", "strategy_used": "unknown",
    }
    if extra_cols: row.update(extra_cols)
    X = pd.DataFrame([row])
    if expected_cols: X = ensure_expected_columns(X, expected_cols)
    return model.predict(X)[0]

def safe_first_visible(dealer_cards: str) -> str:
    cards = [c.strip() for c in dealer_cards.split(",") if c.strip()]
    return cards[0] if cards else "-"

def action_index_safe(val: str) -> int:
    try: return ACTIONS.index(val)
    except: return 0

# -------------------------------
# Estado
# -------------------------------
if "num_decks"    not in st.session_state: st.session_state.num_decks = 4
if "shoe"         not in st.session_state: st.session_state.shoe = new_shoe(st.session_state.num_decks)
if "player_cards" not in st.session_state: st.session_state.player_cards = ""
if "dealer_cards" not in st.session_state: st.session_state.dealer_cards = ""
if "step"         not in st.session_state: st.session_state.step = 1
if "round_over"   not in st.session_state: st.session_state.round_over = True
if "last_rec"     not in st.session_state: st.session_state.last_rec = ACTIONS[0]  # "hit"

# -------------------------------
# UI – un solo panel
# -------------------------------
st.markdown('<div class="h1title"><img src="https://em-content.zobj.net/source/microsoft-teams/337/joker_1f0cf.png"/>'
            '<h1>Blackjack ML – Jugar vs el modelo</h1></div>', unsafe_allow_html=True)

top_col1, top_col2 = st.columns([1,1])
with top_col1:
    num_decks = st.selectbox("N° de mazos", [1,2,4,6,8], index=[1,2,4,6,8].index(st.session_state.num_decks))
with top_col2:
    if num_decks != st.session_state.num_decks:
        st.session_state.num_decks = num_decks
        st.session_state.shoe = new_shoe(num_decks)
        st.session_state.player_cards = ""
        st.session_state.dealer_cards = ""
        st.session_state.step = 1
        st.session_state.round_over = True

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("🎮 Simulador de mano")

# Mesa / paño
st.markdown('<div class="tablefelt">', unsafe_allow_html=True)

# Dealer row
dealer_row = '<div class="row"><div class="label">Dealer</div><div class="cards">{cards}</div></div>'
# Player row
player_row = '<div class="row" style="margin-top:10px"><div class="label">Jugador</div><div class="cards">{cards}</div></div>'

def render_cards(cards_str: str, hide_second: bool=False) -> str:
    cards = [c.strip().upper() for c in cards_str.split(",") if c.strip()]
    html = []
    for i, c in enumerate(cards):
        red = "red" if c in ["♥","♦"] else ""  # (valor simbólico; no mostramos palos reales)
        klass = "cardface " + red
        if hide_second and i == 1: klass = "cardface back"
        html.append(f'<div class="{klass}">{"" if hide_second and i==1 else c}</div>')
    if not html: html.append('<div class="cardface back"></div>')
    return "".join(html)

# Cartas visibles en mesa
dealer_cards_html = render_cards(st.session_state.dealer_cards, hide_second=not st.session_state.round_over)
player_cards_html = render_cards(st.session_state.player_cards, hide_second=False)

st.markdown(dealer_row.format(cards=dealer_cards_html), unsafe_allow_html=True)
st.markdown(player_row.format(cards=player_cards_html), unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /tablefelt

st.markdown('<div class="controls">', unsafe_allow_html=True)

left, right = st.columns([1.25, 0.75])

with left:
    if st.session_state.round_over:
        st.caption("Presioná **Repartir** para iniciar una mano.")
        if st.button("🂠 Repartir", use_container_width=True):
            st.session_state.shoe = new_shoe(st.session_state.num_decks)
            st.session_state.player_cards = ""
            st.session_state.dealer_cards = ""
            st.session_state.step = 1
            st.session_state.round_over = False
            for _ in range(2):
                st.session_state.player_cards = add_card(st.session_state.player_cards, st.session_state.shoe.pop())
                st.session_state.dealer_cards = add_card(st.session_state.dealer_cards, st.session_state.shoe.pop())
    else:
        st.write(f"**Tus cartas:** {st.session_state.player_cards or '-'}  | **Total:** {hand_value(st.session_state.player_cards) if st.session_state.player_cards else '-'}")
        st.write(f"**Dealer (visible):** {safe_first_visible(st.session_state.dealer_cards)}")

        c1, c2, c3 = st.columns(3)
        if c1.button("🤖 Recomendar acción", use_container_width=True):
            rec = recommend_action(model, st.session_state.player_cards, st.session_state.dealer_cards, step=st.session_state.step)
            st.session_state.last_rec = rec
            st.toast(f"Modelo sugiere: {rec.upper()}")

        rec_to_apply = st.selectbox("Acción a aplicar", ACTIONS, index=action_index_safe(st.session_state.last_rec))

        if c2.button("Aplicar acción", use_container_width=True):
            if rec_to_apply in ("hit", "double"):
                st.session_state.player_cards = add_card(st.session_state.player_cards, st.session_state.shoe.pop())
                st.session_state.step += 1
                if hand_value(st.session_state.player_cards) > 21:
                    st.error("¡Te pasaste! Pierdes la mano.")
                    st.session_state.round_over = True
            elif rec_to_apply == "stand":
                # Descubrir carta y jugar dealer hasta 17+
                while hand_value(st.session_state.dealer_cards) < 17 and len(st.session_state.shoe) > 0:
                    st.session_state.dealer_cards = add_card(st.session_state.dealer_cards, st.session_state.shoe.pop())
                p = hand_value(st.session_state.player_cards)
                d = hand_value(st.session_state.dealer_cards)
                st.write(f"Dealer: {st.session_state.dealer_cards} (total {d})")
                if d > 21 or p > d: st.success("¡Ganaste!")
                elif p < d:         st.error("Perdiste 😢")
                else:               st.info("Empate (push).")
                st.session_state.round_over = True
            elif rec_to_apply == "split":
                st.info("Split no está implementado en esta demo.")

        if c3.button("🔄 Nueva mano", use_container_width=True):
            st.session_state.round_over = True

with right:
    st.caption("Estado de la ronda")
    st.metric("Paso", st.session_state.step)
    st.write(f"**Jugador:** {st.session_state.player_cards or '-'}")
    st.write(f"**Total jugador:** {hand_value(st.session_state.player_cards) if st.session_state.player_cards else '-'}")
    st.write(f"**Dealer:** {st.session_state.dealer_cards or '-'}")
    st.write(f"**Visible dealer:** {safe_first_visible(st.session_state.dealer_cards)}")
    st.write(f"**Última recomendación:** {st.session_state.last_rec.upper() if st.session_state.last_rec else '-'}")

st.markdown('</div>', unsafe_allow_html=True)  # /controls
st.markdown('</div>', unsafe_allow_html=True)  # /panel

st.caption("El modelo calcula internamente *player_total*, *player_aces* y *dealer_visible*.")
