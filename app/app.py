from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# PAGE SETUP
# =====================================================================
st.set_page_config(page_title="UFC Fight Lab", page_icon="🥊", layout="wide")

DATA_PATH = Path(__file__).parent.parent / "data" / "ufc-master.csv"

# UFC-style colour palette used by every chart
UFC_RED, UFC_GOLD, UFC_GREY, UFC_BLUE, UFC_DARK_RED = "#D20A0A", "#C9A227", "#9A9AA0", "#4A90D9", "#7A0A0A"
px.defaults.color_discrete_sequence = [UFC_RED, UFC_GOLD, UFC_GREY, UFC_BLUE, UFC_DARK_RED, "#E8E8E8"]
FINISH_COLOURS = {"KO/TKO": UFC_RED, "SUB": UFC_GOLD, "U-DEC": UFC_GREY, "S-DEC": UFC_BLUE}
PERIODS = ["2010-2015", "2016-2020", "2021-2026"]


# =====================================================================
# DATA (loaded once and kept in memory)
# =====================================================================
@st.cache_data
def load_fights():
    """One row per fight (only fights with a winner)."""
    df = pd.read_csv(DATA_PATH)
    df = df[df["Winner"].isin(["Red", "Blue"])].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    has_odds = df["R_odds"].notna() & df["B_odds"].notna() & (df["R_odds"] != df["B_odds"])
    favourite = np.where(df["R_odds"] < df["B_odds"], "Red", "Blue")
    df["favourite_won"] = np.where(has_odds, (favourite == df["Winner"]).astype(float), np.nan)
    return df


@st.cache_data
def build_fighter_history(df):
    """One row per fighter per fight, in date order, with info about the previous fight."""
    rows = []
    for corner, colour, opp in [("R", "Red", "B"), ("B", "Blue", "R")]:
        rows.append(pd.DataFrame({
            "fighter": df[f"{corner}_fighter"],
            "opponent": df[f"{opp}_fighter"],
            "date": df["date"],
            "year": df["year"],
            "weight_class": df["weight_class"],
            "gender": df["gender"],
            "age": df[f"{corner}_age"],
            "opp_age": df[f"{opp}_age"],
            "reach": df[f"{corner}_Reach_cms"],
            "opp_reach": df[f"{opp}_Reach_cms"],
            "won": df["Winner"] == colour,
            "finish": df["finish"],
            "odds": df[f"{corner}_odds"],
            "opp_odds": df[f"{opp}_odds"],
            # fighter profile (career numbers BEFORE this fight)
            "height": df[f"{corner}_Height_cms"],
            "stance": df[f"{corner}_Stance"],
            "wins": df[f"{corner}_wins"],
            "losses": df[f"{corner}_losses"],
            "win_streak": df[f"{corner}_current_win_streak"],
            "lose_streak": df[f"{corner}_current_lose_streak"],
            "wins_ko": df[f"{corner}_win_by_KO/TKO"] + df[f"{corner}_win_by_TKO_Doctor_Stoppage"],
            "wins_sub": df[f"{corner}_win_by_Submission"],
            "wins_dec": (df[f"{corner}_win_by_Decision_Unanimous"] + df[f"{corner}_win_by_Decision_Split"]
                         + df[f"{corner}_win_by_Decision_Majority"]),
            "sig_str": df[f"{corner}_avg_SIG_STR_landed"],
            "sig_str_pct": df[f"{corner}_avg_SIG_STR_pct"],
            "td": df[f"{corner}_avg_TD_landed"],
            "td_pct": df[f"{corner}_avg_TD_pct"],
            "sub_att": df[f"{corner}_avg_SUB_ATT"],
        }))
    h = pd.concat(rows).sort_values(["fighter", "date"]).reset_index(drop=True)
    h["lost_by_ko"] = (~h["won"]) & (h["finish"] == "KO/TKO")

    previous = h.groupby("fighter")
    h["prev_won"] = previous["won"].shift(1)
    h["prev_lost_by_ko"] = previous["lost_by_ko"].shift(1)
    h["days_since_last_fight"] = (h["date"] - previous["date"].shift(1)).dt.days

    # Was the OPPONENT coming off a KO loss?
    opp_info = h[["fighter", "date", "prev_lost_by_ko"]].rename(
        columns={"fighter": "opponent", "prev_lost_by_ko": "opp_prev_lost_by_ko"})
    h = h.merge(opp_info.drop_duplicates(["opponent", "date"]), on=["opponent", "date"], how="left")

    h["period"] = pd.cut(h["year"], [2009, 2015, 2020, 2026], labels=PERIODS).astype(str)
    return h


def implied_prob(odds):
    """American odds -> probability according to the bookmakers."""
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(odds < 0, -odds / (-odds + 100), 100 / (odds + 100))


def bet_profit(bet_odds, bet_won, stake=10):
    """Profit of a bet with American odds: the payout if won, minus the stake if lost."""
    bet_odds = np.asarray(bet_odds, dtype=float)
    payout = np.where(bet_odds > 0, stake * bet_odds / 100, stake * 100 / -bet_odds)
    return np.where(bet_won, payout, -stake)


def insight(text):
    text = " ".join(text.split())          # tidy up double spaces
    st.info(f"💡 **Key insight:** {text}")


def sample_note(n, min_n=300):
    """Warning added to an insight when the selection is too small to trust."""
    if n < min_n:
        return f" ⚠️ *Only {n:,} cases in this selection – treat this as a hint, not a conclusion.*"
    return ""


fights_all = load_fights()
history_all = build_fighter_history(fights_all)

# =====================================================================
# HEADER
# =====================================================================
st.markdown(f"""
<style>
.hero {{
    position: relative; overflow: hidden;
    padding: 2.2rem 2.5rem; margin-bottom: 1.2rem; border-radius: 14px;
    background: linear-gradient(115deg, #D20A0A 0%, #7A0A0A 38%, #141416 75%);
    border: 1px solid #2A2A30;
}}
.hero svg {{ position: absolute; right: -40px; top: 50%; transform: translateY(-50%);
             width: 300px; height: 300px; opacity: 0.18; }}
.hero .kicker {{ color: #F2D27A; letter-spacing: .25em; font-size: .8rem; font-weight: 700; }}
.hero h1 {{ color: #FFFFFF; font-size: 3.2rem; font-weight: 900; letter-spacing: .04em;
            margin: .2rem 0 .3rem 0; padding: 0; line-height: 1.05; text-transform: uppercase; }}
.hero .sub {{ color: #F2F2F2; font-size: 1.1rem; opacity: .9; margin-bottom: 1rem; }}
.hero .chip {{ display: inline-block; margin: .2rem .4rem .2rem 0; padding: .3rem .8rem;
               border-radius: 999px; background: rgba(0,0,0,.35); color: #FFFFFF;
               border: 1px solid rgba(255,255,255,.25); font-size: .85rem; }}
@media (max-width: 640px) {{ .hero h1 {{ font-size: 2.2rem; }} .hero svg {{ width: 180px; height: 180px; }} }}
</style>
<div class="hero">
  <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <polygon points="30,2 70,2 98,30 98,70 70,98 30,98 2,70 2,30"
             fill="none" stroke="#FFFFFF" stroke-width="3"/>
    <polygon points="35,14 65,14 86,35 86,65 65,86 35,86 14,65 14,35"
             fill="none" stroke="#FFFFFF" stroke-width="1.5"/>
  </svg>
  <div class="kicker">DATA SCIENCE · MIXED MARTIAL ARTS</div>
  <h1>🥊 UFC Fight Lab</h1>
  <div class="sub">What makes a fighter win – and can data see what the experts miss?</div>
  <span class="chip">🥊 {len(fights_all):,} fights</span>
  <span class="chip">📅 {fights_all["year"].min()}–{fights_all["year"].max()}</span>
  <span class="chip">👊 {pd.concat([fights_all["R_fighter"], fights_all["B_fighter"]]).nunique():,} fighters</span>
</div>
""", unsafe_allow_html=True)

with st.expander("About this app"):
    st.markdown(
        "**What can this app do?** Explore 16 years of UFC fights (2010–2026), "
        "compare fighters, and (soon) predict and simulate fights.\n\n"
        "**How to use it?** Use the filters in the sidebar to choose the years and divisions. "
        "All charts are interactive: hover for details, zoom, or click the legend to hide series.\n\n"
        "**Data:** Ultimate UFC Dataset (github.com/shortlikeafox/ultimate_ufc_dataset)."
    )

# =====================================================================
# SIDEBAR FILTERS
# =====================================================================
st.sidebar.header("Filters")

min_year, max_year = int(fights_all["year"].min()), int(fights_all["year"].max())
year_range = st.sidebar.slider("Years", min_year, max_year, (min_year, max_year))

gender_choice = st.sidebar.radio("Division", ["All", "Men", "Women"], horizontal=True)
gender_map = {"All": ["MALE", "FEMALE"], "Men": ["MALE"], "Women": ["FEMALE"]}
classes_available = sorted(
    fights_all.loc[fights_all["gender"].isin(gender_map[gender_choice]), "weight_class"].unique())

with st.sidebar.expander("Weight classes"):
    selected_classes = st.multiselect("Select weight classes", classes_available,
                                      default=classes_available, label_visibility="collapsed")


def apply_filters(table):
    return table[table["year"].between(*year_range)
                 & table["gender"].isin(gender_map[gender_choice])
                 & table["weight_class"].isin(selected_classes)]


fights = apply_filters(fights_all)
history = apply_filters(history_all)

st.sidebar.caption(f"Showing **{len(fights):,}** fights")
st.sidebar.caption("ℹ️ These filters apply to the **Explore** tab.")

if fights.empty:
    st.warning("No fights match these filters. Try selecting more years or weight classes.")
    st.stop()

# =====================================================================
# TABS
# =====================================================================
tab_explore, tab_analyzer, tab_predict, tab_simulate = st.tabs(
    ["📊 Explore", "🥊 Fight Analyzer", "🔮 Predict", "🎲 Simulate"])

with tab_explore:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fights", f"{len(fights):,}")
    col2.metric("Favourite wins", f"{fights['favourite_won'].mean():.1%}")
    col3.metric("KO / TKO", f"{(fights['finish'] == 'KO/TKO').mean():.1%}")
    col4.metric("Submissions", f"{(fights['finish'] == 'SUB').mean():.1%}")

    story = st.radio("Choose a story", [
        "📈 The bookmakers' benchmark",
        "⏳ Time and durability",
        "💥 Anatomy of a KO",
        "🧠 Beating the experts",
    ], horizontal=True)
    st.divider()

    # -----------------------------------------------------------------
    # STORY 1 – THE BOOKMAKERS' BENCHMARK
    # -----------------------------------------------------------------
    if story == "📈 The bookmakers' benchmark":
        st.header("📈 The bookmakers' benchmark")
        st.write("Betting odds are the best public estimate of each fighter's chances. "
                 "How good are they?")

        st.subheader("How often does the favourite win?")
        by_year = (fights.groupby("year")["favourite_won"]
                   .agg(win_rate="mean", fights="count").reset_index())
        by_year["win_rate"] *= 100
        fig = px.line(by_year, x="year", y="win_rate", markers=True, hover_data=["fights"],
                      labels={"year": "Year", "win_rate": "Favourite win rate (%)", "fights": "Fights"})
        st.plotly_chart(fig)
        fav_rate = fights["favourite_won"].mean()
        n_odds = int(fights["favourite_won"].notna().sum())
        text = f"In this selection the favourite wins **{fav_rate:.1%}** of fights – "
        if fav_rate >= 0.70:
            text += "these fights are quite predictable for the bookmakers."
        elif fav_rate <= 0.62:
            text += "an unusually unpredictable group, where upsets are common."
        else:
            text += "about 2 out of 3. Bookmakers are accurate, but 1 in 3 fights is an upset."
        valid_years = by_year[by_year["fights"] >= 20]
        if len(valid_years) >= 3:
            best = valid_years.loc[valid_years["win_rate"].idxmax()]
            worst = valid_years.loc[valid_years["win_rate"].idxmin()]
            text += (f" Most predictable year: **{int(best['year'])}** ({best['win_rate']:.1f}%); "
                     f"least predictable: **{int(worst['year'])}** ({worst['win_rate']:.1f}%).")
        insight(text + sample_note(n_odds))

        st.subheader("In which weight classes are upsets most common?")
        upsets = fights.dropna(subset=["favourite_won"]).copy()
        upsets["upset"] = upsets["favourite_won"] == 0
        upset_table = (upsets.groupby("weight_class")["upset"]
                       .agg(upset_rate="mean", fights="count").reset_index())
        upset_table = upset_table[upset_table["fights"] >= 30]
        upset_table["upset_rate"] *= 100
        fig = px.bar(upset_table.sort_values("upset_rate"), x="upset_rate", y="weight_class",
                     orientation="h", text_auto=".1f", hover_data=["fights"],
                     labels={"upset_rate": "Upset rate (%)", "weight_class": "", "fights": "Fights"})
        st.plotly_chart(fig)
        if len(upset_table) >= 2:
            hi = upset_table.loc[upset_table["upset_rate"].idxmax()]
            lo = upset_table.loc[upset_table["upset_rate"].idxmin()]
            avg_upset = upsets["upset"].mean() * 100
            text = (f"Most upsets: **{hi['weight_class']}** ({hi['upset_rate']:.1f}%). "
                    f"Fewest: **{lo['weight_class']}** ({lo['upset_rate']:.1f}%). Average: {avg_upset:.1f}%. ")
            if hi["upset_rate"] - lo["upset_rate"] < 8:
                text += "The differences are small, so the division has little influence on upsets."
            else:
                text += "That is a big gap: some divisions are much harder to predict than others."
            if "Heavyweight" in upset_table["weight_class"].values:
                hw_upset = upset_table.set_index("weight_class").loc["Heavyweight", "upset_rate"]
                hw_ko = (fights.loc[fights["weight_class"] == "Heavyweight", "finish"] == "KO/TKO").mean() * 100
                if hw_upset <= avg_upset:
                    text += (f" Myth busted: heavyweights have the most KOs ({hw_ko:.0f}% of fights) but "
                             f"**not** more upsets ({hw_upset:.1f}%) – the hard puncher is usually the favourite.")
                else:
                    text += (f" Here heavyweights ({hw_ko:.0f}% KOs) do have more upsets than average "
                             f"({hw_upset:.1f}%): the 'puncher's chance' shows up in this selection.")
            insight(text + sample_note(int(upset_table["fights"].sum())))
        else:
            insight("Not enough fights to compare divisions (at least 30 fights per division are needed). "
                    "Try selecting more years or weight classes.")

        st.subheader("€10 on every fight: would you make money?")
        bets = history.dropna(subset=["odds", "opp_odds"])
        bets = bets[bets["odds"] != bets["opp_odds"]].sort_values("date").copy()
        bets["strategy"] = np.where(bets["odds"] < bets["opp_odds"],
                                    "Always bet on the favourite", "Always bet on the underdog")
        bets["profit"] = bet_profit(bets["odds"], bets["won"])
        bets["accumulated_profit"] = bets.groupby("strategy")["profit"].cumsum()
        fig = px.line(bets, x="date", y="accumulated_profit", color="strategy",
                      labels={"date": "Date", "accumulated_profit": "Accumulated profit (€)", "strategy": ""})
        st.plotly_chart(fig)
        roi = bets.groupby("strategy")["profit"].sum() / (bets.groupby("strategy").size() * 10) * 100
        roi_fav = roi.get("Always bet on the favourite", 0)
        roi_dog = roi.get("Always bet on the underdog", 0)
        if roi_fav < 0 and roi_dog < 0:
            text = (f"Betting blindly loses money: **{roi_fav:.1f}%** on favourites and **{roi_dog:.1f}%** on "
                    "underdogs. This is the bookmaker's margin. ")
        else:
            winner = "favourites" if roi_fav > roi_dog else "underdogs"
            text = (f"Surprise: in this selection, always betting on **{winner}** made money "
                    f"(favourites {roi_fav:.1f}%, underdogs {roi_dog:.1f}%). This usually only happens in "
                    "small or unusual groups. ")
        text += ("Underdogs lose more – they are overpriced (favourite-longshot bias). "
                 if roi_dog < roi_fav else "Here favourites lose more than underdogs. ")
        text += "Winning often is not the same as making money."
        insight(text + sample_note(len(bets) // 2))

    # -----------------------------------------------------------------
    # STORY 2 – TIME AND DURABILITY
    # -----------------------------------------------------------------
    elif story == "⏳ Time and durability":
        st.header("⏳ Time and durability")
        st.write("Age, reach and past knockouts: what really makes a difference?")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Age advantage")
            age = history[(history["age"] < history["opp_age"])].copy()   # one row per fight: the younger one
            age["gap_group"] = pd.cut(age["opp_age"] - age["age"], [0, 2, 5, 8, 30],
                                      labels=["1-2 yrs", "3-5 yrs", "6-8 yrs", "9+ yrs"])
            age_table = (age.groupby("gap_group", observed=True)["won"]
                         .agg(win_rate="mean", fights="count").reset_index())
            age_table["win_rate"] *= 100
            fig = px.bar(age_table, x="gap_group", y="win_rate", text_auto=".1f", hover_data=["fights"],
                         range_y=[0, 75],
                         labels={"gap_group": "Age gap", "win_rate": "Younger fighter wins (%)", "fights": "Fights"})
            st.plotly_chart(fig)
        with col_b:
            st.subheader("Reach advantage")
            reach = history[(history["reach"] > history["opp_reach"])].copy()   # the longer-reach fighter
            reach["gap_group"] = pd.cut(reach["reach"] - reach["opp_reach"], [0, 5, 10, 15, 100],
                                        labels=["0-5 cm", "5-10 cm", "10-15 cm", "15+ cm"])
            reach_table = (reach.groupby("gap_group", observed=True)["won"]
                           .agg(win_rate="mean", fights="count").reset_index())
            reach_table["win_rate"] *= 100
            fig = px.bar(reach_table, x="gap_group", y="win_rate", text_auto=".1f", hover_data=["fights"],
                         range_y=[0, 75],
                         labels={"gap_group": "Reach gap", "win_rate": "Longer reach wins (%)", "fights": "Fights"})
            st.plotly_chart(fig)
        if not age_table.empty and not reach_table.empty:
            age_top, reach_top = age_table.iloc[-1], reach_table.iloc[-1]
            text = (f"With the biggest gaps, the younger fighter wins **{age_top['win_rate']:.1f}%** "
                    f"({age_top['gap_group']} older opponent) and the longer-reach fighter wins "
                    f"**{reach_top['win_rate']:.1f}%** ({reach_top['gap_group']} reach gap). ")
            if age_top["win_rate"] - reach_top["win_rate"] > 3:
                text += "**Age matters more than reach.**"
            elif reach_top["win_rate"] - age_top["win_rate"] > 3:
                text += "**In this selection, reach matters more than age.**"
            else:
                text += "**Here age and reach have a similar effect.**"
            small = min(age_top["fights"], reach_top["fights"])
            insight(text + sample_note(int(small), min_n=100))
        else:
            insight("Not enough fights with age and reach data for this selection.")

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Win rate by age")
            by_age = history.dropna(subset=["age"]).copy()
            by_age["age_group"] = pd.cut(by_age["age"], [17, 24, 27, 30, 33, 36, 50],
                                         labels=["<25", "25-27", "28-30", "31-33", "34-36", "37+"])
            curve = (by_age.groupby("age_group", observed=True)
                     .agg(win_rate=("won", "mean"), ko_loss_rate=("lost_by_ko", "mean"),
                          fights=("won", "count")).reset_index())
            curve[["win_rate", "ko_loss_rate"]] *= 100
            fig = px.line(curve, x="age_group", y="win_rate", markers=True, hover_data=["fights"],
                          labels={"age_group": "Age at fight", "win_rate": "Win rate (%)", "fights": "Fights"})
            st.plotly_chart(fig)
        with col_d:
            st.subheader("Does the chin fade with age?")
            fig = px.line(curve, x="age_group", y="ko_loss_rate", markers=True, hover_data=["fights"],
                          labels={"age_group": "Age at fight", "ko_loss_rate": "Lost by KO (%)", "fights": "Fights"})
            fig.update_traces(line_color=UFC_RED)
            st.plotly_chart(fig)
        cv = curve[curve["fights"] >= 50]
        if len(cv) >= 3:
            first, last = cv.iloc[0], cv.iloc[-1]
            text = (f"Win rate goes from **{first['win_rate']:.1f}%** ({first['age_group']}) to "
                    f"**{last['win_rate']:.1f}%** ({last['age_group']}). ")
            ko_min, ko_max = cv["ko_loss_rate"].min(), cv["ko_loss_rate"].max()
            ratio = last["ko_loss_rate"] / first["ko_loss_rate"] if first["ko_loss_rate"] > 0 else np.nan
            if ko_max - ko_min < 3:
                text += (f"But the share of fights lost by KO **barely changes** with age (between "
                         f"{ko_min:.1f}% and {ko_max:.1f}%): in this selection, age costs wins, not chins.")
            elif ratio >= 1.8:
                text += (f"And they lose in a harsher way: the share of fights lost by KO roughly **doubles** "
                         f"({first['ko_loss_rate']:.1f}% → {last['ko_loss_rate']:.1f}%). The chin fades with age.")
            elif ratio >= 1.3:
                text += (f"The share of fights lost by KO **rises clearly** "
                         f"({first['ko_loss_rate']:.1f}% → {last['ko_loss_rate']:.1f}%).")
            else:
                low = cv.loc[cv["ko_loss_rate"].idxmin()]
                high = cv.loc[cv["ko_loss_rate"].idxmax()]
                text += (f"The KO-loss rate moves irregularly (lowest at {low['age_group']}: "
                         f"{low['ko_loss_rate']:.1f}%, highest at {high['age_group']}: {high['ko_loss_rate']:.1f}%), "
                         "with no clear trend.")
            small_groups = curve.loc[curve["fights"] < 150, "age_group"].astype(str).tolist()
            if small_groups:
                text += (f" ⚠️ *Few fights in: {', '.join(small_groups)} – ups and downs there are "
                         "probably noise. Check the y-axis scale too: small changes can look big.*")
            insight(text)
        else:
            insight("Not enough fights per age group for this selection.")

        st.subheader("Do fighters recover after being knocked out?")
        rec = history.dropna(subset=["prev_won"]).copy()
        rec["previous_fight"] = np.select(
            [rec["prev_lost_by_ko"] == True, rec["prev_won"] == False],
            ["Lost by KO", "Lost (other way)"], default="Won")
        p_me, p_opp = implied_prob(rec["odds"]), implied_prob(rec["opp_odds"])
        rec["expected_win"] = p_me / (p_me + p_opp)
        summary = (rec.groupby("previous_fight")
                   .agg(**{"Actual win rate": ("won", "mean"),
                           "Expected by bookmakers": ("expected_win", "mean"),
                           "Knocked out again": ("lost_by_ko", "mean"),
                           "fights": ("won", "count")})
                   .reindex(["Won", "Lost (other way)", "Lost by KO"]).reset_index())
        summary[["Actual win rate", "Expected by bookmakers", "Knocked out again"]] *= 100
        chart = summary.melt(id_vars=["previous_fight", "fights"], var_name="measure", value_name="pct")
        fig = px.bar(chart, x="previous_fight", y="pct", color="measure", barmode="group",
                     text_auto=".1f", hover_data=["fights"],
                     labels={"previous_fight": "Result of previous fight", "pct": "%", "measure": ""})
        st.plotly_chart(fig)
        s = summary.set_index("previous_fight")
        if "Lost by KO" in s.index and s.loc["Lost by KO", "fights"] > 0:
            ko_row, won_row = s.loc["Lost by KO"], s.loc["Won"]
            diff = ko_row["Actual win rate"] - ko_row["Expected by bookmakers"]
            text = (f"After a KO loss, fighters win **{ko_row['Actual win rate']:.1f}%** of their next fight "
                    f"(after a win: {won_row['Actual win rate']:.1f}%). ")
            if diff < -1:
                text += (f"The bookmakers expected {ko_row['Expected by bookmakers']:.1f}%, so they "
                         "**overrate** fighters coming off a KO. ")
            elif diff > 1:
                text += (f"The bookmakers expected only {ko_row['Expected by bookmakers']:.1f}%, so here they "
                         "**underrate** them. ")
            else:
                text += f"The bookmakers price this correctly ({ko_row['Expected by bookmakers']:.1f}% expected). "
            ratio = ko_row["Knocked out again"] / won_row["Knocked out again"] if won_row["Knocked out again"] > 0 else np.nan
            if ratio >= 1.3 or ko_row["Knocked out again"] - won_row["Knocked out again"] >= 5:
                text += (f"And **{ko_row['Knocked out again']:.1f}%** are knocked out again "
                         f"(vs {won_row['Knocked out again']:.1f}% after a win) – **a KO leaves a mark**.")
            else:
                text += (f"Being knocked out again is not much more likely here "
                         f"({ko_row['Knocked out again']:.1f}% vs {won_row['Knocked out again']:.1f}% after a win).")
            insight(text + sample_note(int(ko_row["fights"])))
        else:
            insight("Not enough fighters coming off a KO loss in this selection.")

    # -----------------------------------------------------------------
    # STORY 3 – ANATOMY OF A KO
    # -----------------------------------------------------------------
    elif story == "💥 Anatomy of a KO":
        st.header("💥 Anatomy of a KO")
        st.write("How and when do fights end?")

        st.subheader("How do fights end in each weight class?")
        finish_table = pd.crosstab(fights["weight_class"], fights["finish"], normalize="index") * 100
        finish_table = finish_table.reindex(columns=["KO/TKO", "SUB", "U-DEC", "S-DEC"], fill_value=0)
        fig = px.bar(finish_table.sort_values("KO/TKO"), orientation="h", barmode="stack",
                     color_discrete_map=FINISH_COLOURS,
                     labels={"value": "% of fights", "weight_class": "", "finish": "Finish method"})
        st.plotly_chart(fig)
        if len(finish_table) >= 2:
            decisions = finish_table["U-DEC"] + finish_table["S-DEC"]
            text = (f"Most knockouts: **{finish_table['KO/TKO'].idxmax()}** ({finish_table['KO/TKO'].max():.0f}% of fights). "
                    f"Fewest: **{finish_table['KO/TKO'].idxmin()}** ({finish_table['KO/TKO'].min():.0f}%). "
                    f"Most decisions: **{decisions.idxmax()}** ({decisions.max():.0f}%). ")
            sub_spread = finish_table["SUB"].max() - finish_table["SUB"].min()
            if sub_spread < 10:
                text += "The submission rate is similar in every division."
            else:
                text += (f"Submissions vary a lot: from {finish_table['SUB'].min():.0f}% to "
                         f"{finish_table['SUB'].max():.0f}% ({finish_table['SUB'].idxmax()}).")
        else:
            row = finish_table.iloc[0]
            text = (f"In {finish_table.index[0]}: {row['KO/TKO']:.0f}% KO/TKO, {row['SUB']:.0f}% submissions "
                    f"and {row['U-DEC'] + row['S-DEC']:.0f}% decisions.")
        insight(text + sample_note(len(fights)))

        st.subheader("When do knockouts happen?")
        fight_length = st.radio("Fight length", ["3-round fights", "5-round fights (main events & title fights)"],
                                horizontal=True)
        n_rounds = 3 if fight_length.startswith("3") else 5
        ko = fights[(fights["finish"] == "KO/TKO") & (fights["no_of_rounds"] == n_rounds)]
        ko = ko.dropna(subset=["finish_round", "finish_round_time"])
        ko = ko[ko["finish_round"] <= n_rounds].copy()
        if ko.empty:
            st.warning("No knockouts for these filters.")
        else:
            parts = ko["finish_round_time"].str.split(":")
            seconds = parts.str[0].astype(int) * 60 + parts.str[1].astype(int)
            ko["minute"] = np.ceil(seconds / 60).clip(lower=1).astype(int)
            ko["round"] = "Round " + ko["finish_round"].astype(int).astype(str)
            table = pd.crosstab(ko["round"], ko["minute"], normalize="all") * 100
            fig = px.imshow(table.round(1), text_auto=True, color_continuous_scale="Reds", aspect="auto",
                            labels={"x": "Minute of the round", "y": "", "color": "% of KOs"})
            st.plotly_chart(fig)
            first_round = (ko["finish_round"] == 1).mean()
            peak_round, peak_minute = table.stack().idxmax()
            last_minute = (ko["minute"] == 5).mean()
            text = (f"**{first_round:.1%}** of these {len(ko):,} knockouts happen in round 1. "
                    f"The most dangerous moment is **{peak_round}, minute {peak_minute}** "
                    f"({table.stack().max():.1f}% of all KOs). ")
            if last_minute > 0.22:
                text += (f"The last minute of a round is especially dangerous ({last_minute:.0%} of KOs): "
                         "late-round urgency and stoppages between rounds.")
            if n_rounds == 5:
                text += " Title fighters pace themselves more, so their KOs are spread over more rounds."
            insight(text + sample_note(len(ko), min_n=200))

    # -----------------------------------------------------------------
    # STORY 4 – BEATING THE EXPERTS
    # -----------------------------------------------------------------
    elif story == "🧠 Beating the experts":
        st.header("🧠 Beating the experts")
        st.write("Do our findings contain information that the bookmakers' odds miss? "
                 "We test each one with a simple rule (€10 per fight) and measure the return.")
        st.caption("Odds are used as an expert benchmark, not as betting advice.")

        fh = history.dropna(subset=["odds", "opp_odds"])
        fh = fh[fh["odds"] != fh["opp_odds"]]
        gap = fh["opp_age"] - fh["age"]
        favourite = fh["odds"] < fh["opp_odds"]
        rules = {
            "Baseline: always the favourite": favourite,
            "Baseline: always the underdog": ~favourite,
            "Reach: longer reach (10+ cm)": (fh["reach"] - fh["opp_reach"]) >= 10,
            "KO: opponent coming off a KO loss": fh["opp_prev_lost_by_ko"] == True,
            "Age: younger by 5+ years": gap >= 5,
            "Age: younger by 8+ years": gap >= 8,
            "Age: younger by 5+ years AND favourite": (gap >= 5) & favourite,
        }
        logs = []
        for name, mask in rules.items():
            sel = fh[mask]
            logs.append(pd.DataFrame({"strategy": name, "period": sel["period"].values,
                                      "profit": bet_profit(sel["odds"], sel["won"])}))
        log = pd.concat(logs)

        def roi(p):
            return p.sum() / (len(p) * 10) * 100

        table = log.pivot_table(index="strategy", columns="period", values="profit", aggfunc=roi)
        table = table.reindex(columns=[p for p in PERIODS if p in table.columns])
        table["All years"] = log.groupby("strategy")["profit"].apply(roi)
        table = table.reindex(list(rules.keys()))
        bets_count = log.groupby("strategy").size().reindex(list(rules.keys()))

        st.subheader("Which findings beat the bookmakers? (return on investment, %)")
        fig = px.imshow(table.round(1), text_auto=True, color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0, aspect="auto",
                        labels={"x": "Period", "y": "", "color": "ROI (%)"})
        st.plotly_chart(fig)
        st.caption("Number of bets per rule: " +
                   " · ".join(f"{k.split(':')[0]} {k.split(':')[1].strip()}: {v:,}" for k, v in bets_count.items()))
        periods_present = [p for p in PERIODS if p in table.columns]
        short = {
            "Reach: longer reach (10+ cm)": "Longer reach (10+ cm)",
            "KO: opponent coming off a KO loss": "Opponent coming off a KO",
            "Age: younger by 5+ years": "Younger by 5+ years",
            "Age: younger by 8+ years": "Younger by 8+ years",
            "Age: younger by 5+ years AND favourite": "Younger by 5+ years & favourite",
        }
        strategies = list(short.keys())
        always_pos = [r for r in strategies
                      if len(periods_present) >= 2 and (table.loc[r, periods_present] > 0).all()]
        pos_total = [r for r in strategies if table.loc[r, "All years"] > 0 and r not in always_pos]
        losers = [r for r in strategies if table.loc[r, "All years"] <= 0]
        few = [r for r in strategies if bets_count[r] < 300]

        def names(rule_list):
            return ", ".join(short[r] for r in rule_list)

        lines = []
        if always_pos:
            lines.append(f"✅ **Beat the bookmakers in every period:** {names(always_pos)}")
        if pos_total:
            lines.append(f"🟡 **Make money overall, but not in every period:** {names(pos_total)}")
        if losers:
            lines.append(f"❌ **Lose money – already priced into the odds:** {names(losers)}")
        if not always_pos and not pos_total:
            lines.append("**No rule beats the bookmakers in this selection.**")
        ko_rule = "KO: opponent coming off a KO loss"
        if len(periods_present) >= 2:
            if table.loc[ko_rule, periods_present[0]] > 0 and table.loc[ko_rule, periods_present[-1]] < 0:
                lines.append("📉 The KO edge existed and then disappeared – **the market learns**.")
        if few:
            lines.append(f"⚠️ *Fewer than 300 bets (treat as hints): {names(few)}*")
        st.info("💡 **Key insight**\n\n" + "\n".join(f"- {line}" for line in lines))

        st.subheader("Real examples: veteran favourites beaten by much younger fighters")
        ex = fights.dropna(subset=["R_odds", "B_odds", "R_age", "B_age"]).copy()
        red_fav = ex["R_odds"] < ex["B_odds"]
        examples = pd.DataFrame({
            "Date": ex["date"].dt.date,
            "Favourite": np.where(red_fav, ex["R_fighter"], ex["B_fighter"]),
            "Fav. age": np.where(red_fav, ex["R_age"], ex["B_age"]),
            "Fav. odds": np.where(red_fav, ex["R_odds"], ex["B_odds"]),
            "Winner": np.where(red_fav, ex["B_fighter"], ex["R_fighter"]),
            "Winner age": np.where(red_fav, ex["B_age"], ex["R_age"]),
            "Winner odds": np.where(red_fav, ex["B_odds"], ex["R_odds"]),
            "Title fight": ex["title_bout"],
            "Finish": ex["finish"],
            "upset": np.where(red_fav, "Red", "Blue") != ex["Winner"],
            "rounds": ex["no_of_rounds"],
        })
        examples = examples[examples["upset"] & (examples["Fav. age"] - examples["Winner age"] >= 7)
                            & (examples["rounds"] == 5)]
        st.dataframe(examples.drop(columns=["upset", "rounds"]).sort_values("Date", ascending=False),
                     hide_index=True)
        if len(examples):
            recent = (pd.to_datetime(examples["Date"]).dt.year >= 2024).sum()
            titles = int(examples["Title fight"].sum())
            n_ex = len(examples)
            text = (f"**{n_ex}** main event{'s' if n_ex > 1 else ''} or title fight{'s' if n_ex > 1 else ''} "
                    f"in this selection {'were' if n_ex > 1 else 'was'} won by an underdog at least 7 years "
                    f"younger than the favourite ({titles} title fight{'s' if titles != 1 else ''}, {recent} since 2024).")
            if n_ex >= 5 and recent >= 2:
                text += " Famous veterans keep being favourites after their prime – and it keeps happening."
            insight(text)
        else:
            st.caption("No examples in this selection.")

def fight_analyzer():
    """The Fight Analyzer tab: tale of the tape, data signals and career timeline."""
    st.header("🥊 Fight Analyzer")
    st.write("Pick two fighters and see what 16 years of UFC data say about the matchup.")

    DATA_END = fights_all["date"].max()
    TODAY = pd.Timestamp.today().normalize()
    DECISIONS = ["U-DEC", "S-DEC", "M-DEC"]

    latest = history_all.sort_values("date").groupby("fighter").tail(1).set_index("fighter")
    fights_in_data = history_all.groupby("fighter").size()

    # ---------- Fighter selection ----------
    dream_mode = st.checkbox("🌟 Dream fight mode (any weight class)",
                             help="Match fighters from different weight classes. "
                                  "Each fighter is listed in the weight class of their last fight.")
    if dream_mode:
        options = sorted(latest.index)
        fmt = lambda name: f"{name} ({latest.loc[name, 'weight_class']})"
    else:
        classes = sorted(latest["weight_class"].unique())
        default_class = classes.index("Lightweight") if "Lightweight" in classes else 0
        chosen_class = st.selectbox("Weight class", classes, index=default_class)
        options = sorted(latest.index[latest["weight_class"] == chosen_class])
        fmt = lambda name: name

    def default_index(name, fallback):
        return options.index(name) if name in options else min(fallback, len(options) - 1)

    col_a, col_vs, col_b = st.columns([5, 1, 5])
    fighter_a = col_a.selectbox("🔴 Fighter A", options, index=default_index("Ilia Topuria", 0),
                                format_func=fmt, key="fighter_a")
    col_vs.markdown("<h2 style='text-align:center; margin-top:1.2rem;'>VS</h2>", unsafe_allow_html=True)
    fighter_b = col_b.selectbox("🔵 Fighter B", options, index=default_index("Charles Oliveira", 1),
                                format_func=fmt, key="fighter_b")

    if fighter_a == fighter_b:
        st.warning("Choose two different fighters.")
        return

    # ---------- Build each fighter's profile ----------
    def profile(name):
        r = latest.loc[name]
        won = bool(r["won"])
        last_finish = r["finish"] if isinstance(r["finish"], str) else "-"
        wins = np.nan_to_num(r["wins"]) + won
        losses = np.nan_to_num(r["losses"]) + (not won)
        years_since = (TODAY - r["date"]).days / 365.25
        return {
            "name": name,
            "weight_class": r["weight_class"],
            "age": r["age"] + years_since if pd.notna(r["age"]) else np.nan,
            "height": r["height"], "reach": r["reach"],
            "stance": r["stance"] if isinstance(r["stance"], str) else "-",
            "wins": int(wins), "losses": int(losses),
            "win_pct": wins / (wins + losses) if wins + losses > 0 else np.nan,
            "streak": int(np.nan_to_num(r["win_streak"]) + 1) if won else -int(np.nan_to_num(r["lose_streak"]) + 1),
            "wins_ko": int(np.nan_to_num(r["wins_ko"]) + (won and last_finish == "KO/TKO")),
            "wins_sub": int(np.nan_to_num(r["wins_sub"]) + (won and last_finish == "SUB")),
            "wins_dec": int(np.nan_to_num(r["wins_dec"]) + (won and last_finish in DECISIONS)),
            "sig_str": r["sig_str"], "sig_str_pct": r["sig_str_pct"],
            "td": r["td"], "td_pct": r["td_pct"], "sub_att": r["sub_att"],
            "last_won": won, "last_finish": last_finish, "last_opponent": r["opponent"],
            "last_date": r["date"], "last_lost_by_ko": bool(r["lost_by_ko"]),
            "days_inactive": (DATA_END - r["date"]).days,
            "fights_in_data": int(fights_in_data.get(name, 0)),
        }

    A, B = profile(fighter_a), profile(fighter_b)

    # ---------- 1. Tale of the tape ----------
    st.subheader("📋 Tale of the tape")

    def fmt_num(v, pattern):
        return pattern.format(v) if pd.notna(v) else "–"

    def streak_text(s):
        return f"{s} win{'s' if s > 1 else ''}" if s > 0 else f"{-s} loss{'es' if s < -1 else ''}"

    def last_fight_text(p):
        return (f"{'W' if p['last_won'] else 'L'} vs {p['last_opponent']} "
                f"({p['last_finish']}, {p['last_date']:%b %Y})")

    # (label, value A, value B, text A, text B, which is better: "high" / "low" / None)
    tape = [
        ("Age (today, est.)", A["age"], B["age"], fmt_num(A["age"], "{:.0f}"), fmt_num(B["age"], "{:.0f}"), "low"),
        ("Height", A["height"], B["height"], fmt_num(A["height"], "{:.0f} cm"), fmt_num(B["height"], "{:.0f} cm"), "high"),
        ("Reach", A["reach"], B["reach"], fmt_num(A["reach"], "{:.0f} cm"), fmt_num(B["reach"], "{:.0f} cm"), "high"),
        ("Stance", None, None, A["stance"], B["stance"], None),
        ("UFC record", A["win_pct"], B["win_pct"], f"{A['wins']}-{A['losses']}", f"{B['wins']}-{B['losses']}", "high"),
        ("Current streak", A["streak"], B["streak"], streak_text(A["streak"]), streak_text(B["streak"]), "high"),
        ("Wins by KO/TKO", A["wins_ko"], B["wins_ko"], str(A["wins_ko"]), str(B["wins_ko"]), "high"),
        ("Wins by submission", A["wins_sub"], B["wins_sub"], str(A["wins_sub"]), str(B["wins_sub"]), "high"),
        ("Wins by decision", A["wins_dec"], B["wins_dec"], str(A["wins_dec"]), str(B["wins_dec"]), None),
        ("Sig. strikes landed / min", A["sig_str"], B["sig_str"], fmt_num(A["sig_str"], "{:.2f}"), fmt_num(B["sig_str"], "{:.2f}"), "high"),
        ("Striking accuracy", A["sig_str_pct"], B["sig_str_pct"], fmt_num(A["sig_str_pct"], "{:.0%}"), fmt_num(B["sig_str_pct"], "{:.0%}"), "high"),
        ("Takedowns / 15 min", A["td"], B["td"], fmt_num(A["td"], "{:.2f}"), fmt_num(B["td"], "{:.2f}"), "high"),
        ("Takedown accuracy", A["td_pct"], B["td_pct"], fmt_num(A["td_pct"], "{:.0%}"), fmt_num(B["td_pct"], "{:.0%}"), "high"),
        ("Submission attempts / 15 min", A["sub_att"], B["sub_att"], fmt_num(A["sub_att"], "{:.1f}"), fmt_num(B["sub_att"], "{:.1f}"), "high"),
        ("Weight class", None, None, A["weight_class"], B["weight_class"], None),
        ("Last fight (in data)", None, None, last_fight_text(A), last_fight_text(B), None),
    ]

    html_rows = ""
    for label, va, vb, ta, tb, better in tape:
        a_better = b_better = False
        if better and pd.notna(va) and pd.notna(vb) and va != vb:
            a_better = (va > vb) if better == "high" else (va < vb)
            b_better = not a_better
        html_rows += (
            "<tr style='border-bottom:1px solid #2A2A30;'>"
            f"<td style='padding:.45rem .8rem; text-align:right; width:40%; {'color:#2ECC71; font-weight:700;' if a_better else ''}'>{ta}</td>"
            f"<td style='padding:.45rem .8rem; text-align:center; color:#9A9AA0; font-size:.85rem; width:20%;'>{label}</td>"
            f"<td style='padding:.45rem .8rem; text-align:left; width:40%; {'color:#2ECC71; font-weight:700;' if b_better else ''}'>{tb}</td>"
            "</tr>")

    st.markdown(f"""
    <table style="width:100%; border-collapse:collapse; background:#1C1C21; border-radius:10px; overflow:hidden;">
      <tr style="background:linear-gradient(90deg,#D20A0A,#7A0A0A 45%,#1C1C21 50%,#1E3A5F 55%,#4A90D9);">
        <th style="padding:.7rem; text-align:right; font-size:1.2rem; color:white;">🔴 {A['name']}</th>
        <th style="padding:.7rem; text-align:center; color:white;">VS</th>
        <th style="padding:.7rem; text-align:left; font-size:1.2rem; color:white;">{B['name']} 🔵</th>
      </tr>
      {html_rows}
    </table>
    """, unsafe_allow_html=True)
    st.caption(f"🟢 Green = advantage. Career numbers come from the fighter's UFC record in the dataset. "
               f"Data up to **{DATA_END:%d %B %Y}**: fights after that date are not included. "
               "Age is estimated for today.")

    # ---------- 2. Data signals (based on our findings) ----------
    st.subheader("🚦 What the data says")
    st.caption("Each signal is based on how similar situations ended in past UFC fights (2010–2026).")

    H = history_all
    signals = []   # (colour, fighter, text)

    # Age gap
    if pd.notna(A["age"]) and pd.notna(B["age"]):
        gap = round(abs(A["age"] - B["age"]))
        younger = A if A["age"] < B["age"] else B
        for lo, hi, label in [(9, 99, "9+"), (6, 9, "6-8"), (3, 6, "3-5")]:
            if lo <= gap < hi:
                hist = H[(H["opp_age"] - H["age"] >= lo) & (H["opp_age"] - H["age"] < hi)]
                signals.append(("🟢", younger["name"],
                                f"**{younger['name']}** is about {gap} years younger. In past UFC fights with a "
                                f"{label} year age gap, the younger fighter won **{hist['won'].mean():.0%}** "
                                f"({len(hist):,} fights)."))
                break

    # Reach gap
    if pd.notna(A["reach"]) and pd.notna(B["reach"]):
        gap = round(abs(A["reach"] - B["reach"]))
        longer = A if A["reach"] > B["reach"] else B
        for lo, hi, label in [(15, 999, "15+"), (10, 15, "10-15"), (5, 10, "5-10")]:
            if lo <= gap < hi:
                hist = H[(H["reach"] - H["opp_reach"] >= lo) & (H["reach"] - H["opp_reach"] < hi)]
                signals.append(("🟢", longer["name"],
                                f"**{longer['name']}** has {gap} cm more reach. With a {label} cm reach advantage, "
                                f"fighters won **{hist['won'].mean():.0%}** ({len(hist):,} fights) – a smaller edge "
                                "than age, and already priced in by the bookmakers."))
                break

    # Coming off a KO / danger zone
    ko_again = H.loc[H["prev_lost_by_ko"] == True, "lost_by_ko"].mean()
    ko_after_win = H.loc[H["prev_won"] == True, "lost_by_ko"].mean()
    win_after_ko = H.loc[H["prev_lost_by_ko"] == True, "won"].mean()
    danger = H[(H["prev_lost_by_ko"] == True) & (H["age"] >= 35)]
    for p in (A, B):
        if p["last_lost_by_ko"]:
            if pd.notna(p["age"]) and p["age"] >= 35:
                signals.append(("🔴", p["name"],
                                f"**Danger zone: {p['name']}** is 35+ and coming off a KO loss. Fighters in this "
                                f"situation won only **{danger['won'].mean():.0%}** of their next fight "
                                f"and were knocked out again in **{danger['lost_by_ko'].mean():.0%}** "
                                f"({len(danger):,} cases)."))
            else:
                signals.append(("🔴", p["name"],
                                f"**{p['name']}** is coming off a KO loss. Fighters in this situation won "
                                f"**{win_after_ko:.0%}** of their next fight and were knocked out again in "
                                f"**{ko_again:.0%}** (vs {ko_after_win:.0%} after a win)."))

    # Inactivity
    layoff = H[H["days_since_last_fight"] >= 365]
    for p in (A, B):
        if p["days_inactive"] >= 365:
            signals.append(("🟡", p["name"],
                            f"**{p['name']}** had not fought for over a year by {DATA_END:%B %Y} "
                            f"(last fight: {p['last_date']:%b %Y}). After a 1+ year layoff, fighters won "
                            f"**{layoff['won'].mean():.0%}** of their comeback fight ({len(layoff):,} cases) – ring rust?"))

    # Momentum
    for p in (A, B):
        if p["streak"] >= 3:
            signals.append(("🟢", p["name"], f"**{p['name']}** is on a **{p['streak']}-fight win streak**."))
        elif p["streak"] <= -2:
            signals.append(("🟡", p["name"], f"**{p['name']}** has lost **{-p['streak']} fights in a row**."))

    if signals:
        for colour, _, text in signals:
            box = {"🟢": st.success, "🔴": st.error, "🟡": st.warning}[colour]
            box(f"{colour} {text}")
        green_a = sum(1 for c, n, _ in signals if n == A["name"] and c == "🟢") - sum(
            1 for c, n, _ in signals if n == A["name"] and c in "🔴🟡")
        green_b = sum(1 for c, n, _ in signals if n == B["name"] and c == "🟢") - sum(
            1 for c, n, _ in signals if n == B["name"] and c in "🔴🟡")
        if green_a != green_b:
            edge = A["name"] if green_a > green_b else B["name"]
            insight(f"On balance, the historical signals favour **{edge}**. This is not a prediction – "
                    "the Predict tab (coming soon) will combine all factors in a model.")
        else:
            insight("The historical signals are balanced: on paper, this looks like an even fight.")
    else:
        insight("No strong data signals for this matchup – on paper, it looks like an even fight.")

    # ---------- 3. Career staircase ----------
    st.subheader("📈 Career staircase")
    st.caption("Each win is a step up, each loss a step down (UFC wins minus losses). "
               "💥 KO/TKO · 🔒 submission · ⚖️ decision. Hover over a step to see the fight.")

    icons = {"KO/TKO": "💥", "SUB": "🔒", "U-DEC": "⚖️", "S-DEC": "⚖️", "M-DEC": "⚖️"}
    line_colours = {fighter_a: UFC_RED, fighter_b: UFC_BLUE}
    fig = go.Figure()
    for name in (fighter_a, fighter_b):
        c = history_all[history_all["fighter"] == name].sort_values("date").copy()
        c["wins_after"] = c["wins"].fillna(0) + c["won"]
        c["losses_after"] = c["losses"].fillna(0) + (~c["won"])
        c["net"] = c["wins_after"] - c["losses_after"]
        c["icon"] = c["finish"].map(icons).fillna("•")
        c["result"] = np.where(c["won"], "Win", "Loss")
        c["hover"] = (c["result"] + " vs " + c["opponent"] + " (" + c["finish"].fillna("-") + ")<br>"
                      + c["date"].dt.strftime("%d %b %Y") + "<br>Record after: "
                      + c["wins_after"].astype(int).astype(str) + "-" + c["losses_after"].astype(int).astype(str))
        # the staircase line
        fig.add_trace(go.Scatter(x=c["date"], y=c["net"], mode="lines", line_shape="hv",
                                 line=dict(color=line_colours[name], width=3), name=name, hoverinfo="skip"))
        # one marker per fight: green = win, red = loss, with the method icon on top
        fig.add_trace(go.Scatter(x=c["date"], y=c["net"], mode="markers+text", text=c["icon"],
                                 textposition="top center", textfont=dict(size=16),
                                 marker=dict(size=11, color=np.where(c["won"], "#2ECC71", UFC_RED),
                                             line=dict(width=2, color=line_colours[name])),
                                 hovertext=c["hover"], hoverinfo="text", showlegend=False))
    fig.update_layout(height=420, yaxis_title="UFC wins minus losses", xaxis_title="",
                      legend=dict(orientation="h", y=1.08), margin=dict(t=40))
    st.plotly_chart(fig)

    # ---------- 4. How they win ----------
    st.subheader("🥋 How they win")
    col_ha, col_hb = st.columns(2)
    for col, p, colour in ((col_ha, A, UFC_RED), (col_hb, B, UFC_BLUE)):
        methods = pd.DataFrame({"Method": ["KO/TKO", "Submission", "Decision"],
                                "Wins": [p["wins_ko"], p["wins_sub"], p["wins_dec"]]})
        if methods["Wins"].sum() == 0:
            col.info(f"{p['name']} has no UFC wins in the data yet.")
            continue
        pie = px.pie(methods, names="Method", values="Wins", hole=0.55, color="Method",
                     color_discrete_map={"KO/TKO": UFC_RED, "Submission": UFC_GOLD, "Decision": UFC_GREY})
        pie.update_traces(textinfo="percent+value", sort=False)
        pie.update_layout(height=300, showlegend=True, legend=dict(orientation="h", y=-0.1),
                          annotations=[dict(text=f"<b>{p['wins']}</b><br>wins", showarrow=False, font_size=16)],
                          title=dict(text=p["name"], x=0.5, xanchor="center"))
        col.plotly_chart(pie, key=f"pie_{p['name']}")
    main_a = max(("KO/TKO", A["wins_ko"]), ("submission", A["wins_sub"]), ("decision", A["wins_dec"]), key=lambda x: x[1])
    main_b = max(("KO/TKO", B["wins_ko"]), ("submission", B["wins_sub"]), ("decision", B["wins_dec"]), key=lambda x: x[1])
    if A["wins"] and B["wins"]:
        if main_a[0] == main_b[0]:
            insight(f"Both fighters win mostly by **{main_a[0]}** – expect a battle on the same ground.")
        else:
            insight(f"Clash of styles: **{A['name']}** wins mostly by **{main_a[0]}**, "
                    f"**{B['name']}** mostly by **{main_b[0]}**.")


with tab_analyzer:
    fight_analyzer()

with tab_predict:
    st.info("🚧 Coming soon: win probability from a machine learning model vs the bookmakers.")

with tab_simulate:
    st.info("🚧 Coming soon: simulate the fight 10,000 times.")

st.divider()
st.caption("UFC Fight Lab – student project (HAN, Minor Data Driven Decision Making). "
           "Not affiliated with or endorsed by the UFC. Odds are used as an expert benchmark, not as betting advice.")
