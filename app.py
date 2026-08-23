import streamlit as st
import requests
import pandas as pd
import pulp
import io

st.set_page_config(page_title="FPL Optimizer", page_icon="⚽", layout="centered")

# --- 1. UI CONFIGURATION (Sidebar) ---
st.sidebar.header("⚙️ Configuration")
my_team_id = st.sidebar.number_input("Team ID", value=68303, step=1)
gameweek = st.sidebar.number_input("Gameweek", value=1, step=1)
assumed_budget = st.sidebar.number_input("Assumed Budget (£m)", value=100.0, step=0.1)

# Dynamic Inputs
free_transfers = st.sidebar.slider("Available Free Transfers", 1, 5, 1)
max_transfers = st.sidebar.slider("Max Transfers to Check", 1, 6, 2)
is_wildcard = st.sidebar.checkbox("Playing Wildcard? (15 Transfers)")

# Captaincy Multi-Select
st.sidebar.subheader("👑 Premium Captains")
captain_options = st.sidebar.multiselect(
    "Select players to prioritize (+80 FI boost):",
    ['Haaland', 'Palmer', 'Saka', 'B.Fernandes', 'Salah', 'Isak'],
    default=['Haaland', 'Palmer', 'Saka', 'B.Fernandes']
)

name_mappings = {
    "B.Fernandes": "B.Fernandes",
    "Haaland": "Haaland",
    "Szoboszlai": "Szoboszlai",
    "Groß": "Groß",
    "N.Williams": "N.Williams",
    "Calvert-Lewin": "Calvert-Lewin"
}

# --- 2. CACHED DATA FUNCTIONS ---
@st.cache_data(ttl=3600) # Caches data for 1 hour so the UI is snappy
def get_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()
    teams_df = pd.DataFrame(data['teams'])
    team_mapping = dict(zip(teams_df['id'], teams_df['short_name']))
    players_df = pd.DataFrame(data['elements'])
    columns_to_keep = ['id', 'web_name', 'team', 'element_type', 'now_cost']
    players_df = players_df[columns_to_keep]
    players_df['now_cost'] = players_df['now_cost'] / 10
    players_df['team_code'] = players_df['team'].map(team_mapping)
    return players_df

@st.cache_data(ttl=3600)
def get_elevenify_data():
    base_dw_url = "https://datawrapper.dwcdn.net/MmYOs/20/" # REPLACE XXXXX
    csv_url = base_dw_url + "dataset.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(csv_url, headers=headers)
    if response.status_code == 200:
        return pd.read_csv(io.StringIO(response.text))
    return None

def get_public_team_data(team_id, previous_gameweek):
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{previous_gameweek}/picks/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return [pick['element'] for pick in response.json()['picks']]
    return None

# --- 3. OPTIMIZER FUNCTION ---
# (Paste your EXACT optimize_squad function here)
def optimize_squad(merged_df, current_team_ids, budget, exact_transfers):
    prob = pulp.LpProblem("FPL_Optimizer", pulp.LpMaximize)
    players = merged_df.index.tolist()
    
    squad_vars = pulp.LpVariable.dicts("Squad", players, cat='Binary')
    captain_vars = pulp.LpVariable.dicts("Captain", players, cat='Binary')
    
    prob += pulp.lpSum([squad_vars[p] for p in players]) == 15
    prob += pulp.lpSum([squad_vars[p] for p in players if merged_df.loc[p, 'element_type'] == 1]) == 2
    prob += pulp.lpSum([squad_vars[p] for p in players if merged_df.loc[p, 'element_type'] == 2]) == 5
    prob += pulp.lpSum([squad_vars[p] for p in players if merged_df.loc[p, 'element_type'] == 3]) == 5
    prob += pulp.lpSum([squad_vars[p] for p in players if merged_df.loc[p, 'element_type'] == 4]) == 3
    
    for team in merged_df['team_code'].unique():
        prob += pulp.lpSum([squad_vars[p] for p in players if merged_df.loc[p, 'team_code'] == team]) <= 3
        
    prob += pulp.lpSum([squad_vars[p] * merged_df.loc[p, 'now_cost'] for p in players]) <= budget
    
    prob += pulp.lpSum([captain_vars[p] for p in players]) == 1
    for p in players:
        prob += captain_vars[p] <= squad_vars[p]
        
    transfers_in = pulp.lpSum([squad_vars[p] for p in players if merged_df.loc[p, 'id'] not in current_team_ids])
    if exact_transfers >= 15:
        prob += transfers_in <= 15
    else:
        prob += transfers_in == exact_transfers
    
    objective = []
    for p in players:
        base_fi = merged_df.loc[p, 'Future Importance']
        cap_boost = merged_df.loc[p, 'Captaincy_Boost']
        objective.append(squad_vars[p] * base_fi + captain_vars[p] * (base_fi + (cap_boost * 80)))
        
    prob += pulp.lpSum(objective)
    prob.solve(pulp.PULP_CBC_CMD(msg=False)) 
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        selected_ids = [merged_df.loc[p, 'id'] for p in players if squad_vars[p].varValue == 1]
        total_fi = pulp.value(prob.objective)
        return selected_ids, total_fi
    else:
        return None, 0

# --- 4. MAIN APP DASHBOARD ---
st.title("⚽ FPL Transfer Optimizer")

if st.button("🚀 Run Optimizer", type="primary"):
    with st.spinner("Fetching data and crunching numbers..."):
        fpl_df = get_fpl_data()
        elevenify_df = get_elevenify_data()

        if elevenify_df is not None:
            elevenify_cleaned = elevenify_df[['Player', 'Team', 'Future Importance', 'GW2 Captaincy Option?']].copy()
            fpl_df['merge_name'] = fpl_df['web_name'].replace(name_mappings)
            # --- THE FIX ---
            # Restoring the strict dual-match to prevent name collisions
            merged_df = pd.merge(
                fpl_df, 
                elevenify_cleaned, 
                left_on=['merge_name', 'team_code'], 
                right_on=['Player', 'Team'], 
                how='left'
            )
            merged_df['Future Importance'] = merged_df['Future Importance'].fillna(10)
            
            merged_df['Captaincy_Boost'] = (
                merged_df['web_name'].isin(captain_options) | 
                merged_df['merge_name'].isin(captain_options) |
                merged_df['GW2 Captaincy Option?'].notna()
            ).astype(int)

            my_current_team_ids = get_public_team_data(my_team_id, gameweek)

            if my_current_team_ids:
                baseline_ids, base_fi = optimize_squad(merged_df, my_current_team_ids, assumed_budget, exact_transfers=0)
                st.success(f"**Baseline Squad Future Importance:** {base_fi:.1f}")
                
                scenarios = [15] if is_wildcard else range(1, max_transfers + 1)
                
                for moves in scenarios:
                    rec_ids, new_fi = optimize_squad(merged_df, my_current_team_ids, assumed_budget, exact_transfers=moves)
                    
                    if rec_ids:
                        fi_diff = new_fi - base_fi
                        hit_penalty = max(0, (moves - free_transfers) * 4) 
                        net_fi_diff = fi_diff - hit_penalty
                        
                        if net_fi_diff > 0 or is_wildcard:
                            out_ids = [pid for pid in my_current_team_ids if pid not in rec_ids]
                            in_ids = [pid for pid in rec_ids if pid not in my_current_team_ids]
                            
                            out_names = [fpl_df.loc[fpl_df['id'] == pid, 'web_name'].values[0] for pid in out_ids]
                            in_names = [fpl_df.loc[fpl_df['id'] == pid, 'web_name'].values[0] for pid in in_ids]
                            
                            with st.expander(f"Option: {moves} Transfer(s) | Net FI Gain: +{net_fi_diff:.1f}"):
                                st.write(f"**🔴 SELL:** {', '.join(out_names) if out_names else 'None'}")
                                st.write(f"**🟢 BUY:** {', '.join(in_names) if in_names else 'None'}")