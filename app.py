import streamlit as st
import requests
import pandas as pd
import pulp
import io
import re

st.set_page_config(page_title="FPL Optimiser", page_icon="⚽", layout="centered")

# --- 1. UI CONFIGURATION (Sidebar) ---
st.sidebar.header("⚙️ Configuration")

# Force Refresh Button with immediate UI reload
if st.sidebar.button("🧹 Force Refresh Data"):
    st.cache_data.clear()
    st.rerun()

PRESET_MANAGERS = {
    "Toby": 68303,
    "Femi": 6179091,
    "James": 7312692,
    "Henry": 6407418,
    "Dara": 4565888,
    "David": 7971367,
    "Other / Custom": None
}

selected_manager = st.sidebar.selectbox("Select Manager", list(PRESET_MANAGERS.keys()))

if selected_manager == "Other / Custom":
    my_team_id = int(st.sidebar.number_input("Enter Custom Team ID", value=68303, step=1))
else:
    my_team_id = PRESET_MANAGERS[selected_manager]
    st.sidebar.caption(f"Team ID: `{my_team_id}`")

gameweek = st.sidebar.number_input("Gameweek", value=1, step=1)
assumed_budget = st.sidebar.number_input("Assumed Budget (£m)", value=100.0, step=0.1)

free_transfers = st.sidebar.slider("Available Free Transfers", 1, 5, 1)
max_transfers = st.sidebar.slider("Max Transfers to Check", 1, 6, 2)
is_wildcard = st.sidebar.checkbox("Playing Wildcard? (15 Transfers)")

st.sidebar.subheader("👑 Premium Captains")
captain_options = st.sidebar.multiselect(
    "Select players to prioritise (+30 FI boost):",
    ['Erling Haaland', 'Haaland', 'Palmer', 'Saka', 'Bruno Fernandes', 'B.Fernandes', 'Salah', 'Isak'],
    default=['Erling Haaland', 'Haaland', 'Palmer', 'Saka', 'Bruno Fernandes', 'B.Fernandes']
)

# Comprehensive Team Normaliser Dictionary
TEAM_NORMALISER = {
    'arsenal': 'ARS', 'ars': 'ARS',
    'aston villa': 'AVL', 'villa': 'AVL', 'avl': 'AVL',
    'bournemouth': 'BOU', 'afc bournemouth': 'BOU', 'bou': 'BOU',
    'brentford': 'BRE', 'bre': 'BRE',
    'brighton': 'BHA', 'brighton & hove albion': 'BHA', 'bha': 'BHA',
    'chelsea': 'CHE', 'che': 'CHE',
    'crystal palace': 'CRY', 'palace': 'CRY', 'cry': 'CRY',
    'everton': 'EVE', 'eve': 'EVE',
    'fulham': 'FUL', 'ful': 'FUL',
    'ipswich': 'IPS', 'ipswich town': 'IPS', 'ips': 'IPS',
    'leicester': 'LEI', 'leicester city': 'LEI', 'lei': 'LEI',
    'leeds': 'LEE', 'leeds united': 'LEE', 'lee': 'LEE',
    'liverpool': 'LIV', 'liv': 'LIV',
    'manchester city': 'MCI', 'man city': 'MCI', 'mci': 'MCI',
    'manchester united': 'MUN', 'man utd': 'MUN', 'man united': 'MUN', 'mun': 'MUN',
    'newcastle': 'NEW', 'newcastle united': 'NEW', 'new': 'NEW',
    'nottingham forest': 'NFO', "nott'm forest": 'NFO', 'nfo': 'NFO',
    'southampton': 'SOU', 'sou': 'SOU',
    'sunderland': 'SUN', 'sun': 'SUN',
    'tottenham': 'TOT', 'tottenham hotspur': 'TOT', 'spurs': 'TOT', 'tot': 'TOT',
    'west ham': 'WHU', 'west ham united': 'WHU', 'whu': 'WHU',
    'wolverhampton wanderers': 'WOL', 'wolves': 'WOL', 'wol': 'WOL'
}

# --- 2. DATA FUNCTIONS ---
@st.cache_data(ttl=900)
def get_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()
    teams_df = pd.DataFrame(data['teams'])
    team_mapping = dict(zip(teams_df['id'], teams_df['short_name']))
    
    players_df = pd.DataFrame(data['elements'])
    columns_to_keep = ['id', 'web_name', 'first_name', 'second_name', 'team', 'element_type', 'now_cost']
    players_df = players_df[columns_to_keep]
    players_df['now_cost'] = players_df['now_cost'] / 10
    players_df['team_code'] = players_df['team'].map(team_mapping)
    
    # Create full name column for matching
    players_df['full_name'] = players_df['first_name'] + ' ' + players_df['second_name']
    return players_df

@st.cache_data(ttl=900)
def get_elevenify_data():
    chart_id = "MmYOs"
    base_url = f"https://datawrapper.dwcdn.net/{chart_id}/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        first_response = requests.get(base_url, headers=headers)
        match = re.search(r'url=(\d+)', first_response.text, re.IGNORECASE)
        latest_version = match.group(1) if match else "1"
        
        csv_url = f"https://datawrapper.dwcdn.net/{chart_id}/{latest_version}/dataset.csv"
        csv_response = requests.get(csv_url, headers=headers)
        if csv_response.status_code == 200:
            return pd.read_csv(io.StringIO(csv_response.text))
    except Exception:
        pass
    return None


def get_public_team_data(team_id, previous_gameweek):
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{previous_gameweek}/picks/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return [pick['element'] for pick in response.json()['picks']]
    return None

# --- 3. OPTIMIZER FUNCTION ---
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
st.title("⚽ FPL Transfer Optimiser")

if st.button("🚀 Run Optimiser", type="primary"):
    with st.spinner("Fetching data and crunching numbers..."):
        fpl_df = get_fpl_data()
        elevenify_df = get_elevenify_data()

        if elevenify_df is not None:
            elevenify_df.columns = elevenify_df.columns.str.strip()

            # Dynamic column detection
            player_col = next((c for c in elevenify_df.columns if 'player' in c.lower()), 'Player')
            team_col = next((c for c in elevenify_df.columns if 'team' in c.lower()), 'Team')
            fi_col = next((c for c in elevenify_df.columns if 'importance' in c.lower()), 'Future Importance')
            cap_col = next((c for c in elevenify_df.columns if 'captain' in c.lower()), None)

            cols_to_extract = [player_col, team_col, fi_col]
            rename_map = {player_col: 'Player', team_col: 'Team', fi_col: 'Future Importance'}
            
            if cap_col:
                cols_to_extract.append(cap_col)
                rename_map[cap_col] = 'Captaincy_Option'

            elevenify_cleaned = elevenify_df[cols_to_extract].rename(columns=rename_map).copy()
            
            if 'Captaincy_Option' not in elevenify_cleaned.columns:
                elevenify_cleaned['Captaincy_Option'] = None

            # Normalise team codes across both datasets
            elevenify_cleaned['team_norm'] = elevenify_cleaned['Team'].astype(str).str.lower().str.strip().map(TEAM_NORMALISER)
            fpl_df['team_norm'] = fpl_df['team_code'].astype(str).str.lower().str.strip().map(TEAM_NORMALISER)

            # Robust multi-pass merge: Web Name -> Full Name -> Second Name
            # Pass 1: Match on web_name + team
            merge1 = pd.merge(
                fpl_df,
                elevenify_cleaned,
                left_on=['web_name', 'team_norm'],
                right_on=['Player', 'team_norm'],
                how='left'
            )

            # Pass 2: Match on full_name + team for remaining missing values
            merge2 = pd.merge(
                merge1,
                elevenify_cleaned,
                left_on=['full_name', 'team_norm'],
                right_on=['Player', 'team_norm'],
                how='left',
                suffixes=('', '_full')
            )
            
            merge2['Future Importance'] = merge2['Future Importance'].combine_first(merge2['Future Importance_full'])
            merge2['Captaincy_Option'] = merge2['Captaincy_Option'].combine_first(merge2['Captaincy_Option_full'])

            merged_df = merge2.drop(columns=[col for col in merge2.columns if '_full' in col or col in ['Player_x', 'Player_y']])
            merged_df['Future Importance'] = pd.to_numeric(merged_df['Future Importance'], errors='coerce').fillna(10)

            merged_df['Captaincy_Boost'] = (
                merged_df['web_name'].isin(captain_options) | 
                merged_df['full_name'].isin(captain_options) |
                merged_df['Captaincy_Option'].notna()
            ).astype(int)

            my_current_team_ids = get_public_team_data(my_team_id, gameweek)

            if my_current_team_ids:

                baseline_ids, base_fi = optimize_squad(merged_df, my_current_team_ids, assumed_budget, exact_transfers=0)
                st.subheader("📋 Current Squad")
                squad_df = merged_df[merged_df['id'].isin(my_current_team_ids)].copy()

                squad_display = squad_df[['web_name', 'team_code', 'now_cost', 'Future Importance', 'Captaincy_Boost']].copy()
                squad_display.rename(columns={
                    'web_name': 'Player', 
                    'team_code': 'Team', 
                    'now_cost': 'Price (£m)', 
                    'Captaincy_Boost': 'Cap Boost'
                }, inplace=True)

                squad_display.sort_values(by='Future Importance', ascending=False, inplace=True)
                squad_display.reset_index(drop=True, inplace=True)

                st.dataframe(squad_display, use_container_width=True)
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
        else:
            st.error("🚨 Failed to fetch the dataset. Datawrapper might be down or the ID is incorrect.")