import streamlit as st
import requests
import pandas as pd
import pulp
import io
import plotly.express as px

st.set_page_config(page_title="FPL Optimiser", page_icon="⚽", layout="centered")

# --- NEW FUNCTION: Fetch Latest GW ---
@st.cache_data(ttl=3600) # Caches for 1 hour to keep the app fast
def get_latest_gw():
    try:
        data = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        for event in data.get('events', []):
            if event.get('is_current'):
                return event['id']
    except:
        pass
    return 1 # Fallback to GW1 if the API fails or it is pre-season

# --- 1. UI CONFIGURATION (Sidebar) ---
st.sidebar.header("⚙️ Configuration")

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

latest_gw = get_latest_gw()
gameweek = st.sidebar.number_input("Gameweek", value=latest_gw, step=1)

prioritise_xi = st.sidebar.checkbox("Prioritise Starting 11", value=True, help="Weighs starting players at 100% FI and bench players at 10% FI.")

free_transfers = st.sidebar.slider("Available Free Transfers", 1, 5, 1)
max_transfers = st.sidebar.slider("Max Transfers to Check", 1, 6, 2)
is_wildcard = st.sidebar.checkbox("Playing Wildcard? (15 Transfers)")
assumed_bank = st.sidebar.number_input("Money in Bank (£m)", value=0.0, step=0.1, help="Added to your total squad value.")

st.sidebar.subheader("👑 Premium Captains")
captain_options = st.sidebar.multiselect(
    "Select players to prioritise (+80 FI boost):",
    ['Palmer', 'Saka', 'Isak', 'B.Fernandes', 'Haaland'],
    default=['Palmer', 'Saka', 'Isak', 'B.Fernandes', 'Haaland']
)

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
    columns_to_keep = ['id', 'web_name', 'first_name', 'second_name', 'team', 'element_type', 'now_cost', 'chance_of_playing_next_round']
    players_df = players_df[columns_to_keep]
    players_df['now_cost'] = players_df['now_cost'] / 10
    
    players_df['chance_of_playing_next_round'] = pd.to_numeric(players_df['chance_of_playing_next_round'], errors='coerce').fillna(100)
    
    players_df['team_code'] = players_df['team'].map(team_mapping)
    players_df['full_name'] = players_df['first_name'] + ' ' + players_df['second_name']
    return players_df

@st.cache_data(ttl=60)
def get_model_data():
    csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQW_3C7uC0pY0_CPCGcqWiOKq7t2esNSmmejclfaE5dTgAfxLsec_dnJ-m40qJk7TWxjSRcN7KMivZm/pub?output=csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(csv_url, headers=headers)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            return df, "Google Sheet"
    except Exception:
        pass
    return None, None

def get_public_team_data(team_id, previous_gameweek):
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{previous_gameweek}/picks/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return [pick['element'] for pick in response.json()['picks']]
    return None

# --- 3. OPTIMIZER FUNCTION ---
def optimize_squad(merged_df, current_team_ids, budget, exact_transfers, prioritise_xi=True):
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
    
    transfers_in = pulp.lpSum([squad_vars[p] for p in players if merged_df.loc[p, 'id'] not in current_team_ids])
    if exact_transfers >= 15:
        prob += transfers_in <= 15
    else:
        prob += transfers_in == exact_transfers

    for p in players:
        prob += captain_vars[p] <= squad_vars[p]
        
        is_injured = merged_df.loc[p, 'chance_of_playing_next_round'] == 0
        is_unlisted = not merged_df.loc[p, 'In_Sheet']
        if (is_unlisted or is_injured) and merged_df.loc[p, 'id'] not in current_team_ids:
            prob += squad_vars[p] == 0
    
    objective = []
    
    if prioritise_xi:
        starting_vars = pulp.LpVariable.dicts("Starter", players, cat='Binary')
        
        prob += pulp.lpSum([starting_vars[p] for p in players]) == 11
        prob += pulp.lpSum([starting_vars[p] for p in players if merged_df.loc[p, 'element_type'] == 1]) == 1
        prob += pulp.lpSum([starting_vars[p] for p in players if merged_df.loc[p, 'element_type'] == 2]) >= 3
        prob += pulp.lpSum([starting_vars[p] for p in players if merged_df.loc[p, 'element_type'] == 2]) <= 5
        prob += pulp.lpSum([starting_vars[p] for p in players if merged_df.loc[p, 'element_type'] == 3]) >= 2
        prob += pulp.lpSum([starting_vars[p] for p in players if merged_df.loc[p, 'element_type'] == 3]) <= 5
        prob += pulp.lpSum([starting_vars[p] for p in players if merged_df.loc[p, 'element_type'] == 4]) >= 1
        prob += pulp.lpSum([starting_vars[p] for p in players if merged_df.loc[p, 'element_type'] == 4]) <= 3

        for p in players:
            prob += starting_vars[p] <= squad_vars[p]
            prob += captain_vars[p] <= starting_vars[p]
            
            base_fi = merged_df.loc[p, 'Future Importance']
            cap_boost = merged_df.loc[p, 'Captaincy_Boost']
            objective.append(
                starting_vars[p] * base_fi + 
                (squad_vars[p] - starting_vars[p]) * (base_fi * 0.1) + 
                captain_vars[p] * (base_fi + (cap_boost * 80))
            )
    else:
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
        elevenify_df, source_name = get_model_data() 
        
        if elevenify_df is not None:
            st.caption(f"Loaded Data Source: `{source_name}`")
            elevenify_df.columns = elevenify_df.columns.str.strip()

            col_lower = {c.lower().strip(): c for c in elevenify_df.columns}
            
            player_col = next((col_lower[c] for c in col_lower if 'player' in c), elevenify_df.columns[0])
            team_col = next((col_lower[c] for c in col_lower if 'team' in c or 'club' in c), None)
            pos_col = next((col_lower[c] for c in col_lower if 'pos' in c), None)
            fi_col = next((col_lower[c] for c in col_lower if 'importance' in c or 'future' in c), elevenify_df.columns[-1])
            cap_col = next((col_lower[c] for c in col_lower if 'captain' in c), None)

            cols_to_extract = [player_col, fi_col]
            rename_map = {player_col: 'Player', fi_col: 'Future Importance'}

            if team_col:
                cols_to_extract.append(team_col)
                rename_map[team_col] = 'Team'
            if pos_col:
                cols_to_extract.append(pos_col)
                rename_map[pos_col] = 'Position'
            if cap_col:
                cols_to_extract.append(cap_col)
                rename_map[cap_col] = 'Captaincy_Option'

            elevenify_cleaned = elevenify_df[cols_to_extract].rename(columns=rename_map).copy()
            
            if 'Team' not in elevenify_cleaned.columns:
                elevenify_cleaned['Team'] = "Unknown"
            if 'Position' not in elevenify_cleaned.columns:
                elevenify_cleaned['Position'] = "MID"

            elevenify_cleaned['Player_lower'] = elevenify_cleaned['Player'].astype(str).str.lower().str.strip()
            elevenify_cleaned['team_norm'] = elevenify_cleaned['Team'].astype(str).str.lower().str.strip().map(TEAM_NORMALISER)
            elevenify_cleaned['pos_norm'] = elevenify_cleaned['Position'].astype(str).str.upper().str.strip()

            pos_mapping = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            fpl_df['pos_norm'] = fpl_df['element_type'].map(pos_mapping)
            fpl_df['team_norm'] = fpl_df['team_code'].astype(str).str.lower().str.strip().map(TEAM_NORMALISER)
            
            fpl_df['web_name_lower'] = fpl_df['web_name'].astype(str).str.lower().str.strip()
            fpl_df['first_name_lower'] = fpl_df['first_name'].astype(str).str.lower().str.strip()
            fpl_df['second_name_lower'] = fpl_df['second_name'].astype(str).str.lower().str.strip()
            fpl_df['full_name_lower'] = fpl_df['first_name_lower'] + ' ' + fpl_df['second_name_lower']

            sheet_dict = dict(zip(elevenify_cleaned['Player_lower'], elevenify_cleaned['Future Importance']))

            def lookup_fi(row):
                web = str(row['web_name_lower'])
                full = str(row['full_name_lower'])

                if full in sheet_dict:
                    return sheet_dict[full]
                if web in sheet_dict:
                    return sheet_dict[web]
                        
                return -999

            merged_df = fpl_df.copy()
            merged_df['Future Importance'] = merged_df.apply(lookup_fi, axis=1)
            merged_df['Future Importance'] = pd.to_numeric(merged_df['Future Importance'], errors='coerce').fillna(-999)
            merged_df['In_Sheet'] = merged_df['Future Importance'] != -999
            merged_df.loc[~merged_df['In_Sheet'], 'Future Importance'] = 10

            merged_df['Status'] = merged_df['chance_of_playing_next_round'].apply(
                lambda x: '✅' if x == 100 else ('❌' if x == 0 else '⚠️')
            )

            has_cap_col = 'Captaincy_Option' in elevenify_cleaned.columns
            if has_cap_col and 'Player_lower' in elevenify_cleaned.columns:
                cap_sheet_dict = dict(zip(elevenify_cleaned['Player_lower'], elevenify_cleaned['Captaincy_Option']))
                merged_df['sheet_cap'] = merged_df['web_name_lower'].map(cap_sheet_dict)
                captain_flag = merged_df['sheet_cap'].notna()
            else:
                captain_flag = False

            merged_df['Captaincy_Boost'] = (
                merged_df['web_name'].isin(captain_options) | 
                captain_flag
            ).astype(int)

            ipswich_palmer_mask = (merged_df['web_name'].str.lower() == 'palmer') & (merged_df['team_code'] == 'IPS')
            merged_df.loc[ipswich_palmer_mask, 'Future Importance'] = 10
            merged_df.loc[ipswich_palmer_mask, 'Captaincy_Boost'] = 0
            merged_df.loc[ipswich_palmer_mask, 'In_Sheet'] = False

            my_current_team_ids = get_public_team_data(my_team_id, gameweek)

            if my_current_team_ids:
                squad_current_value = merged_df[merged_df['id'].isin(my_current_team_ids)]['now_cost'].sum()
                dynamic_budget = squad_current_value + assumed_bank
                
                baseline_ids, base_fi = optimize_squad(merged_df, my_current_team_ids, dynamic_budget, exact_transfers=0, prioritise_xi=prioritise_xi)
                
                st.subheader(f"📋 Current Squad (Total Value: £{squad_current_value:.1f}m)")
                squad_df = merged_df[merged_df['id'].isin(my_current_team_ids)].copy()

                squad_display = squad_df[['Status', 'web_name', 'team_code', 'now_cost', 'Future Importance', 'Captaincy_Boost']].copy()
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
                
                # 1. Run optimization across all requested transfer counts
                scenario_results = []
                for moves in scenarios:
                    rec_ids, new_fi = optimize_squad(merged_df, my_current_team_ids, dynamic_budget, exact_transfers=moves, prioritise_xi=prioritise_xi)
                    
                    if rec_ids:
                        fi_diff = new_fi - base_fi
                        hit_penalty = max(0, (moves - free_transfers) * 8) 
                        net_fi_diff = fi_diff - hit_penalty
                        avg_gain = net_fi_diff / moves if moves > 0 else 0
                        
                        out_ids = [pid for pid in my_current_team_ids if pid not in rec_ids]
                        in_ids = [pid for pid in rec_ids if pid not in my_current_team_ids]
                        out_names = [merged_df.loc[merged_df['id'] == pid, 'web_name'].values[0] for pid in out_ids]
                        in_names = [merged_df.loc[merged_df['id'] == pid, 'web_name'].values[0] for pid in in_ids]
                        
                        scenario_results.append({
                            'moves': moves,
                            'new_fi': new_fi,
                            'net_fi_diff': net_fi_diff,
                            'avg_gain': avg_gain,
                            'out_names': out_names,
                            'in_names': in_names
                        })

                # 2. Dynamic Advice Banner based on available Free Transfers
                if not is_wildcard and scenario_results:
                    valid_ft_moves = [r for r in scenario_results if r['moves'] <= free_transfers]
                    efficient_moves = [r for r in valid_ft_moves if r['avg_gain'] >= 10.0]
                    
                    if efficient_moves:
                        best_move = efficient_moves[-1]
                        rec_moves = best_move['moves']
                        rolled_fts = free_transfers - rec_moves
                        
                        if rolled_fts > 0:
                            st.info(
                                f"💡 **Recommendation: Make {rec_moves} transfer(s) and roll {rolled_fts} FT.**\n\n"
                                f"- **{rec_moves} Move(s):** Net **+{best_move['net_fi_diff']:.1f} FI** (Average **+{best_move['avg_gain']:.1f} FI / transfer**).\n"
                                f"- Additional transfers drop below the +10.0 FI/transfer efficiency threshold. "
                                f"Rolling will give you **{min(5, rolled_fts + 1)} FTs** next gameweek."
                            )
                        else:
                            st.success(
                                f"🚀 **Recommendation: Use all {free_transfers} Free Transfers.**\n\n"
                                f"All {free_transfers} moves meet your efficiency threshold (Average: **+{best_move['avg_gain']:.1f} FI / move**, Net: **+{best_move['net_fi_diff']:.1f} FI**)."
                            )
                    else:
                        if free_transfers < 5:
                            best_1 = next((r for r in scenario_results if r['moves'] == 1), None)
                            gain_text = f"+{best_1['net_fi_diff']:.1f} FI" if best_1 else "low return"
                            st.info(
                                f"💡 **Recommendation: Roll your transfer (0 moves).**\n\n"
                                f"The best 1-transfer move only nets {gain_text} (< 10.0 threshold). "
                                f"Rolling lets you bank **{free_transfers + 1} FTs** next gameweek."
                            )
                        else:
                            st.warning(
                                "⚠️ **Bank Cap Alert:** You are at the maximum of 5 banked FTs. "
                                "Make at least 1 move so you don't burn an incoming transfer next week."
                            )

                # 3. Render Individual Option Expanders
                for res in scenario_results:
                    if res['net_fi_diff'] > 0 or is_wildcard:
                        moves = res['moves']
                        efficiency_badge = f"(Avg: +{res['avg_gain']:.1f}/move)" if moves > 1 else ""
                        
                        with st.expander(f"Option: {moves} Transfer(s) | Net FI Gain: +{res['net_fi_diff']:.1f} {efficiency_badge}"):
                            if moves <= free_transfers and res['avg_gain'] < 10.0 and not is_wildcard:
                                st.caption("⚠️ *Below the +10.0 FI/transfer threshold. Consider rolling remaining FT(s).*")
                            st.write(f"**🔴 SELL:** {', '.join(res['out_names']) if res['out_names'] else 'None'}")
                            st.write(f"**🟢 BUY:** {', '.join(res['in_names']) if res['in_names'] else 'None'}")
                                
                # Feature D: Plotly Visualisation
                st.divider()
                st.subheader("📊 Price vs. Future Importance")
                plot_df = merged_df[merged_df['In_Sheet'] & (merged_df['Future Importance'] > 10)].copy()
                if not plot_df.empty:
                    fig = px.scatter(
                        plot_df, 
                        x='now_cost', 
                        y='Future Importance', 
                        color='pos_norm',
                        hover_name='web_name',
                        hover_data={'team_code': True, 'now_cost': True, 'Future Importance': True, 'pos_norm': False, 'Status': True},
                        labels={'now_cost': 'Price (£m)', 'Future Importance': 'FI', 'pos_norm': 'Position'},
                        title="Identify High-Value Budget Targets"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature E: Multi-Gameweek "Mini-Wildcard" Planner
                st.divider()
                st.subheader("📅 Mini-Wildcard Target Planner")
                st.write("FPL allows banking up to 5 Free Transfers. Check what a massive free overhaul looks like if you save up your moves:")
                
                current_ft = free_transfers
                max_rollable = 5
                
                if current_ft >= max_rollable:
                    st.info("You already have the maximum 5 Free Transfers banked! Use the regular transfer options above to plan your overhaul.")
                else:
                    planner_scenarios = range(current_ft + 1, max_rollable + 1)
                    tabs = st.tabs([f"Roll to {ft} FTs (Wait {ft - current_ft} GWs)" for ft in planner_scenarios])
                    
                    for i, target_ft in enumerate(planner_scenarios):
                        with tabs[i]:
                            mw_ids, mw_fi = optimize_squad(merged_df, my_current_team_ids, dynamic_budget, exact_transfers=target_ft, prioritise_xi=prioritise_xi)
                            
                            if mw_ids:
                                mw_diff = mw_fi - base_fi
                                out_ids = [pid for pid in my_current_team_ids if pid not in mw_ids]
                                in_ids = [pid for pid in mw_ids if pid not in my_current_team_ids]
                                
                                out_names = [merged_df.loc[merged_df['id'] == pid, 'web_name'].values[0] for pid in out_ids]
                                in_names = [merged_df.loc[merged_df['id'] == pid, 'web_name'].values[0] for pid in in_ids]
                                
                                st.success(f"**Target Squad Future Importance:** {mw_fi:.1f} (Net Gain: +{mw_diff:.1f} FI for 0 hits)")
                                st.write(f"**🔴 SELL:** {', '.join(out_names) if out_names else 'None'}")
                                st.write(f"**🟢 BUY:** {', '.join(in_names) if in_names else 'None'}")
                                
                                with st.expander("View Projected Target Squad"):
                                    mw_squad_df = merged_df[merged_df['id'].isin(mw_ids)].copy()
                                    mw_display = mw_squad_df[['Status', 'web_name', 'team_code', 'now_cost', 'Future Importance', 'Captaincy_Boost']].copy()
                                    mw_display.rename(columns={'web_name': 'Player', 'team_code': 'Team', 'now_cost': 'Price (£m)', 'Captaincy_Boost': 'Cap Boost'}, inplace=True)
                                    mw_display.sort_values(by='Future Importance', ascending=False, inplace=True)
                                    st.dataframe(mw_display.reset_index(drop=True), use_container_width=True)

        else:
            st.error("🚨 Failed to fetch the Google Sheet CSV link.")