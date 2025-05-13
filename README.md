### 🏀 **Project Goal**

To build an NBA Draft prediction model that predicts every Lottery Pick **(picks 1 to 14)**, matching **teams with prospects in the correct order** based on historical team needs, prospect quality, and draft logic.

### 📌 **Methodology**

#### **Overview**

This project aims to simulate the lottery portion of the NBA Draft by evaluating how well each prospect fits the needs of each team and predicting the actual draft order. The simulation will be trained on historical drafts and validated by comparing the predicted draft outcomes to the actual pick-by-pick results.

#### **1. Data Collection**

- **Team-Level Data (per year):**
    
    - Offensive and defensive ratings, league ranks
        
    - Team record and final standings
        
    - Roster depth by position (guards/wings/bigs)
        
    - Average team age or timeline (for development fit)
        
- **Prospect-Level Data:**
    
    - NCAA/International/OTE stats (raw and usage-adjusted)
        
    - Physical attributes (height, wingspan, weight, combine scores)
        
    - Background (college, international, G League, OTE, etc.)
        
    - Mock draft position (for baseline/insider signal)
        
    - Prospect age and classification group (young/old/intl)
        
- **Draft Metadata:**
    
    - Historical lottery orders (picks 1–14)
        
    - Historical actual draft results
        
    - Prospect-to-team matching outcomes (who was drafted by whom)
        

#### **2. Problem Framing**

The model will simulate the draft **sequentially**:

- For each pick, the model will:
    
    1. Evaluate remaining prospects for the current team
        
    2. Score each team–prospect pair
        
    3. Select the top-ranked match
        
    4. Remove the chosen player from the pool
        
    5. Move to the next pick
        

> This framing reflects a **ranking and matching problem**, not a standard classification task.

#### **3. Feature Engineering**

For every **(team, prospect)** pair:

- **Fit-based Features:**
    
    - Positional need × prospect position
        
    - Shooting need × 3PT shooting %
        
    - Defensive need × STL% / BLK%
        
    - Rebounding need × REB%
        
    - Timeline fit = |team avg age – prospect age|
        
    - Mock draft delta = |pick number – avg mock draft position|
        
- **Prospect Evaluation Features:**
    
    - Efficiency metrics (TS%, BPM, WS/40, etc.)
        
    - Archetype cluster (e.g. 3&D wing, traditional big, playmaker guard)
        
    - Usage-adjusted stats (for fair comparison)
        
    - Physical measurements + combine scores
        
- **Team-Level Features:**
    
    - Depth chart by position
        
    - Offensive/defensive rankings
        
    - Team pace, scheme type (optional)
        
    - Draft capital / trade history (optional)
        

#### **4. Modeling Strategy**

- **Primary Approach**:
    
    - Train a **pairwise scoring model** (e.g., Random Forest, Gradient Boosted Trees)
        
    - Input: (team, prospect) features
        
    - Output: suitability score for that team-prospect pairing
        
- **Alternative Models to Try**:
    
    - kNN (team picks player most similar to their past picks)
        
    - XGBoost ranking objective (`rank:pairwise`)
        
    - (Optional later) Feedforward neural network for pair scoring
        
- **Dimensionality Reduction (Optional)**:
    
    - Use **PCA** for prospect similarity visualization or feature compression
        
    - Use **t-SNE** or **UMAP** to explore archetype clusters
        

#### **5. Archetype Classification (Optional but Valuable)**

- Cluster historical NBA players into playstyle archetypes
    
- Classify prospects into closest archetype based on stats
    
- Add archetype as a feature for team–prospect fit evaluation
    

#### **6. Incorporating Insider Knowledge**

- Use average mock draft position as a **proxy feature** for consensus value
    
- Optionally, scrape sentiment from Reddit, Twitter, or media coverage
    
    - Use NLP to quantify hype (e.g., frequency, positivity, buzz)
        

#### **7. Evaluation Strategy**

- For historical drafts, simulate the draft year by year:
    
    - At each pick, predict who gets selected and remove from pool
        
    - Compare full mock draft to actual results
        
- **Evaluation Metrics:**
    
    - **Top-1 Accuracy**: % of picks where predicted player == actual player
        
    - **Top-14 Overlap**: % of predicted lottery players who ended up in the real lottery
        
    - **Average Draft Slot Error**: Mean absolute difference between predicted and real pick
        
    - **Kendall’s Tau / Spearman's ρ**: Rank correlation of predicted vs. actual draft order
        

> 🎯 Target: Achieve a baseline match rate comparable to ESPN/nbadraft.net mocks, not a hard 85% accuracy (which is unrealistic even for pros)

#### **8. Final Application (Post-Lottery)**

- Once 2025 draft lottery order is confirmed:
    
    - Use trained model to simulate pick-by-pick predictions
        
    - Output: Full lottery mock draft from #1 to #14

### ⚠️ Limitations

Despite efforts to build a robust and thoughtful prediction model, several limitations must be acknowledged:

#### 1. **Draft-Day Trades Are Not Accounted For**

- The model assumes that the team making the selection is the one that ultimately **retains the player**, not just announces the pick.
    
- **Draft-day trades** (e.g., team selects a player on behalf of another as part of a trade) are **not predictable** and are excluded from modeling assumptions.
    

#### 2. **Historical Draft Orders Are Manually Adjusted**

- Historical draft orders have been modified to reflect the **actual team that owned each pick** at the time of the draft, including cases involving **unprotected picks**, **pick swaps**, or **prior trades**.
    
- This adjustment is essential for modeling team-specific decision-making, but relies on external verification and **manual data correction**.
    

#### 3. **Limited Training Data**

- Each year only provides 14 lottery picks, meaning that across ~15 seasons, there are fewer than 250 training examples.
    
- This **small sample size** restricts the model’s complexity and may affect generalizability.
    

#### 4. **Team Needs Are Estimated Retrospectively**

- Features such as roster depth or positional needs are **inferred from publicly available statistics** and depth charts, which may not capture nuanced internal evaluations by NBA front offices.
    

#### 5. **Prospect Evaluation Is Inherently Noisy**

- College and international statistics are context-dependent (e.g., team system, pace, competition level).
    
- Even usage-adjusted or advanced stats **cannot fully capture intangibles** like basketball IQ, attitude, or fit with organizational culture.
    

#### 6. **Insider Knowledge Gaps**

- The model lacks access to private workout data, medical reports, agent influence, or off-court factors that heavily influence real-world draft decisions.
    
- While mock drafts and media sentiment may serve as proxies, they do not substitute for insider information.
    

#### 7. **Sequential Simulation Compounds Error**

- Since the model predicts the draft pick-by-pick in order, **errors early in the draft affect later predictions** (e.g., a wrongly selected player is no longer available to the next team).
    
- This mimics real-world constraints but introduces cascading inaccuracies.