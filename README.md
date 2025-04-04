
---

# 🏈 AI-Driven Coaching Assistant for American Football

This project leverages cutting-edge AI technologies to assist professional American football teams with real-time, data-driven strategy formulation. By combining spatio-temporal analysis, deep learning models, and large language models (LLMs), we aim to empower coaches with actionable insights during game time.

---

## 📈 Project Overview

Advancements in technology and data accessibility have revolutionized the way football is played and coached. This project integrates multiple data sources to develop a comprehensive **AI Coaching Assistant** that:

- Analyzes player tracking data.
- Classifies offensive/defensive schemes.
- Predicts play outcomes.
- Retrieves historical play tactics and contextual information.

---

## 🧠 Methodology

### 📊 Data Sources
- **NFL Next Gen Stats**: Positional tracking data of players.
- **Public NFL Competitions**: NFL Big Data Bowl datasets.
- **External News and Stats**: Supplementary metadata for context.

### 🔧 Data Processing
- Cleaning and standardization of play frames.
- Focused extraction on offensive players.
- Temporal segmentation and spatial alignment.

### 🧬 Feature Engineering
- Spatial features: Player formations and relative positioning.
- Temporal features: Player movement across frames.
- Interaction features: Multi-agent dynamics.

### 🏗️ Model Architecture
- **CNN**: Detects spatial formations and patterns.
- **LSTM**: Captures temporal dynamics and movement.
- **Combined CNN-LSTM**: For comprehensive play-type classification and success prediction.

### 🧮 Model Performance (CNN-LSTM)
| Prediction Task                        | Accuracy |
|---------------------------------------|----------|
| Play-action type (T/F)                | 0.80     |
| Run-pass option type (T/F)            | 0.89     |
| Offensive formation (multi-class)     | 0.56     |
| Receiver alignment (multi-class)      | 0.41     |
| Defensive pass coverage (multi-class) | 0.31     |
| General offensive play type           | 0.61     |
| Run concept (multi-class)             | 0.53     |
| Offensive play success (T/F)          | 0.91     |

---

## 🔍 RAG-Enhanced Insights

We integrate **Retrieval-Augmented Generation (RAG)** to:
- Retrieve historical playbooks and scouting reports.
- Utilize play embeddings for enhanced similarity-based retrieval.

**Precision@10**: 98.3%  
(*Fraction of relevant items in the top 10 search results*)

---

## 💻 Application & Interface

### 🎮 Use Cases
- Pre-game preparation and scouting.
- In-game strategy adjustments.
- Post-game review and performance evaluation.

### ⚙️ Components
- Real-time dashboard for visualization.
- Play classification and outcome prediction.
- AI-generated insights and counter-strategy suggestions.

---

## 🏫 Acknowledgments

This project was developed as part of:

**University of Missouri - Kansas City**  
**CS5542: Big Data Analytics and Applications**  
**Professor**: Dr. Yugyung Lee

---

## 📚 References

- Cheong, L. L., Zeng, X., & Tyagi, A. (2021). *Prediction of defensive player trajectories...*. MIT Sloan.
- Ding, N. et al. (2022). *Deep Reinforcement Learning in a Racket Sport...*. IEEE Access.
- Liu, H. et al. (2023). *Automated player identification...*. Scientific Reports.
- Lopez, M. et al. (2024). *NFL Big Data Bowl 2025*. [Kaggle](https://kaggle.com/competitions/nfl-big-data-bowl-2025)
- Raabe, D. et al. (2023). *Graph representation for sports data*. Applied Intelligence.
- Song, H. et al. (2023). *Explainable defense coverage classification...*. MIT Sloan.
- Wang, Z. et al. (2024). *TacticAI: AI assistant for football tactics*. Nature Communications.
- Westerberg, J. (2017). *Deep learning for action classification in football*.
