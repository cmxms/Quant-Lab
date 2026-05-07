# Quant Lab Data Flow

```mermaid
graph TD
    subgraph "Phase 1: Ingestion & Processing"
        A[Tradier/DataBento API] --> B["update_nq_data.py<br/>(Raw OHLCV)"]
        B --> C["rollover_stitch.py<br/>(Continuous Series)"]
        C --> D["feature_gen.py<br/>(14 Stationary Features)"]
        D --> E1["label_targets_long.py<br/>(Triple Barrier Targets)"]
        D --> E2["label_targets_short.py<br/>(Triple Barrier Targets)"]
    end

    subgraph "Phase 2: Refinement & Selection"
        E1 --> F["mi_processing.py<br/>(Refinement Pipeline)"]
        E2 --> F
        
        F --> F1["Chronological Split<br/>(70/10/20)"]
        F1 --> F2["Mutual Information Scoring<br/>(Mutual Info Classif)"]
        F2 --> F3["The Great Purge<br/>(Pruning Low-Score Features)"]
    end

    subgraph "Final Output"
        F3 --> G1["Final Clean Train.parquet"]
        F3 --> G2["Final Clean Val.parquet"]
        F3 --> G3["Final Clean Holdout.parquet"]
        F3 --> G4["MI Scorecard (Visualization)"]
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style G1 fill:#9f9,stroke:#333,stroke-width:4px
    style G2 fill:#9f9,stroke:#333,stroke-width:4px
    style G3 fill:#9f9,stroke:#333,stroke-width:4px
```

### Tool Summary
1.  **Ingestion (`update_nq_data.py`)**: Fetches raw 1-minute OHLCV data.
2.  **Continuity (`rollover_stitch.py`)**: Resolves contract gaps to create a seamless timeline.
3.  **Features (`feature_gen.py`)**: Transforms raw prices into stationary ML features.
4.  **Labeling (`label_targets_*.py`)**: Generates classification targets via Triple Barrier Method.
5.  **Refinement (`mi_processing.py`)**: Handles the split, calculates importance, and prunes the noise.
