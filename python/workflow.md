```mermaid
    graph LR
    subgraph Input Layers
        A[input_ids] --> B
        C[attention_mask] --> B
        D[structured_features_input] --> E
    end

    subgraph Textual Branch
        B(TFDistilBertModel) --> F[CLS Token Embedding]
        F -- Dropout (0.2) --> G
    end

    subgraph Structured Features Branch
        E(Dense - 128, ReLU) -- Dropout (0.2) --> H
        H --> I(Dense - 64, ReLU)
        I -- Dropout (0.2) --> J
    end

    subgraph Fusion & Output
        G & J -- Concatenate --> K[Combined Features]
        K --> L(Dense - 64, ReLU)
        L -- Dropout (0.3) --> M
        M --> N(Dense - 1, Sigmoid)
        N --> O[Phishing Probability]
    end

    %% Node Styling (consistent with previous workflow)
    style A fill:#FFDDC1,stroke:#E67E22,stroke-width:2px;
    style C fill:#FFDDC1,stroke:#E67E22,stroke-width:2px;
    style D fill:#FFDDC1,stroke:#E67E22,stroke-width:2px;

    style B fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style F fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
    style G fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;

    style E fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style H fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
    style I fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style J fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;

    style K fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
    style L fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style M fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
    style N fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style O fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
