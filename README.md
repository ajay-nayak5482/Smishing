```mermaid
graph TD
    subgraph Phase 1: Data Ingestion & Initial Preparation
        A[Raw SMS Datasets] --> B(Data Ingestion & Unification)
        B --> C(Data Cleaning & Initial Feature Engineering)
        C --> D{Cleaned & Engineered Dataset}
    end

    subgraph Phase 2: Data Preprocessing for Model Input
        D --> E(Data Splitting & Final Transformation)
        E --> F1[Text Input: IDs & Mask]
        E --> F2[Structured Input]
        E --> F3[Target Labels]
    end

    subgraph Phase 3: Initial Hybrid Model Training - Baseline Defender
        F1 & F2 & F3 --> G(Hybrid Model Architecture Definition)
        G --> H[Initial Hybrid Model Architecture]
        H --> I(Initial Model Training)
        I --> J{Trained Baseline Model}
        I --> K{Initial Training History}
    end

    subgraph Phase 4: Iterative Adversarial Training Loop - Robustness Enhancement
        J -- Start Loop - Current Model --> L(Iteration X)
        L --> M(Adversarial Example Generation)
        M --> N{New Adversarial Examples}
        N & P_data[Original Training Data] --> O(Data Augmentation)
        O --> P(Model Fine-tuning)
        P --> Q{Updated Robust Model}
        Q & R_data[Original Test Data] --> S(Robustness Evaluation)
        S --> T{Clean & Adversarial Accuracy History}
        Q -- Loop Back --> L
    end

    subgraph Phase 5: Final Model Export & Deployment Preparation
        Q -- Final Iteration --> U(Final Model Export & Conversion)
        U --> V[Final Robust Model - .keras]
        U --> W[Final Robust Model - .tflite]
    end

    %% Node Styling
    style A fill:#FFDDC1,stroke:#E67E22,stroke-width:2px;
    style B fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style C fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style D fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;

    style E fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style F1 fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
    style F2 fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
    style F3 fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;

    style G fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style H fill:#FFDDC1,stroke:#E67E22,stroke-width:2px;
    style I fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style J fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
    style K fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;

    style L fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style M fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style N fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
    style O fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style P fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style Q fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
    style S fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style T fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
    style P_data fill:#FFDDC1,stroke:#E67E22,stroke-width:2px;
    style R_data fill:#FFDDC1,stroke:#E67E22,stroke-width:2px;

    style U fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    style V fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
    style W fill:#ADD8E6,stroke:#3498DB,stroke-width:2px;
