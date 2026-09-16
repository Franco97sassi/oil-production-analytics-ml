# Production-oriented architecture

```mermaid
flowchart LR
    A[Official monthly CSV] --> B[Schema and duplicate validation]
    B --> C[Consecutive well-level lag features]
    C --> D[Train period]
    C --> E[Model-selection period]
    C --> F[Conformal calibration period]
    C --> G[Untouched final holdout]
    D --> H[Candidate pipelines]
    E --> H
    H --> I[Versioned model bundle]
    F --> I
    G --> J[Auditable metrics and drift report]
    I --> K[FastAPI container]
    K --> L[Individual and batch predictions]
    K --> M[Liveness and readiness]
    K --> N[Model lineage]
```

The training process is offline and the API never retrains during a request.
The container receives the model as a read-only mounted artifact. A future
production deployment can replace that mount with a versioned artifact store
without changing the prediction contract.
