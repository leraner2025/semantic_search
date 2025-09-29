contextual-api/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI entry point
│   ├── pipeline.py              # DocAI + embedding logic
│   ├── models.py                # Request/response schemas
│   └── utils.py                 # Embedding + similarity helpers
│
├── config/
│   └── settings.py              # Project ID, location, table names
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── tests/
    └── test_pipeline.py
