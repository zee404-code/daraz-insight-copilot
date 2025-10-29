# Project Contributions

| Member | ERP | Tasks |
| :--- | :--- | :--- |
| Farah Inayat | 26912 | - Implement the core ML Model Training and Evaluation script. <br> - Develop the Data Ingestion and preprocessing components (including ETL scripts).<br> - Implement and confirm Test Coverage >= 80% (via pytest in the CI pipeline).<br> - Set up ML Model Metrics (e.g., accuracy, precision, latency) for logging and reporting. <br> - Completed the README.md including architecture diagram <br> - Develop the FastAPI Inference API endpoint (/predict). <br> - Complete the API Documentation by adding the Example cURL + JSON schema to the /docs <br> - Configure and document Cloud Integration with at least 2 distinct cloud services and provide annotated screenshots|
| Zehra Ahmed | 26965 | - Configure and integrate the Data/Model Quality Monitoring with the Evidently dashboard for data drift. <br> - Write the foundational documentation: LICENSE file and CODE_OF_CONDUCT.md. <br> - Implement the robust, multi-stage Dockerfile (including non-root user and healthcheck). <br> - Complete the CI/CD Pipeline (.github/workflows/ci.yml) to include Building and Pushing the Docker Image to GHCR <br> - Set up the Prometheus + Grafana stack for Infrastructure Monitoring (collecting at least three metrics like gpu_utilisation) <br> - Implement and enforce all Pre-commit Hooks (including detect-secrets). <br> - Implement Dependency vulnerability scanning via pip-audit. <br> - Configure and document Cloud Integration with at least 2 distinct cloud services and provide annotated screenshots|

## Branch Naming Convention

We follow a conventional commit style for our branch names to keep the repository history clean:

* **`feat/`**: For adding a new feature (e.g., `feat/add-api-endpoint`).
* **`fix/`**: For fixing a bug (e.g., `fix/fix-test-import-error`).
* **`docs/`**: For changes to documentation (e.g., `docs/update-readme`).
* **`ci/`**: For changes to the CI/CD pipeline (e.g., `ci/fix-pipeline-error`).
* **`style/`**: For code formatting and linting fixes (e.g., `style/run-black`).
* **`refactor/`**: For code cleanup that doesn't add features or fix bugs (e.g., `refactor/model-loading`).
* **`infra/`**: For infrastructure changes (e.g., `infra/update-docker-compose`).
