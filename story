Building on the previously developed intent identification logic, which parses free-text user submissions to detect primary and secondary intents, this story focuses on exposing that capability via a dedicated Intent Finder API.

The API will allow external systems or downstream services to submit text and receive structured outputs identifying all embedded intents along with their descriptions and requested data elements. This provides a standardized interface to consume intent information programmatically, enabling further processing such as CUI mapping, nature/sub-nature decomposition, and downstream reasoning pipelines.

The API must handle multiple intents within a single input, preserve contextual relationships, and maintain consistency with the intent definitions and decomposition logic developed in the predecessor story.

Business Value: Providing an API for intent identification enables scalable and reusable integration across applications, supports automation of downstream data extraction and normalization, reduces manual interpretation effort, and ensures consistent semantic understanding of user-submitted clinical text.




AC :API accepts input text (single paragraph or multi-paragraph) and optional metadata parameters (e.g., source, timestamp).

API returns a structured list of intents, where each intent includes:

Intent name or identifier

Description of the intent

Requested data/information elements

Optional sub-components or sub-natures (if available)

Supports multiple intents per input, maintaining correct ordering and contextual grouping.

Output format is standardized (JSON schema) for downstream consumption.

API is capable of handling common variations in input text, including spelling, abbreviations, and sentence structure differences.

Logging includes:

Input received

Intents detected

If applicable, any unmapped or ambiguous cases flagged for review

Default behavior maintains prior intent identification accuracy; no intents are lost in translation to the API output.

Unit tests verify:

Correct detection of single and multiple intents from representative sample inputs

Accuracy of description and requested data fields

Stability of output schema

API is deployed to AIF only.

Documentation (README & SWAGGER) includes:

API endpoints, input/output schema, and sample requests/responses

Guidelines for integration with downstream pipelines (CUI mapping, clustering, summarization)

API Capabilities page and a subpage is updated as appropriate.
