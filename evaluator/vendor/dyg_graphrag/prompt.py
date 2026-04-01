"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS["dynamic_QA"] = """

You are an AI assistant that answers questions based on both temporal event sequences and relevant text chunks.

# TASK
Answer the question using both the provided events and text chunks. The events show temporal relationships, while the text chunks provide additional context and details.

# TEMPORAL RELEVANCE GUIDELINE
**Prioritize information from time periods that align with the question's temporal scope.** If a question asks about a specific time period, focus on events and information that occurred within or near that timeframe. Consider whether nearby temporal information provides relevant context or background that could help answer the question.

# ANALYSIS APPROACH
1. **Query Semantics Understanding**: Analyze the semantic intent of the question:
   - **Existential queries**: Questions about what exists/happens at a specific time
   - **Continuity queries**: Questions about ongoing states, processes, or relationships
   - **Boundary queries**: Questions about beginnings, endings, or transitions
   - **Aggregate queries**: Questions requiring synthesis across multiple time points

2. **Temporal Scope Identification**: Identify the exact time constraints in the question:
   - **Subject Consistency Check**: Verify that the event's subject matches the question's focus
   - Extract specific dates, time periods, or temporal references
   - Mark any temporal qualifiers (e.g., "during", "between", "before", "after")

3. **Evidence Time Filtering**: Before using any event or chunk as evidence:
   - **Assess temporal relevance**: Evaluate whether each piece of evidence falls within or near the question's time scope
   - **Consider contextual value**: Include information from nearby time periods if it provides essential background
   - **Prioritize temporal proximity**: Give preference to evidence closer to the target time period

4. **Event Analysis**: Review the temporal sequence of events:
   - **Temporal Ordering**: Analyze when events happen and their chronological relationships
   - **State Persistence and Change**: 
     - **Persistent states**: Properties or conditions that continue until explicitly changed
     - **Instantaneous events**: Actions that happen at specific moments
     - **Processes**: Ongoing activities with duration
     - **Transitions**: Changes from one state to another
   - **Office/Role Incumbency Reasoning**: For questions about who held a position at a given time:
     - Treat holding an office as a persistent state that starts at appointment and lasts until explicit end
     - Choose the most recent appointment before or at the query time that has no earlier termination
     - If no explicit end event is found before the query time, assume the office holder remains in position

5. **Entity-Event Relationships**: Analyze connections between entities and events:
   - **Agent relationships**: Who performs or causes an action
   - **Patient relationships**: Who/what is affected by an action
   - **Locative relationships**: Where something happens
   - **Attributive relationships**: Properties or characteristics at specific times

6. **Chunk Analysis**: Extract relevant information from the text chunks that supplements the events

7. **Cross-Reference**: Connect information between events and chunks to provide a comprehensive answer

# RESPONSE GUIDELINES
- **TEMPORAL RELEVANCE ASSESSMENT**: Assess whether evidence timestamps are relevant to the question's time scope
- Reference specific events (e.g., "Event #3") or chunks when supporting your answer
- If information is available in both events and chunks, prioritize the most specific and temporally relevant details
- Apply temporal logic: if an event occurred at time T1 and no change is mentioned by T2, assume continuity
- **If limited information exists for the queried time period**: 
  - State the temporal limitations and explain how nearby temporal information helps provide context
  - Make reasonable inferences based on:
    1. **Career Continuity**: If a person's career shows clear progression in a field/organization, infer likely continuity
    2. **Role Persistence**: If someone holds a position before and after the queried time, infer likely continued role
    3. **Institutional Affiliation**: If consistently associated with an institution, infer likely continued affiliation
  - Label inferences clearly and explain the reasoning chain
- Start with the direct answer followed by justification citing key events
- If the answer is based on inference, state: "Based on inference from surrounding evidence: [answer]" and explain the reasoning

# QUESTION
{question}

# EVENTS
{events_data}

# TEXT CHUNKS
{chunks_data}

# ANSWER
"""

PROMPTS["dynamic_QA_wo_timeline"] = """You are an AI assistant that answers questions based on relevant text chunks.

# TASK
Answer the question using the provided text chunks to provide additional context and details.

# CRITICAL TEMPORAL CONSTRAINT RULE
**NEVER use information from time periods that do not match the question's temporal scope.** If a question asks about a specific time period (e.g., "between 2000-2001"), only use information that occurred within or are explicitly valid for that exact time range. Using information from different time periods (e.g., 2009-2012 data for a 2000-2001 question) is a critical error.

# ANALYSIS APPROACH
1. **Query Semantics Understanding**: Analyze the semantic intent of the question:
   - **Existential queries**: Questions about what exists/happens at a specific time
   - **Causal queries**: Questions about causes, effects, or consequences over time
   - **Comparative queries**: Questions comparing states across different time periods
   - **Continuity queries**: Questions about ongoing states, processes, or relationships
   - **Boundary queries**: Questions about beginnings, endings, or transitions
   - **Aggregate queries**: Questions requiring synthesis across multiple time points

2. **Temporal Scope Identification**: FIRST identify the exact time constraints in the question:
   - Extract specific dates, time periods, or temporal references
   - Determine the precise temporal boundaries for valid evidence
   - Mark any temporal qualifiers (e.g., "during", "between", "before", "after")

3. **Evidence Time Filtering**: Before using any chunk as evidence:
   - **Verify temporal relevance**: Check that each piece of evidence falls within the question's time scope
   - **Reject out-of-scope evidence**: Discard any information from time periods outside the question's temporal constraints
   - **Flag temporal mismatches**: Identify when available evidence doesn't match the queried time period

4. **Chunk Analysis**: Extract relevant information from the text chunks

5. **Temporal Reasoning Strategies**: Apply sophisticated temporal logic:
   - **Forward chaining**: From past information, infer current states
   - **Backward chaining**: From current states, infer necessary past information
   - **Interval reasoning**: For queries about time periods, consider all relevant information within and bounding the interval
   - **Default persistence**: If X was true at time T1 and no change is mentioned by T2, assume X remains true at T2
   - **Temporal granularity matching**: Align the precision of your answer with the query's temporal specificity

# RESPONSE GUIDELINES
- **MANDATORY TEMPORAL VERIFICATION**: Before citing any evidence, verify its timestamps match the question's time scope
- Reference specific chunks when supporting your answer, but ONLY if they are from the correct time period
- Prioritize the most specific and relevant details from the correct time period
- Distinguish between definite facts and probable inferences
- **If no information exists for the queried time period**: Explicitly state "No information is available for the specified time period" rather than using irrelevant temporal data
- When uncertain, clearly state the limitations of the available information

# QUESTION
{question}

# TEXT CHUNKS
{chunks_data}

# ANSWER
"""


PROMPTS["dynamic_event_units"] = """Extract health events from this text as JSON.

Rules:
- Each event = one factual statement with explicit subject (no pronouns), time, and action
- "time": ISO-8601 date (YYYY-MM-DD, YYYY-MM, or YYYY). Use "static" only for permanent attributes
- "sentence": self-contained, retrievable statement with full subject name
- "context": optional, ≤80 tokens of background from same chunk
- Split multi-date sentences into separate events
- If no events found, return {"events": []}

Output JSON:
{"events": [{"sentence": "...", "context": "...", "time": "YYYY-MM-DD"}]}

Example:
Input: "[2024-01-15] Event: Annual_Checkup, Type: medical_visit, Start: 2024-01-15, Duration: 1 days"
Output: {"events": [{"sentence": "Patient had annual checkup on 2024-01-15.", "context": "Medical visit lasting 1 day.", "time": "2024-01-15"}]}

Text: {input_text}
Output:
"""
PROMPTS["time_entity_extraction"] = """You are an expert at analyzing queries to extract temporal constraints and named entities.

Given the following query: "{query}"

Extract:
1. Time constraints: Any temporal constraints mentioned in the query, including explicit dates, relative time references, or time periods.
2. Entities: Any named entities mentioned in the query that the user might be interested in.

Please return your analysis in JSON format with the following structure:
```json
{
  "time_constraints": {
    "start_time": "YYYY-MM-DD or null if not mentioned", 
    "end_time": "YYYY-MM-DD or null if not mentioned"
  },
  "entities": ["entity1", "entity2", ...]
}
```

Notes:
- For time constraints, normalize to ISO date formats (YYYY-MM-DD) when possible
- For single time points (like "in 2010" or "May 2023"):
  * If the context implies events AT or DURING this time, set both start_time and end_time to cover that specific period
  * If the context implies events AFTER this time, set start_time to this time and end_time to null
  * If the context implies events BEFORE this time, set start_time to null and end_time to this time
- For directional time expressions:
  * "after 2020" → set start_time: "2020-01-01", end_time: null
  * "before 2020" → set start_time: null, end_time: "2020-01-01"  
  * "since 2020" → set start_time: "2020-01-01", end_time: null
- For relative times (like "last week"), convert to absolute dates based on the current date
- For time expressions like "in the 1990s", use the appropriate start and end dates (1990-01-01 and 1999-12-31)
- For entities, extract proper nouns, organizations, people, and key concepts
- Use null for time values that aren't specified

Current date for reference: {current_date}

{examples}

JSON Response:
"""

PROMPTS["time_entity_extraction_examples"] = [
    """Example 1:
Query: "What happened to Apple stock between January 2020 and March 2021?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2020-01-01",
    "end_time": "2021-03-31"
  },
  "entities": ["Apple", "stock"]
}
```""",

    """Example 2:
Query: "Tell me about Microsoft's acquisitions in the last decade"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2013-01-01",
    "end_time": "2023-12-31"
  },
  "entities": ["Microsoft", "acquisitions"]
}
```""",

    """Example 3:
Query: "What were Tesla's major challenges in 2022?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2022-01-01",
    "end_time": "2022-12-31"
  },
  "entities": ["Tesla", "challenges"]
}
```""",

    """Example 4:
Query: "How did COVID-19 affect global economies?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2019-12-01",
    "end_time": null
  },
  "entities": ["COVID-19", "global economies"]
}
```""",

    """Example 5:
Query: "What happened after Google's IPO in August 2004?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2004-08-01", 
    "end_time": null
  },
  "entities": ["Google", "IPO"]
}
```""",

    """Example 6:
Query: "Tell me about events before the 2008 financial crisis."
JSON Response:
```json
{
  "time_constraints": {
    "start_time": null, 
    "end_time": "2008-09-15"
  },
  "entities": ["financial crisis"]
}
```""",

    """Example 7:
Query: "What innovations were made in May 2019?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2019-05-01", 
    "end_time": "2019-05-31"
  },
  "entities": ["innovations"]
}
```"""
]

PROMPTS[
    "event_continue_extraction"
] = """MANY events were missed in the last extraction. Add them below using the same JSON format with an "events" list:
"""

PROMPTS[
    "event_if_loop_extraction"
] = """It appears some events may have still been missed. Answer YES | NO if there are still events that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]



PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]
