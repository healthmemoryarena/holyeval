# HolyEval Architecture

```mermaid
graph TB
    subgraph S1 [1. Synthetic User]
        direction TB
        SP[<b>Static Profile</b><br/>Age, Gender, Occupation<br/>Medical History, Allergies]
        DD[<b>Dynamic Data</b><br/>Current Symptoms, Vitals<br/>Medication, Sleep/Exercise]
        DB[<b>Dynamic Behavior</b><br/>Diet, Routine, Exercise<br/>Stress Events]
        SE[Synthetic Engine]
        VU((Virtual User))
        
        SP --> SE
        DD --> SE
        DB --> SE
        SE --> VU
    end

    subgraph S2 [2. Evaluation Framework]
        direction LR
        TA[TestAgent<br/><i>Acts as user</i>]
        TrA[TargetAgent<br/><i>System under test</i>]
        EA[EvalAgent<br/><i>Judge & Analyst</i>]
        
        TA <-- "Interaction Semantics<br/>Operation / Feedback" --> TrA
        TA -->|"Report Interaction"| EA
        TrA -->|"Report Logs"| EA
    end

    subgraph S3 [3. System Under Test]
        direction TB
        GUI[<b>GUI Engine</b><br/>Click, Swipe, Input<br/>Screenshot, Positioning]
        LLM[<b>LLM Engine</b><br/>API, Messaging<br/>Multi-turn Dialogue]
        
        Models[ChatGPT, Theta, Gemini, etc.]
        
        GUI --> Models
        LLM --> Models
    end

    subgraph S4 [4. Benchmark Generator]
        BA[<b>Business Abstraction</b><br/>Symptom Entry, Safety<br/>User Experience]
        MR[<b>Metric Recall</b><br/>Profile-based<br/>AI-generated]
        Opt[Iterative Optimization:<br/>Gen → Exec → Feedback → Opt]
        
        BA --> MR
        MR --> Opt
    end

    subgraph S5 [5. Benchmark Scheduler]
        BS[Scenario Manager]
        Sched[Schedule Execution]
        Agg[Aggregate Results]
        
        BS --> Sched
        Sched --> Agg
    end

    subgraph S6 [6. Benchmark Wrapper]
        Conv[Format Conversion]
        Ext[MedAgentBench, HealthBench,<br/>AgentClinic, etc.]
        
        Ext --> Conv
    end

    VU --> TA
    TrA --> GUI
    TrA --> LLM
    MR --> S5
    Conv --> S5
    S5 -->|"Schedule"| S2
    EA -->|"Results"| S5
```

## Component Descriptions

### 1. Synthetic User
Generates realistic user personas and behaviors.
- **Static Profile**: Demographics and historical data.
- **Dynamic Data**: Real-time state (symptoms, vitals).
- **Dynamic Behavior**: Lifestyle patterns and stressors.

### 2. Evaluation Framework
The core orchestration layer.
- **TestAgent**: Simulates the user's side of the conversation.
- **TargetAgent**: Interfaces with the system being evaluated.
- **EvalAgent**: Scores the interaction based on predefined metrics.

### 3. System Under Test (SUT)
The target model or application.
- **GUI Engine**: For testing visual interfaces.
- **LLM Engine**: For testing text-based APIs.

### 4. Benchmark Generator
Creates evaluation content.
- **Metric Recall**: Generates specific test cases based on business requirements.
- **Optimization**: Continuously improves benchmark quality.

### 5. Benchmark Scheduler
Manages the execution flow and result aggregation.

### 6. Benchmark Wrapper
Adapts external benchmarks (like HealthBench or AgentClinic) into the HolyEval format.
