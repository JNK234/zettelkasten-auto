
Created: 25-03-2026 19:34

Status: #baby 

Tags: [[enterprise-deployment]] [[compliance-testing]] [[production-requirements]]

# Enterprise Agent Evaluation Challenges

## Definition
A set of evaluation requirements unique to production enterprise deployments that academic benchmarks typically ignore: role-based access control, reliability guarantees (SLAs), long-horizon interactions spanning days or weeks, regulatory compliance (GDPR, HIPAA), and complete audit trails. These requirements fundamentally change what must be evaluated and how.

## Key Points
- Role-based access control: agents must respect organizational permissions—an HR bot shouldn't access financial data, analyst agents shouldn't modify production systems
- Reliability guarantees: enterprises need SLAs and pass^k metrics, not just pass@k—"works 90% of the time" is unacceptable for financial transactions
- Long-horizon interactions: enterprise workflows span days or weeks across hundreds of interactions, requiring persistent memory and consistency; most benchmarks test single sessions
- Compliance requirements: agents must follow regulations that vary by industry and jurisdiction (GDPR, HIPAA, SOX, PCI-DSS)—no academic benchmark tests for these
- Audit trails: every agent action must be logged, traceable, and explainable with reasoning, not just action logs—required for regulatory compliance and incident investigation
- Academic benchmarks assume full access, single sessions, no regulatory constraints, and focus on capability rather than operational reliability

## Connections
Represents the gap between research prototypes and production systems. Related to the broader software engineering principle that non-functional requirements (security, reliability, compliance) often dominate deployment costs. Highlights why benchmark performance doesn't predict production success.



# See Also

- [[data-leakage-in-static-llm-benchmarks]]
- [[state-verification-vs-output-verification]]
- [[inference-task-design-for-llm-evaluation]]
- [[pass-hat-k-reliability-metric]]
- [[human-in-the-loop-processes]]

# References

1. [[cale]]
