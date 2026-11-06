Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #supported

# Reward should be anchored in a deterministic, programmatic check of the outcome wherever one can be written.

## Relations
informs:: [[QUE 02 - Designing a verifier so high reward means solved]]
supported-by:: [[EVD - PatTut says use verifiable signals whenever possible]], [[EVD - ART-E's golden answers make its reward close to a ground-truth verifier]], [[EVD - Reward is a deterministic outcome check 1 only if the final database equals the data]], [[EVD - CVE-Bench graded time-based SQL injection by whether the string SLEEP appeared in th]]

## Scope
Verifiable domains (code, SQL, DB state, golden answers). Non-verifiable tasks lack such an anchor.

## Related zettels
- [[verifiable-rule-based-rewards-prevent-reward-hacking]]
