# Prompt Refinement Harness

A minimal testing framework for evaluating and refining system prompts for multi-agent DBT chatbot domains (Mindfulness, Emotion Regulation, Distress Tolerance, Interpersonal Effectiveness).  
It runs test cases directly through AWS Bedrock (Claude 3.5 Sonnet) to ensure consistent, reproducible behavior.

---

## Overview

- **Global prompt:** Shared tone, empathy, and safety rules (`global_system_prompt.txt`)
- **Domain prompts:** Skill-specific logic in `prompts/<domain>/vN.yaml`
- **Test cases:** Defined in `eval/cases.yaml`
- **Outputs:** Logged automatically as `.csv`, `.md`, and `.json` in `eval/results/<timestamp>/`

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure AWS credentials
export AWS_PROFILE=prompt-refinement
export AWS_REGION=us-west-2
# Ensure your AWS profile has Bedrock invoke access


Project Structure 

global_system_prompt.txt
prompts/
  interpersonal/v1.yaml
  emotion/v1.yaml
eval/
  cases.yaml
  results/
manifest.yaml
harness.py


Output Location:

eval/results/<timestamp>/
  ├─ results.csv
  ├─ results.md
  └─ results.json

Contributing/Changing a Prompt:

# Create a new branch
git checkout -b feature/prompt-update

# Add or update prompts
nano prompts/interpersonal/v2.yaml

# Add test cases
nano eval/cases.yaml

# Run harness and check results
python harness.py --env dev
less eval/results/<timestamp>/results.md

# Commit and push
git add .
git commit -m "Updated interpersonal prompt v2"
git push origin feature/prompt-update

