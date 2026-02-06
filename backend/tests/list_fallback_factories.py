"""Identify unique fallback factories."""
import re

with open(r'C:\Users\vinic\Documents\Projetos\ChatKnime\generated_pipeline.py', encoding='utf-8') as f:
    code = f.read()

# Find all factory comments followed by fallback indicator
pattern = r'# Factory: (org\.knime\.[^\s\r\n]+)'
factories = re.findall(pattern, code)

# Find lines with FALLBACK
lines = code.split('\n')
fallback_factories = set()
for i, line in enumerate(lines):
    if 'FALLBACK' in line and i > 0:
        # Look back for factory
        for j in range(i, max(0, i-5), -1):
            match = re.search(r'Factory: (org\.knime\.[^\s]+)', lines[j])
            if match:
                fallback_factories.add(match.group(1))
                break

print(f"Total fallback factories: {len(fallback_factories)}")
print("\n".join(sorted(fallback_factories)))
