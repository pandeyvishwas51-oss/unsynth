# Security policy

UnSynth is a local-first text toolkit. Please do **not** open public issues
for vulnerabilities that could let an attacker:

- exfiltrate text sent to an optional cloud backend;
- execute code via a malicious plugin;
- read files outside a requested path.

Email the maintainers privately, or open a GitHub security advisory.

## What UnSynth will never do

- Ship a default cloud API key or phone-home detector.
- Claim cryptographic removal of a private-key watermark.
- Execute model-generated code.

## Plugin caution

`runtime.plugin_dirs` imports Python files. Treat that directory like
`PYTHONPATH`: only load code you trust.
