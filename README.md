# threadscope
Bring Your Own Thread - Upload a CSV to visualize long-form interaction dynamics
Visualizes long-form interaction dynamics such as drift, stability, and perturbation response.

# Supported transcript format:
Transcripts that use explicit speaker markers, for example:

'You said: Hello'
'ChatGPT said: Hi there'

Marker strings are configurable in the UI, but the transcript must follow a consistent “marker → content” pattern.

# Not supported
- Arbitrarily formatted prose
- Chat logs without explicit speaker markers
- Exported HTML or JSON without preprocessing

# CSV converter can be found here: 
https://huggingface.co/spaces/Threadbourne/convert-to-csv

# CSV Converter Output: 
The output generates a CSV which includes the following columns:
- turn
- speaker
- chars
- words
- tokens_est (lightweight estimate)

# Privacy
Files are processed in-session only and are not stored.
 

Privacy
Files are processed in-session only and are not stored.
